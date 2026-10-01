"""Phonon-2 STT engine (FermionResearch/Phonon-2) - CPU, English only.

Drives a local `fermion serve` process (OpenAI-compatible HTTP on 127.0.0.1)
instead of loading a model in-process:

  - load()  self-heals the local server: if nothing answers /health it spawns
            `python -m fermion.cli serve phonon-2 --port <port>` DETACHED
            (CREATE_NO_WINDOW on Windows, so the GUI stays console-free) and
            waits for /health. The server's first request triggers the one-time
            ~164 MB model download; transcription requests are retried until a
            total deadline, which covers that window.
  - finalize_transcription()  renders the accumulated PTT buffer as a
            16 kHz mono WAV and POSTs it to /v1/audio/transcriptions.

The process is started with CREATE_NEW_PROCESS_GROUP so Ctrl-C on a console
session never forwards down to it; it keeps serving after the app exits
(systemd-style on the Linux side, a plain detached process here).
"""

import os
import subprocess
import threading
import time
import wave

import numpy as np

from core.logger import logger
from .base import STTModel, AudioBuffer

# Local-only endpoint; nothing leaves the machine.
PORT = 8900

# How long load() waits for a freshly spawned server (model load is ~5-30s).
SERVER_READY_TIMEOUT_S = 180
# Total wall-clock budget for finalize_transcription() (covers the server's
# first-request model download on a fresh machine).
TRANSCRIBE_DEADLINE_S = 240


class PhononEngine(STTModel):
    def __init__(self, port=PORT):
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.model = None
        self.status = "Unloaded"
        # 'cpu' once the server is confirmed serving (always, on this engine)
        self.loaded_device = None
        self._server_proc = None
        self._server_lock = threading.Lock()
        # Shared PTT buffer (same contract as the other engines)
        self.audio_buffer = AudioBuffer()

    # ------------------------------------------------------------------
    # server lifecycle
    # ------------------------------------------------------------------
    @property
    def _health_url(self):
        return f"{self.base_url}/health"

    @property
    def _transcribe_url(self):
        return f"{self.base_url}/v1/audio/transcriptions"

    def _health(self):
        import urllib.request
        try:
            with urllib.request.urlopen(self._health_url, timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def _find_server_log(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, "phonon_server.log")

    def _kill_port_squatter(self):
        """If a dead-end process holds self.port (a crashed prior server that
        died but left its listener, or a bind-loser), free the port so a fresh
        spawn can bind. Best-effort; runs only immediately before spawning."""
        try:
            import psutil
        except ImportError:
            return  # psutil not installed - skip; health gate still works
        try:
            for conn in psutil.net_connections(kind="tcp"):
                if conn.laddr and conn.laddr.port == self.port and \
                        conn.status == psutil.CONN_LISTEN and conn.pid:
                    try:
                        proc = psutil.Process(conn.pid)
                        logger.warning(f"Killing stale process holding port {self.port}: "
                                       f"{proc.pid} ({' '.join(proc.cmdline()[:4])})")
                        proc.kill()
                        proc.wait(timeout=5)
                    except Exception as e:
                        logger.warning(f"Could not kill port {self.port} holder: {e}")
        except Exception:
            pass  # psutil absent / no permission - the health gate still works

    def _ensure_server(self):
        """Idempotent: returns True when /health answers, spawning a detached
        `fermion serve` when it doesn't. Raises RuntimeError on failure."""
        with self._server_lock:
            if self._health():
                return True

            self._kill_port_squatter()
            import sys
            venv_py = sys.executable
            log_path = self._find_server_log()
            log_file = open(log_path, "ab")
            cmd = [venv_py, "-m", "fermion.cli", "serve", "phonon-2",
                   "--port", str(self.port)]
            logger.info(f"Phonon server not healthy - spawning: {' '.join(cmd)} (log: {log_path})")
            try:
                kwargs = {}
                if os.name == "nt":
                    # CREATE_NO_WINDOW is essential under pythonw.exe: a
                    # console child of a console-less parent would otherwise
                    # auto-allocate a visible console (a blank cmd window).
                    CREATE_NO_WINDOW = 0x08000000
                    CREATE_NEW_PROCESS_GROUP = 0x00000200
                    kwargs = dict(
                        creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
                        close_fds=True,
                        stdin=subprocess.DEVNULL,
                        stdout=log_file,
                        stderr=log_file,
                        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    )
                else:
                    kwargs = dict(
                        start_new_session=True,
                        close_fds=True,
                        stdin=subprocess.DEVNULL,
                        stdout=log_file,
                        stderr=log_file,
                    )
                self._server_proc = subprocess.Popen(cmd, **kwargs)
            except Exception as e:
                raise RuntimeError(f"failed to spawn the Phonon server: {e}")

            # Wait for the server to answer /health.
            deadline = time.monotonic() + SERVER_READY_TIMEOUT_S
            while time.monotonic() < deadline:
                if self._health():
                    logger.info("Phonon server is healthy.")
                    return True
                if self._server_proc is not None and \
                        self._server_proc.poll() is not None:
                    raise RuntimeError(
                        f"Phonon server process exited early (rc={self._server_proc.returncode}); "
                        f"see {log_path} for details"
                    )
                time.sleep(1)
            raise RuntimeError(
                f"Phonon server did not become healthy at {self._health_url} within "
                f"{SERVER_READY_TIMEOUT_S}s - see {log_path}"
            )

    # ------------------------------------------------------------------
    # STTModel interface
    # ------------------------------------------------------------------
    def load(self):
        try:
            self.status = "Loading..."
            logger.info("Loading Phonon-2 (CPU, English only) via local serve process...")
            self._ensure_server()
            # The server is the 'model'; set it so buffer gating matches
            # the other engines (finalize checks self.model is not None).
            self.model = {"kind": "phonon-2", "base_url": self.base_url}
            self.loaded_device = "cpu"
            self.status = "Ready"
            logger.info("Phonon-2 ready (server at %s)." % self.base_url)
        except Exception as e:
            self.status = "Error"
            logger.error(f"Failed to load Phonon-2 engine: {e}")
            raise

    def unload(self):
        # Intentionally leave the server running: it serves other callers and
        # restart is slow (model load). Mirrors the systemd pattern used on
        # the Linux side - the process outlives the app on purpose.
        self.model = None
        self.audio_buffer.clear()
        self.status = "Unloaded"
        logger.info("Phonon-2 engine unloaded (local server kept running).")

    def process_audio(self, audio_data):
        if self.model is None:
            return None
        if audio_data is not None:
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            self.audio_buffer.append(audio_data.flatten())
        return None

    def clear_buffer(self):
        self.audio_buffer.clear()
        logger.debug("Phonon buffer cleared.")

    def _wav_bytes(self, audio):
        """Render mono float32 [-1, 1] @ 16 kHz as 16-bit PCM WAV bytes."""
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16).tobytes()
        import io
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)
        return buf.getvalue()

    def finalize_transcription(self):
        """POST the accumulated PTT buffer to the local server (lock-free)."""
        if self.model is None:
            return None
        chunks = self.audio_buffer.finalize()
        if not chunks:
            return None
        audio = np.concatenate(chunks)
        if audio.ndim > 1:
            audio = audio.flatten()
        logger.info(f"Transcribing {len(audio)} samples via Phonon-2 (CPU)...")

        wav_bytes = self._wav_bytes(audio)
        import json
        import urllib.error
        import urllib.request
        import uuid

        boundary = uuid.uuid4().hex
        head = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode()
        body = head + wav_bytes + f"\r\n--{boundary}--\r\n".encode()

        deadline = time.monotonic() + TRANSCRIBE_DEADLINE_S
        attempt = 0
        last_err = None
        while time.monotonic() < deadline:
            attempt += 1
            try:
                req = urllib.request.Request(
                    self._transcribe_url,
                    data=body,
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=min(120, max(5, int(deadline - time.monotonic())))) as r:
                    text = json.loads(r.read().decode()).get("text", "").strip()
                    if text:
                        logger.info(f"Phonon-2 transcription: {text}")
                        return text
                    logger.info("Phonon-2 transcription: (empty)")
                    return None
            except Exception as e:
                last_err = e
                # Cold-cache first request: the server downloads the model
                # before responding, which can take minutes on a fresh box.
                if attempt == 1:
                    logger.warning(
                        "First Phonon request failed; server may be downloading "
                        f"the model. Retrying up to {TRANSCRIBE_DEADLINE_S}s. ({e})"
                    )
                time.sleep(2)
        logger.error(f"Phonon-2 transcription failed after retries: {last_err}")
        return None

    def get_status(self):
        return self.status
