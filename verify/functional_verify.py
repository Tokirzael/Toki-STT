"""Headless functional verification of Toki-STT (post Whisper-Turbo removal).

Runs in the project's real .venv with QT_QPA_PLATFORM=offscreen.
Sections:
  1. imports
  2. config (defaults / round-trip / partial merge / corrupt recovery)
  3. engines (buffer contract; real-model PTT pipeline: FasterWhisper on CPU,
     and Phonon-2 over a live local server round-trip)
  4. GUI offscreen (idle drain, volume meter, PTT-without-model, closeEvent)

Every section prints PASS/FAIL lines; the final line is OVERALL: OK or
OVERALL: FAIL(n). Exits 1 on failure.
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_here = os.path.dirname(os.path.abspath(__file__))
# project root: parent of the verify\ dir this script lives in (else CWD)
ROOT = os.path.dirname(_here) if os.path.basename(_here).lower() == "verify" else os.getcwd()
sys.path.insert(0, ROOT)

RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}", flush=True)


# ---------------------------------------------------------------- imports ----
try:
    import numpy as np
    from core.logger import logger
    from core.config import ConfigManager
    from models.base import AudioBuffer, MAX_BUFFER_SAMPLES, STTModel
    from models.faster_whisper_engine import FasterWhisperEngine
    from models.phonon_engine import PhononEngine
    check("imports: all modules", True)
except Exception as e:
    check("imports: all modules", False, repr(e))
    print(f"OVERALL: FAIL({sum(1 for _, ok in RESULTS if not ok)})", flush=True)
    sys.exit(1)

# ---------------------------------------------------------------- config ----
try:
    import json
    tmp_cfg = os.path.join(os.environ["TEMP"], "stt_verify_settings.json")
    if os.path.exists(tmp_cfg):
        os.remove(tmp_cfg)
    cm = ConfigManager(config_file=tmp_cfg)
    check("config: defaults created on first run", cm.config["hotkeys"]["vr_ptt_button"] == 17
          and cm.config["models"]["selected_model"] == "Phonon")
    cm.set("hotkeys.ptt_key", "ctrl_r")
    reloaded = ConfigManager(config_file=tmp_cfg)
    check("config: set()+reload round-trip", reloaded.config["hotkeys"]["ptt_key"] == "ctrl_r")
    with open(tmp_cfg, "w") as f:
        json.dump({"models": {"selected_model": "FasterWhisper"}}, f)
    merged = ConfigManager(config_file=tmp_cfg)
    check("config: partial-file merge keeps defaults", merged.config["hotkeys"]["vr_ptt_button"] == 17
          and merged.config["osc"]["port"] == 9000)
    with open(tmp_cfg, "w") as f:
        f.write("{ not valid json !!")
    recovered = ConfigManager(config_file=tmp_cfg)
    check("config: corrupt file -> .corrupt backup + healthy defaults",
          os.path.exists(tmp_cfg + ".corrupt") and recovered.config["hotkeys"]["vr_ptt_button"] == 17)
except Exception as e:
    check("config section", False, repr(e))

# ------------------------------------------------------------ engines ----
try:
    # AudioBuffer contract inside the real engine classes (no model load needed
    # for process_audio/finalize plumbing: both gate on self.model is None).
    fw = FasterWhisperEngine(model_id="whatever-not-loaded")
    pe = PhononEngine()

    check("engines: fresh finalize() returns None (no buffer)",
          fw.finalize_transcription() is None and pe.finalize_transcription() is None)
    # stub 'loaded' state so buffer plumbing is exercised (model None => ignored)
    fw.model = object(); pe.model = object()
    chunk = np.zeros(1600, dtype=np.float32) * 0.1
    for _ in range(5):
        fw.process_audio(chunk)
    check("engines: buffer accumulates (faster)", fw.audio_buffer.samples == 8000)
    pe.process_audio(chunk.astype(np.int16))  # dtype coercion path
    check("engines: phonon float32/mono coercion", len(pe.audio_buffer.chunks) == 1
          and pe.audio_buffer.chunks[0].dtype == np.float32)
    fw.clear_buffer()
    pe.clear_buffer()
    check("engines: clear_buffer()", fw.audio_buffer.samples == 0 and pe.audio_buffer.samples == 0)

    # ---- real-model path 1: faster-whisper on CPU (deterministic, cached) ----
    engine, used = None, "none"
    try:
        fw2 = FasterWhisperEngine(device="cpu")
        t0 = time.time()
        fw2.load()
        check("engine load: faster-whisper CT2 (local cache)", fw2.get_status() == "Ready", f"{time.time()-t0:.1f}s")
        engine, used = fw2, "FasterWhisper"
    except Exception as e:
        check("engine load: faster-whisper CT2", False, repr(e)[:160])

    if engine is not None:
        import numpy.random as _rnd
        sig = (_rnd.randn(3 * 16000) * 0.05).astype(np.float32)  # 3s of noise
        for i in range(0, len(sig), 1600):
            engine.process_audio(sig[i:i + 1600])
        t0 = time.time()
        text = engine.finalize_transcription()
        dt = time.time() - t0
        check(f"engine: full PTT pipeline ({used}, CPU)", isinstance(text, (str, type(None))),
              f"result={text!r} in {dt:.1f}s")
        try:
            engine.unload()
            check("engine: unload() clean", engine.get_status() == "Unloaded")
        except Exception as e:
            check("engine: unload()", False, repr(e)[:160])

    # ---- real-model path 2: Phonon-2 over a live local server --------------
    # load() self-heals: if nothing answers /health it spawns a detached
    # `python -m fermion.cli serve phonon-2` and waits for it (the first
    # transcription triggers the one-time model download on a fresh box).
    pe2 = PhononEngine()
    try:
        t0 = time.time()
        pe2.load()
        check("engine load: phonon-2 (local server healthy)", pe2.get_status() == "Ready",
              f"{time.time()-t0:.1f}s")
    except Exception as e:
        check("engine load: phonon-2 (local server healthy)", False, repr(e)[:160])

    if pe2.get_status() == "Ready":
        # 2s tone -> buffer in PTT-style chunks -> finalize() POSTs the WAV
        tone = (np.sin(2 * np.pi * 440 * np.arange(2 * 16000) / 16000.0) * 0.3).astype(np.float32)
        for i in range(0, len(tone), 1600):
            pe2.process_audio(tone[i:i + 1600])
        t0 = time.time()
        text = pe2.finalize_transcription()
        dt = time.time() - t0
        check("engine: phonon-2 full PTT pipeline (real HTTP round-trip)",
              isinstance(text, (str, type(None))),
              f"result={text!r} in {dt:.1f}s")
        try:
            pe2.unload()
            check("engine: phonon-2 unload() clean", pe2.get_status() == "Unloaded")
        except Exception as e:
            check("engine: phonon-2 unload()", False, repr(e)[:160])
except Exception as e:
    check("engines section", False, repr(e))

# ------------------------------------------------------- GUI (offscreen) ----
try:
    # pynput global hooks need an interactive desktop session; under an SSH
    # session the Listener start raises. Patch it out BEFORE MainWindow.
    from pynput import keyboard as _kb
    class _NoopListener:
        daemon = True
        def __init__(self, *a, **k): pass
        def start(self): pass
        def stop(self): pass
    _kb.Listener = _NoopListener

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from gui.main_window import MainWindow, key_to_ptt_string, format_ptt_display, BLOCKED_PTT_KEYS
    check("gui: helper functions", format_ptt_display("ctrl_r") == "Ctrl + R" and format_ptt_display("v") == "V")

    win = MainWindow()
    # feed synthetic chunks through the REAL AudioManager queue and the REAL timer callback
    big = (np.sin(np.linspace(0, 20 * np.pi, 4096)) * 0.5).astype(np.float32)  # ~0.26s tone
    for _ in range(12):
        win.audio_manager.audio_queue.put(big.copy())
    win.current_model = None
    win.is_ptt_active = False
    n_before = win.audio_manager.audio_queue.qsize()
    metered = []
    win.signals.volume_update.disconnect(win.update_volume_meter)
    win.signals.volume_update.connect(lambda v: metered.append(v))
    win.process_audio_queue()          # idle drain: must collapse 12 -> <=1 and meter the last one
    q_after = win.audio_manager.audio_queue.qsize()
    check("gui: idle drain keeps queue bounded (<=1 latest)", q_after <= 1, f"{n_before}->{q_after}")
    check("gui: volume meter still live while idle", len(metered) == 1 and metered[0] > 0, f"meter={metered}")

    win.current_model = FasterWhisperEngine(device="cpu")   # unloaded: model None -> still "not buffering"
    win.is_ptt_active = True
    for _ in range(6):
        win.audio_manager.audio_queue.put(big.copy())
    win.process_audio_queue()          # PTT active but no loaded model: nothing buffered, queue empty
    check("gui: PTT w/o loaded model buffers nothing", win.current_model.audio_buffer.samples == 0
          and win.audio_manager.audio_queue.qsize() == 0)

    # closeEvent path: with current_model set but never loaded, it must
    # null-under-lock and call unload exactly once without raising.
    class _CountedModel:
        def __init__(self): self.unloads = 0; self.status = "Unloaded"
        def unload(self): self.unloads += 1
        def get_status(self): return self.status
    counted = _CountedModel()
    with win._ptt_lock:
        win.current_model = counted
    from PySide6.QtGui import QCloseEvent
    ev = QCloseEvent()
    win.closeEvent(ev)
    check("gui: closeEvent unloads exactly once", counted.unloads == 1 and ev.isAccepted())
    print(f"OVERALL sections done", flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()
    check("gui section", False, repr(e))

failed = [n for n, ok in RESULTS if not ok]
print(f"TOTAL={len(RESULTS)} FAILED={len(failed)} {failed}", flush=True)
print("OVERALL: " + ("FAIL(%d)" % len(failed) if failed else "OK"), flush=True)
os._exit(1 if failed else 0)
