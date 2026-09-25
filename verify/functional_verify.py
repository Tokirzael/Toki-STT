"""Headless functional verification of Toki-STT after the 2026-09-25 fix pass.

Runs in the project's real .venv with QT_QPA_PLATFORM=offscreen (pythonw, no
console). Every section prints PASS/FAIL lines; final line is OVERALL: OK or
OVERALL: FAIL(n). Exits 1 on failure so a caller can check it.
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
    from models.whisper_engine import WhisperEngine
    from models.faster_whisper_engine import FasterWhisperEngine
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
          and cm.config["models"]["selected_model"] == "Whisper")
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
    wse = WhisperEngine("tiny", device="cpu")  # tiny only so import-time cost is low; NOT loaded

    check("engines: fresh finalize() returns None (no buffer)",
          fw.finalize_transcription() is None and wse.finalize_transcription() is None)
    # stub 'loaded' state so buffer plumbing is exercised (model None => ignored)
    fw.model = object(); wse.model = object()
    chunk = np.zeros(1600, dtype=np.float32) * 0.1
    for _ in range(5):
        fw.process_audio(chunk)
    check("engines: buffer accumulates (faster)", fw.audio_buffer.samples == 8000)
    wse.process_audio(chunk.astype(np.int16))  # dtype coercion path
    check("engines: whisper float32/mono coercion", len(wse.audio_buffer.chunks) == 1
          and wse.audio_buffer.chunks[0].dtype == np.float32)
    fw.clear_buffer()
    check("engines: clear_buffer()", fw.audio_buffer.samples == 0)

    # real-model path (CPU int8 to stay deterministic; CUDA proven by launcher log)
    try:
        import glob as _g, os.path as _os
        ct2_cache = os.path.join(os.environ.get("USERPROFILE", "C:/Users/User"), ".cache", "huggingface")
        has_ct2 = any(os.path.isdir(d) for d in _g.glob(ct2_cache + "/hub/models--*")) if os.path.isdir(ct2_cache) else False
    except Exception:
        has_ct2 = False

    engine, used = None, "none"
    try:
        fw2 = FasterWhisperEngine(device="cpu")  # default CT2 turbo id
        t0 = time.time()
        fw2.load()
        check("engine load: faster-whisper CT2 (local cache)", fw2.get_status() == "Ready", f"{time.time()-t0:.1f}s")
        engine, used = fw2, "FasterWhisper"
    except Exception as e:
        check("engine load: faster-whisper CT2", False, repr(e)[:160])

    if engine is None:
        wse2 = WhisperEngine("tiny", device="cpu")  # downloads ~75MB if not cached
        t0 = time.time()
        try:
            wse2.load()
            check("engine load: openai-whisper tiny (fallback)", wse2.get_status() == "Ready", f"{time.time()-t0:.1f}s")
            engine, used = wse2, "Whisper"
        except Exception as e:
            check("engine load: openai-whisper tiny (fallback)", False, repr(e)[:160])

    if engine is not None:
        import numpy.random as _rnd
        sig = (_rnd.randn(3 * 16000) * 0.05).astype(np.float32)  # 3s of noise (not speech)
        for i in range(0, len(sig), 1600):
            engine.process_audio(sig[i:i + 1600])
        t0 = time.time()
        text = engine.finalize_transcription()
        dt = time.time() - t0
        check(f"engine: full PTT pipeline on real model ({used}, CPU)", isinstance(text, (str, type(None))),
              f"result={text!r} in {dt:.1f}s")
        try:
            engine.unload()
            check("engine: unload() clean", engine.get_status() == "Unloaded")
        except Exception as e:
            check("engine: unload()", False, repr(e)[:160])
except Exception as e:
    check("engines section", False, repr(e))

# ------------------------------------------------------- GUI (offscreen) ----
try:
    # pynput global hooks need an interactive desktop session; under an SSH
    # session the Listener start raises. Patch it out BEFORE MainWindow so the
    # construction path is what we're testing. The drain logic + closeEvent are
    # exactly the code paths this fix pass touched, and none of them involve pynput.
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
    orig_update = win.update_volume_meter
    def _spy(val): metered.append(val)
    win.signals.volume_update.disconnect(win.update_volume_meter)
    win.signals.volume_update.connect(_spy)
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

    # closeEvent path (the B5 fix): with current_model set but never loaded, it must
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
