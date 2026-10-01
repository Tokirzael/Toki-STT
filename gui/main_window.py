import threading
import time

import numpy as np

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QComboBox, QGroupBox, QLineEdit,
                               QRadioButton, QProgressBar, QTextEdit, QStatusBar,
                               QPushButton, QButtonGroup)
from PySide6.QtCore import Signal, QObject, Slot, QTimer

from core.logger import logger
from core.config import config_manager
from core.audio import AudioManager
from core.input import InputManager, VRInputManager, button_name
from core.osc_client import OSCController, CHATBOX_INPUT, CHATBOX_TYPING
from models.faster_whisper_engine import FasterWhisperEngine
from models.phonon_engine import PhononEngine

from .styles import DARK_THEME

# Silence (seconds) after the last sent message before clearing the VRChat chatbox
CHAT_CLEAR_DELAY = 10.0
# How long (ms) the user has to press a controller button during VR key recording
VR_RECORD_TIMEOUT_MS = 10000
# Same for keyboard PTT capture: the window during which one keystroke is awaited
KEY_RECORD_TIMEOUT_MS = 5000

# Keys that must never be captured as a PTT binding (Esc cancels; the rest are
# reserved so the user can't lock themselves out of the UI / this tool)
BLOCKED_PTT_KEYS = {"esc", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10",
                    "insert", "delete", "home", "end", "page_up", "page_down"}


def key_to_ptt_string(key):
    """Normalize a pynput keyboard event to the string form used by
    InputManager._matches_key (e.g., 'v', 'ctrl_r'). Returns None if the key is
    unsuitable as a PTT binding."""
    char = getattr(key, "char", None)
    if char:
        return char.lower()
    try:
        name = str(key).split(".", 1)[1]
    except (AttributeError, IndexError):
        return None
    if name in BLOCKED_PTT_KEYS:
        return None
    return name


def format_ptt_display(ptt_string):
    """Human-friendly label for the PTT field, e.g. 'ctrl_r' -> 'Ctrl+R'."""
    from pynput import keyboard as _kb
    try:
        _kb.Key[ptt_string]
        return " + ".join(part.capitalize() for part in ptt_string.split("_"))
    except KeyError:
        if len(ptt_string) == 1 and (97 <= ord(ptt_string) <= 122):
            return ptt_string.upper()
        return ptt_string


class WorkerSignals(QObject):
    """Signals for thread communication"""
    transcription_result = Signal(str)
    transcription_started = Signal()
    transcription_finished = Signal()
    status_update = Signal(str)
    volume_update = Signal(int)
    error_occurred = Signal(str)
    vr_key_recorded = Signal(int, int)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Toki-STT VRChat Transcription")
        self.resize(600, 700)
        self.setStyleSheet(DARK_THEME)

        self.signals = WorkerSignals()
        self.signals.transcription_result.connect(self.update_transcript)
        self.signals.transcription_started.connect(self._on_transcription_started)
        self.signals.transcription_finished.connect(self._on_transcription_finished)
        self.signals.status_update.connect(self.update_status)
        self.signals.volume_update.connect(self.update_volume_meter)
        self.signals.error_occurred.connect(self.handle_error)
        self.signals.vr_key_recorded.connect(self.handle_vr_key_recorded)

        # Core Components
        self.audio_manager = AudioManager()
        self.input_manager = InputManager(
            on_press_callback=self.handle_input_press,
            on_release_callback=self.handle_input_release
        )

        def _on_vr_key_recorded_callback(button_id, device_id):
            self.signals.vr_key_recorded.emit(button_id, device_id)

        self.vr_input_manager = VRInputManager(
            on_press_callback=self.handle_input_press,
            on_release_callback=self.handle_input_release,
            on_vr_key_recorded=_on_vr_key_recorded_callback
        )
        self.osc_controller = OSCController()

        self.current_model = None
        # Single logical PTT state shared by keyboard and VR inputs
        self.is_ptt_active = False
        # Guards is_ptt_active/current_model against concurrent keyboard + VR
        # callbacks, background transcription and model load/unload threads.
        self._ptt_lock = threading.Lock()
        self._transcribe_thread = None
        self._pending_finalization = False
        self._unload_thread = None

        # Chatbox auto-clear state
        self._last_final_sent = 0.0
        self._chat_cleared = True
        self._record_timer = None

        # GUI Setup
        self.setup_ui()
        self.load_settings()

        # Start Services
        self.osc_controller.start()
        self.input_manager.start()
        if self.vr_input_manager.initialize():
            self.vr_input_manager.start()

        # Audio Processing Timer (polling queue)
        self.audio_timer = QTimer()
        self.audio_timer.timeout.connect(self.process_audio_queue)
        self.audio_timer.start(50) # 20fps check

        # Chatbox auto-clear timer (PTT mode)
        self.chat_clear_timer = QTimer()
        self.chat_clear_timer.timeout.connect(self._check_chat_clear)
        self.chat_clear_timer.start(1000)

        # Initial Model Load is manual via the Load/Unload button
        self.update_status("Select a model and click 'Load Model'.")

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Audio Settings ---
        audio_group = QGroupBox("Audio Input")
        audio_layout = QVBoxLayout()

        self.mic_combo = QComboBox()
        self.refresh_mics()
        self.mic_combo.currentIndexChanged.connect(self.change_mic)

        audio_layout.addWidget(QLabel("Microphone:"))
        audio_layout.addWidget(self.mic_combo)

        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(0, 100)
        self.volume_bar.setValue(0)
        audio_layout.addWidget(QLabel("Input Level:"))
        audio_layout.addWidget(self.volume_bar)

        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)

        # --- Model Settings ---
        model_group = QGroupBox("STT Engine")
        model_layout = QVBoxLayout()

        self.model_group_btn = QButtonGroup(self)
        # Faster Whisper: GPU-only (CT2 float16 on CUDA) - the VRAM figure is
        # measured on this box (RTX 4090); the CT2 build allocates outside
        # torch's tracker, so it shows up in nvidia-smi but not in
        # torch.cuda.memory_allocated().
        # Phonon-2: runs on CPU/RAM by design (FermionResearch/Phonon-2,
        # ~164 MB download, English only) - a local 'fermion serve' process
        # answers each PTT release over loopback HTTP.
        self.rb_faster = QRadioButton("Faster Whisper (PTT)  —  turbo CT2 float16, ~2 GB VRAM (GPU only)")
        self.rb_phonon = QRadioButton("Phonon-2 (PTT)  —  CPU/RAM, English only (~164 MB model, local server)")
        self.model_group_btn.addButton(self.rb_faster)
        self.model_group_btn.addButton(self.rb_phonon)

        model_layout.addWidget(self.rb_faster)
        model_layout.addWidget(self.rb_phonon)

        # Load/Unload Button
        self.btn_load_model = QPushButton("Load Model")
        self.btn_load_model.clicked.connect(self.toggle_model_load)
        model_layout.addWidget(self.btn_load_model)

        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # --- Input Settings ---
        input_group = QGroupBox("Controls (Push-To-Talk)")
        input_layout = QVBoxLayout()

        hk_layout = QHBoxLayout()
        self.txt_ptt_key = QLineEdit()
        self.txt_ptt_key.setReadOnly(True)
        hk_layout.addWidget(QLabel("Keyboard PTT:"))
        hk_layout.addWidget(self.txt_ptt_key)

        # Both Record buttons share one width so the two rows stay visually
        # aligned (Qt sizes each button to its own hint by default, which left
        # "Record VR PTT" ~2x wider than "Record PTT").
        self.btn_record_key = QPushButton("Record PTT")
        self.btn_record_key.clicked.connect(self.record_key)
        hk_layout.addWidget(self.btn_record_key)
        input_layout.addLayout(hk_layout)

        vr_layout = QHBoxLayout()
        self.lbl_vr_ptt = QLabel("VR PTT: -")
        self.btn_rec_vr_ptt = QPushButton("Record VR PTT")
        self.btn_rec_vr_ptt.clicked.connect(lambda: self.record_vr_key("PTT"))
        # Both Record buttons share one width so the two rows stay visually
        # aligned. Measured from QFontMetrics with the stylesheet's actual font
        # (Segoe UI 14px bold + 8/16px padding) — sizeHint() at this point is
        # measured before that font applies and came out ~50% too narrow.
        from PySide6.QtGui import QFont, QFontMetrics
        _btn_font = QFont("Segoe UI")
        _btn_font.setPixelSize(14)
        _btn_font.setBold(True)
        _fm = QFontMetrics(_btn_font)
        _text_w = max(_fm.horizontalAdvance(self.btn_record_key.text()),
                      _fm.horizontalAdvance(self.btn_rec_vr_ptt.text()))
        shared_width = _text_w + 32 + 12   # padding (16px/side) + breathing room
        self.btn_record_key.setFixedWidth(shared_width)
        self.btn_rec_vr_ptt.setFixedWidth(shared_width)
        vr_layout.addWidget(self.lbl_vr_ptt)
        vr_layout.addStretch(1)
        vr_layout.addWidget(self.btn_rec_vr_ptt)
        input_layout.addLayout(vr_layout)

        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        # --- OSC Settings ---
        osc_group = QGroupBox("VRChat OSC")
        osc_layout = QHBoxLayout()

        self.txt_osc_ip = QLineEdit("127.0.0.1")
        self.txt_osc_port = QLineEdit("9000")

        osc_layout.addWidget(QLabel("IP:"))
        osc_layout.addWidget(self.txt_osc_ip)
        osc_layout.addWidget(QLabel("Port:"))
        osc_layout.addWidget(self.txt_osc_port)

        btn_update_osc = QPushButton("Apply OSC")
        btn_update_osc.clicked.connect(self.update_osc)
        osc_layout.addWidget(btn_update_osc)
        osc_group.setLayout(osc_layout)
        main_layout.addWidget(osc_group)

        # --- Output Preview ---
        preview_group = QGroupBox("Transcript Preview")
        preview_layout = QVBoxLayout()
        self.txt_preview = QTextEdit()
        self.txt_preview.setReadOnly(True)
        preview_layout.addWidget(self.txt_preview)
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("Initializing...")

    def refresh_mics(self):
        # Block signals while populating so addItem() doesn't trigger
        # change_mic() (and start the audio stream) mid-refresh.
        self.mic_combo.blockSignals(True)
        try:
            devices = self.audio_manager.list_devices()
            self.mic_combo.clear()
            for dev in devices:
                self.mic_combo.addItem(dev['name'], dev['id'])
        finally:
            self.mic_combo.blockSignals(False)

    def load_settings(self):
        config = config_manager.config

        # Set Model (legacy "Whisper" / "WhisperTurbo" selections migrate to
        # FasterWhisper, the GPU engine that replaced the old openai-whisper)
        selected = config['models']['selected_model']
        if selected == "Phonon":
            self.rb_phonon.setChecked(True)
        else:  # FasterWhisper, and anything legacy -> the GPU engine
            self.rb_faster.setChecked(True)

        # Set Mic (block signals so the stream is started only once, below,
        # on the finally selected device)
        saved_mic = config['audio']['mic_id']
        self.mic_combo.blockSignals(True)
        try:
            if saved_mic is not None:
                index = self.mic_combo.findData(saved_mic)
                if index >= 0:
                    self.mic_combo.setCurrentIndex(index)
        finally:
            self.mic_combo.blockSignals(False)
        # Start the capture stream on the selected device
        self.change_mic()

        # Set PTT Key (None-safe for fresh installs)
        ptt_key = config['hotkeys'].get('ptt_key') or ""
        self.txt_ptt_key.setText(format_ptt_display(ptt_key) if ptt_key else "")
        self.input_manager.set_ptt_key(ptt_key)

        # Set VR PTT (defaults mirror ConfigManager.DEFAULT_CONFIG - the merge
        # guarantees these keys exist; .get() is only a belt-and-braces fallback)
        vr_ptt_btn = config['hotkeys'].get('vr_ptt_button', 17)
        vr_ptt_dev = config['hotkeys'].get('vr_ptt_device', 0)
        self.lbl_vr_ptt.setText(f"VR PTT: {button_name(vr_ptt_btn)}")
        self.vr_input_manager.vr_ptt_button = vr_ptt_btn
        self.vr_input_manager.vr_ptt_device = vr_ptt_dev

        # Set OSC
        self.txt_osc_ip.setText(config['osc']['ip'])
        self.txt_osc_port.setText(str(config['osc']['port']))
        # Apply the saved endpoint to the controller as well, so a non-local
        # VRChat host works without having to click "Apply OSC" first.
        try:
            osc_port = int(config['osc']['port'])
        except (TypeError, ValueError):
            osc_port = 9000
        self.osc_controller.update_settings(config['osc']['ip'], osc_port)

    def toggle_model_load(self):
        if self.current_model:
            self.unload_model()
        else:
            model_type = "Phonon" if self.rb_phonon.isChecked() else "FasterWhisper"
            self.load_model(model_type)

    def load_model(self, model_type):
        self.update_status(f"Loading {model_type}...")
        self.btn_load_model.setEnabled(False)
        self.rb_faster.setEnabled(False)
        self.rb_phonon.setEnabled(False)

        # Lock input settings
        self.txt_ptt_key.setEnabled(False)
        self.btn_record_key.setEnabled(False)
        self.btn_rec_vr_ptt.setEnabled(False)

        threading.Thread(target=self._load_model_thread, args=(model_type,), daemon=True).start()

    def _load_model_thread(self, model_type):
        engine = None
        try:
            if model_type == "Phonon":
                engine = PhononEngine()
            else:
                # GPU-only: float16 CT2 on CUDA, default large-v3-turbo repo
                engine = FasterWhisperEngine(device="cuda")

            engine.load()

            # Start audio stream after model is loaded
            self.audio_manager.start_stream()
            if not self.audio_manager.is_recording:
                raise RuntimeError("Audio stream failed to start - check your microphone selection.")

            # Only publish the engine once it is fully usable; on failure the
            # previous state (no model) is left untouched.
            with self._ptt_lock:
                self.current_model = engine

            config_manager.set('models.selected_model', model_type)
            # Signal UI (thread-safe via status_update signal)
            self.signals.status_update.emit("MODEL_LOAD_COMPLETE")
        except Exception as e:
            # Free resources if the engine loaded but something downstream failed
            if engine is not None:
                try:
                    engine.unload()
                except Exception as cleanup_error:
                    logger.warning(f"Error cleaning up failed model load: {cleanup_error}")
            self.signals.error_occurred.emit(str(e))
            self.signals.status_update.emit("MODEL_LOAD_ERROR")

    def unload_model(self):
        # Wait for an in-flight transcription so the model isn't deleted mid-inference
        self._wait_for_transcription()
        if not self.current_model:
            return
        self.update_status(f"Unloading {self.current_model.__class__.__name__}...")
        self.btn_load_model.setEnabled(False)
        # Run the (blocking) teardown off the GUI thread so the UI stays responsive
        self._unload_thread = threading.Thread(target=self._unload_model_thread, daemon=True)
        self._unload_thread.start()

    def _unload_model_thread(self):
        with self._ptt_lock:
            model = self.current_model
            self.current_model = None
        try:
            if model:
                model.unload()
        except Exception as e:
            logger.error(f"Error unloading model: {e}")

        # NOTE: the audio stream intentionally keeps running after an unload so
        # the input level meter stays live; it is only stopped in closeEvent().

        import gc
        gc.collect()

        self.signals.status_update.emit("MODEL_UNLOADED")

    def _wait_for_transcription(self, timeout=120.0):
        """Block until any in-flight background transcription has finished.

        `timeout` bounds the wait so a stuck inference can't freeze the UI
        (or the window close) forever.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._ptt_lock:
                worker = self._transcribe_thread
            if worker is None:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(f"Timed out after {timeout}s waiting for transcription to finish.")
                return
            worker.join(timeout=min(0.5, remaining))

    def _reset_input_controls(self):
        """Re-enable the controls that are locked while a model is loaded/loading."""
        self.rb_faster.setEnabled(True)
        self.rb_phonon.setEnabled(True)
        self.txt_ptt_key.setEnabled(True)
        self.btn_record_key.setEnabled(True)
        self.btn_rec_vr_ptt.setEnabled(True)
        self.btn_load_model.setEnabled(True)

    def _device_label(self, engine):
        """Where the model actually lives ('cpu'/'cuda') after load."""
        d = getattr(engine, "loaded_device", None) or getattr(engine, "device", "?")
        return d.upper() if d in ("cpu", "cuda") else str(d).upper()

    def change_mic(self):
        idx = self.mic_combo.currentData()
        if idx is None:
            return
        if idx == self.audio_manager.device_index:
            return  # already using this device
        self.audio_manager.stop_stream()
        self.audio_manager.set_device(idx)
        self.audio_manager.start_stream()
        config_manager.set('audio.mic_id', idx)

    def record_key(self):
        """Open a short global capture window; the next keystroke becomes the PTT key."""
        from pynput import keyboard as _kb

        captured = {}

        def on_press(key):
            # pynput quirk: listener.stop() blocks until callbacks finish, so it
            # must never be called from inside a callback. Returning False exits
            # the loop; _finish() (GUI thread) calls stop() afterwards.
            normalized = key_to_ptt_string(key)
            if str(key) == "key.esc":
                captured["cancelled"] = True  # Esc cancels the capture window
                return False
            if normalized is None:
                return True  # ignored key (blocked set / modifier-only) - keep listening
            captured["ptt"] = normalized
            return False

        listener = _kb.Listener(on_press=on_press, suppress=False)
        listener.daemon = True
        listener.start()

        self.btn_record_key.setEnabled(False)
        self.update_status("Press the key to use as PTT (Esc cancels)...")

        def _finish():
            if "ptt" not in captured:
                self.btn_record_key.setEnabled(True)
                self.update_status("PTT recording timed out - no key detected.")
                return
            ptt = captured["ptt"]
            # Mirror the VR PTT flow: capture updates the field, apply + persist.
            self.txt_ptt_key.setText(format_ptt_display(ptt))
            self.input_manager.set_ptt_key(ptt)
            config_manager.set('hotkeys.ptt_key', ptt)
            self.btn_record_key.setEnabled(True)
            self.update_status(f"PTT bound: {format_ptt_display(ptt)}")

        def _run():
            listener.stop()
            QTimer.singleShot(0, _finish)  # hop back to the GUI thread

        QTimer.singleShot(KEY_RECORD_TIMEOUT_MS, _run)

    def record_vr_key(self, mode):
        # (Re)initialize on demand so SteamVR starting after app launch works
        if not self.vr_input_manager.start():
            self.update_status("SteamVR not running - start SteamVR, then try again.")
            return
        self.vr_input_manager.record_mode = mode
        self.btn_rec_vr_ptt.setEnabled(False)
        self.update_status("Press a VR Controller button to bind VR PTT...")
        # Timeout: cancel the recording window after a few seconds
        self._record_timer = QTimer(self)
        self._record_timer.setSingleShot(True)
        self._record_timer.timeout.connect(self._cancel_vr_record)
        self._record_timer.start(VR_RECORD_TIMEOUT_MS)

    def _cancel_vr_record(self):
        self.vr_input_manager.record_mode = None
        self.btn_rec_vr_ptt.setEnabled(True)
        self.update_status("VR key recording timed out - no button detected.")

    @Slot(int, int)
    def handle_vr_key_recorded(self, button_id, device_id):
        if self._record_timer is not None:
            self._record_timer.stop()
        self.lbl_vr_ptt.setText(f"VR PTT: {button_name(button_id)}")
        self.vr_input_manager.vr_ptt_button = button_id
        self.vr_input_manager.vr_ptt_device = device_id
        config_manager.update({
            "hotkeys.vr_ptt_button": button_id,
            "hotkeys.vr_ptt_device": device_id,
        })
        self.btn_rec_vr_ptt.setEnabled(True)
        self.update_status(f"VR PTT bound: {button_name(button_id)}")

    def update_osc(self):
        ip = self.txt_osc_ip.text().strip()
        try:
            port = int(self.txt_osc_port.text().strip())
        except ValueError:
            self.update_status("Invalid OSC port - enter a number (e.g. 9000).")
            return
        if not ip or not (0 < port < 65536):
            self.update_status("Invalid OSC settings - check the IP and port.")
            return
        self.osc_controller.update_settings(ip, port)
        config_manager.update({
            "osc.ip": ip,
            "osc.port": port,
        })
        self.update_status("OSC Settings Updated")

    def handle_input_press(self):
        # Called when PTT is pressed (keyboard or VR)
        with self._ptt_lock:
            model = self.current_model
            if model is None:
                return
            if self.is_ptt_active:
                return  # PTT already held (e.g. keyboard + VR at the same time)
            self.is_ptt_active = True

        self.signals.status_update.emit("Listening...")
        # Clear buffer to start fresh (the engine logs the clear itself)
        if hasattr(model, 'clear_buffer'):
            model.clear_buffer()
        self.osc_controller.send_message(CHATBOX_TYPING, True, immediate=True)

    def handle_input_release(self):
        # Called when PTT is released (keyboard or VR)
        with self._ptt_lock:
            if not self.is_ptt_active:
                return
            self.is_ptt_active = False

            if self._transcribe_thread is not None and self._transcribe_thread.is_alive():
                # A transcription is still running; queue this utterance to run
                # right after it instead of dropping or double-processing it.
                self._pending_finalization = True
                return

            model = self.current_model
            if model is None:
                # Model was unloaded while PTT was held; nothing to transcribe
                self.signals.status_update.emit("Ready")
                return

            # Transcribe off the input thread so the UI and PTT stay responsive
            self._transcribe_thread = threading.Thread(
                target=self._run_transcription, args=(model,), daemon=True
            )
            self._transcribe_thread.start()

        self.signals.status_update.emit("Transcribing...")
        self.signals.transcription_started.emit()
        self.osc_controller.send_message(CHATBOX_TYPING, False, immediate=True)

    def _run_transcription(self, model):
        """Background worker: transcribe the held-PTT buffer for `model`."""
        try:
            text = model.finalize_transcription()
            if text:
                self.signals.transcription_result.emit(text)
            else:
                self.signals.status_update.emit("Ready")
        except Exception as e:
            logger.error(f"Transcription Failed: {e}")
            self.signals.error_occurred.emit(f"Transcription Failed: {e}")
        finally:
            started_next = False
            with self._ptt_lock:
                self._transcribe_thread = None
                if self._pending_finalization:
                    self._pending_finalization = False
                    pending_model = self.current_model
                    if pending_model is not None:
                        self._transcribe_thread = threading.Thread(
                            target=self._run_transcription, args=(pending_model,), daemon=True
                        )
                        self._transcribe_thread.start()
                        started_next = True
            if started_next:
                # Chain continues: keep the Load button disabled
                self.signals.transcription_started.emit()
            else:
                self.signals.transcription_finished.emit()

    @Slot()
    def _on_transcription_started(self):
        # Disable loading/unloading while a transcription is in flight
        self.btn_load_model.setEnabled(False)

    @Slot()
    def _on_transcription_finished(self):
        self.btn_load_model.setEnabled(True)

    def process_audio_queue(self):
        # While nothing will buffer (no model loaded or PTT not held) keep only
        # the LATEST chunk from the queue: the input-level meter stays live,
        # queue RAM stays bounded on a stalled GUI thread, and stale audio can't
        # seed the next PTT session. With PTT active every chunk is kept.
        chunks = []
        buffering = bool(self.current_model and self.is_ptt_active)
        while True:
            chunk = self.audio_manager.get_audio_chunk()
            if chunk is None:
                break
            if not buffering:
                chunks = [chunk]
            else:
                chunks.append(chunk)

        for chunk in chunks:
            # Calculate volume
            rms = np.sqrt(np.mean(chunk**2))
            val = int(rms * 500)
            self.signals.volume_update.emit(min(val, 100))

            # Log signal strength occasionally (every 500 chunks to avoid spam)
            if not hasattr(self, '_audio_log_counter'):
                self._audio_log_counter = 0
            self._audio_log_counter += 1
            if self._audio_log_counter % 500 == 0:
                logger.debug(f"Audio Signal RMS: {rms:.6f} (Queue Processed)")

            # Buffer audio while PTT is held (Whisper / FasterWhisper)
            if buffering:
                self.current_model.process_audio(chunk)

    def _check_chat_clear(self):
        """Clears the VRChat chatbox once CHAT_CLEAR_DELAY seconds have passed
        since the last sent message."""
        if self._chat_cleared or not self._last_final_sent:
            return
        if time.time() - self._last_final_sent >= CHAT_CLEAR_DELAY:
            self.osc_controller.send_message(CHATBOX_INPUT, "")
            self._chat_cleared = True

    @Slot(str)
    def update_transcript(self, text):
        if not text:
            return
        self.txt_preview.append(f">> {text}")
        self._trim_preview()
        self.osc_controller.send_message(CHATBOX_INPUT, text)
        self._last_final_sent = time.time()
        self._chat_cleared = False
        self.update_status("Sent to VRChat")

    def _trim_preview(self, max_blocks=500, trim_blocks=100):
        """Keep the transcript preview from growing without bound in long sessions."""
        doc = self.txt_preview.document()
        if doc.blockCount() <= max_blocks:
            return
        cursor = QTextCursor(doc)
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        for _ in range(trim_blocks):
            cursor.movePosition(QTextCursor.MoveOperation.Down, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()

    @Slot(str)
    def update_status(self, text):
        if text == "MODEL_LOAD_COMPLETE":
            self.btn_load_model.setText("Unload Model")
            self.btn_load_model.setEnabled(True)
            device = self._device_label(self.current_model)
            self.status_bar.showMessage(f"Model Loaded: {self.current_model.__class__.__name__} on {device}")
        elif text == "MODEL_LOAD_ERROR":
            self.btn_load_model.setText("Load Model")
            self._reset_input_controls()
            self.status_bar.showMessage("Error Loading Model")
        elif text == "MODEL_UNLOADED":
            self.btn_load_model.setText("Load Model")
            self._reset_input_controls()
            self.status_bar.showMessage("Model unloaded. Select a model.")
        else:
            self.status_bar.showMessage(text)

    @Slot(int)
    def update_volume_meter(self, val):
        self.volume_bar.setValue(val)

    @Slot(str)
    def handle_error(self, message):
        self.update_status(f"Error: {message}")
        logger.error(f"GUI Error: {message}")

    def closeEvent(self, event):
        # Stop polling before tearing down services so nothing new is queued
        self.audio_timer.stop()
        self.chat_clear_timer.stop()
        # Let in-flight transcription / unload finish before deleting the model
        self._wait_for_transcription()
        if self._unload_thread is not None and self._unload_thread.is_alive():
            self._unload_thread.join(timeout=30)
        self.audio_manager.stop_stream()
        self.input_manager.stop()
        self.vr_input_manager.stop()
        self.osc_controller.stop()
        # Take ownership of the model under the same lock _unload_model_thread
        # uses, so a teardown that finished mid-closeEvent is never unloaded twice:
        # whoever nulls it first owns the (blocking) unload.
        with self._ptt_lock:
            leftover = self.current_model
            self.current_model = None
        if leftover:
            try:
                leftover.unload()
            except Exception as e:
                logger.error(f"Error unloading model on close: {e}")
        event.accept()
