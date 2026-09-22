import threading
import time

from pynput import keyboard

from .logger import logger

try:
    # The "pyopenvr" project (github.com/cmbruns/pyopenvr), published on PyPI as "openvr".
    import openvr
    OPENVR_AVAILABLE = True
except ImportError:
    logger.warning("openvr module not found. VR input will be disabled.")
    OPENVR_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error importing openvr: {e}")
    OPENVR_AVAILABLE = False


# Friendly names for common SteamVR controller button IDs as reported in events.
_BUTTON_NAMES = {
    # SteamVR Input API standard button IDs (k_EVRButtonId)
    17: "A",
    18: "B",
    19: "X",
    20: "Y",
    21: "System / Menu",
    31: "Touchpad / Proximity",
    32: "Squeeze / Touchpad Click",
    33: "Trigger",
    34: "Trackpad Click",
    35: "Side Button 1",
    36: "Side Button 2",
    # Legacy controller-API style IDs (older runtimes)
    1: "System Button",
    2: "Y / Index A",
    3: "B / Trigger",
    4: "Thumbstick Click",
    5: "Grip",
    6: "X",
    7: "A (legacy)",
}


def button_name(button_id):
    """Human readable name for a SteamVR button ID (falls back to 'Button N')."""
    return _BUTTON_NAMES.get(button_id, f"Button {button_id}")


class InputManager:
    """Handles the global keyboard PTT hook using pynput."""

    def __init__(self, ptt_key='v', on_press_callback=None, on_release_callback=None):
        self.ptt_key = ptt_key
        self.on_press_callback = on_press_callback
        self.on_release_callback = on_release_callback
        self.listener = None
        self.is_ptt_held = False

    def start(self):
        if self.listener:
            self.stop()
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        logger.info(f"Keyboard listener started (PTT key: {self.ptt_key or 'none'}).")

    def stop(self):
        if self.listener:
            self.listener.stop()
            self.listener = None

    def set_ptt_key(self, ptt_key):
        """Change the PTT key at runtime (no listener restart needed)."""
        self.ptt_key = (ptt_key or "").strip()

    def _on_press(self, key):
        if not self.is_ptt_held and self._matches_key(key, self.ptt_key):
            self.is_ptt_held = True
            if self.on_press_callback:
                self.on_press_callback()

    def _on_release(self, key):
        if self.is_ptt_held and self._matches_key(key, self.ptt_key):
            self.is_ptt_held = False
            if self.on_release_callback:
                self.on_release_callback()

    def _matches_key(self, key_event, config_key):
        if not config_key:
            return False
        config_key = config_key.strip().lower()
        if not config_key:
            return False
        char = getattr(key_event, "char", None)
        if char and char.lower() == config_key:
            return True
        key_str = str(key_event).lower()
        return key_str == config_key or key_str == f"key.{config_key}"


class VRInputManager:
    """Handles OpenVR controller button input via event polling.

    Uses the "pyopenvr" project (github.com/cmbruns/pyopenvr), published on
    PyPI under the name "openvr". A background app receives controller button
    events from all tracked devices while SteamVR is running, even when another
    foreground app (e.g. VRChat) owns the headset.
    """

    # Debounce windows (seconds) for proximity-sensor flicker / accidental taps.
    MIN_PRESS_GAP = 0.20   # Ignore presses arriving this soon after the previous release
    MIN_HOLD_TIME = 0.10   # Ignore releases arriving this soon after a press (treat as tap)

    def __init__(self, on_press_callback=None, on_release_callback=None,
                 vr_ptt_button=17, on_vr_key_recorded=None):
        self.vr_system = None
        self.on_press_callback = on_press_callback
        self.on_release_callback = on_release_callback
        self.vr_ptt_button = vr_ptt_button
        # Only accept events from this tracked device index (0 = any controller)
        self.vr_ptt_device = 0
        self.on_vr_key_recorded = on_vr_key_recorded
        self.record_mode = None
        self.running = False
        self.thread = None
        self._press_emitted = {}
        self._press_start = {}
        self._last_release = {}

    def initialize(self):
        if not OPENVR_AVAILABLE:
            return False
        try:
            self.vr_system = openvr.init(openvr.VRApplication_Background)
            logger.info("OpenVR initialized successfully.")
            return True
        except openvr.error_code.InitError_Init_NoServerForBackgroundApp:
            logger.warning("SteamVR not running.")
            return False
        except Exception as e:
            logger.error(f"OpenVR Init failed: {e}")
            return False

    def start(self):
        """Start the polling thread, (re-)initializing OpenVR if needed.

        Returns True if the poll loop is running, False if OpenVR could not
        be initialized (e.g. SteamVR is not running).
        """
        if not self.vr_system:
            if not self.initialize():
                return False
        if self.is_running():
            return True
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()
        logger.info("VR Input polling started.")
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            self.thread = None
        if self.vr_system:
            openvr.shutdown()
            self.vr_system = None

    def is_running(self):
        return bool(self.running and self.thread is not None and self.thread.is_alive())

    def _poll_loop(self):
        if not OPENVR_AVAILABLE:
            return

        event = openvr.VREvent_t()

        while self.running:
            # Safety check inside loop
            vr_sys = self.vr_system
            if not vr_sys:
                break

            while vr_sys.pollNextEvent(event):
                if event.eventType == openvr.VREvent_ButtonPress:
                    self._handle_button(pressed=True, event=event)
                elif event.eventType == openvr.VREvent_ButtonUnpress:
                    self._handle_button(pressed=False, event=event)

            time.sleep(0.01)

    def _handle_button(self, pressed, event):
        button = event.data.controller.button
        device = event.trackedDeviceIndex
        now = time.time()

        # Record mode: the next button press becomes the new binding
        if self.record_mode:
            if pressed and self.on_vr_key_recorded:
                self.on_vr_key_recorded(button, device)
            self.record_mode = None
            return

        if button != self.vr_ptt_button:
            return
        if self.vr_ptt_device and device != self.vr_ptt_device:
            return

        if pressed:
            # Debounce: swallow rapid re-presses right after a release (proximity flicker)
            if now - self._last_release.get(button, 0.0) < self.MIN_PRESS_GAP:
                self._press_emitted[button] = False
                return
            self._press_emitted[button] = True
            self._press_start[button] = now
            if self.on_press_callback:
                self.on_press_callback()
        else:
            # Ignore the release of a press we never emitted
            if not self._press_emitted.get(button, False):
                return
            self._press_emitted[button] = False
            self._last_release[button] = now
            # Debounce: swallow very short taps
            if now - self._press_start.get(button, now) < self.MIN_HOLD_TIME:
                return
            if self.on_release_callback:
                self.on_release_callback()