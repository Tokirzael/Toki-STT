import copy
import json
import os

from .logger import logger


class ConfigManager:
    DEFAULT_CONFIG = {
        "audio": {
            "mic_id": None
        },
        "models": {
            "selected_model": "Phonon",  # Phonon or FasterWhisper
        },
        "osc": {
            "ip": "127.0.0.1",
            "port": 9000
        },
        "hotkeys": {
            "ptt_key": "v",                  # Default PTT
            "vr_ptt_button": 17,             # SteamVR Input API k_EVRButtonId_A (A=17, B=18, X=19, Y=20); rebind via "Record VR PTT"
            "vr_ptt_device": 0               # 0 = any controller, else tracked device index
        }
    }

    def __init__(self, config_file=None):
        if config_file is None:
            # Anchor to the project root so the app works from any working directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_file = os.path.join(base_dir, "settings.json")
        self.config_file = os.path.abspath(config_file)
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_file):
            logger.info("No config file found, creating default.")
            self.save_config(self.DEFAULT_CONFIG)
            # Deep-copy: callers may mutate the returned dict (set/_merge aliasing),
            # which must never leak back into DEFAULT_CONFIG.
            return copy.deepcopy(self.DEFAULT_CONFIG)
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                # Merge with default to ensure all keys exist.
                # Deep-copy the defaults: _merge_config aliases missing nested
                # sections, and a later set() would otherwise mutate DEFAULT_CONFIG.
                return self._merge_config(copy.deepcopy(self.DEFAULT_CONFIG), config)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # A corrupt file must not persist: back it up and fall back to the
            # defaults (which save_config will rewrite over the bad file), so a
            # single botched write can't lock the user out of their settings.
            try:
                os.replace(self.config_file, self.config_file + ".corrupt")
            except Exception as backup_error:
                logger.warning(f"Could not back up corrupt config: {backup_error}")
            # Rewrite a healthy defaults file immediately (don't wait for the
            # next set()) so restarts never see a missing or corrupt config.
            self.save_config(copy.deepcopy(self.DEFAULT_CONFIG))
            return copy.deepcopy(self.config)

    def _merge_config(self, default, current):
        """Recursive merge to ensure new keys in default are added to current config."""
        for key, value in default.items():
            if key not in current:
                current[key] = value
            elif isinstance(value, dict) and isinstance(current.get(key), dict):
                self._merge_config(value, current[key])
        return current

    def save_config(self, config=None):
        if config is not None:
            # Deep-copy so self.config never aliases DEFAULT_CONFIG (or any caller dict)
            # that a later set() would mutate.
            self.config = copy.deepcopy(config)
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info("Config saved.")
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def _set_key(self, key, value):
        keys = key.split('.')
        target = self.config
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value

    def set(self, key, value):
        self._set_key(key, value)
        self.save_config()

    def update(self, updates):
        """Apply multiple dotted-key updates with a single config save."""
        for key, value in updates.items():
            self._set_key(key, value)
        self.save_config()

# Global config instance
config_manager = ConfigManager()