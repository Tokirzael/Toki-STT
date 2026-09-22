import sounddevice as sd
import numpy as np
import threading
import time
import queue
from .logger import logger

class AudioManager:
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
        self.device_index = None
        # Rate-limit audio status warnings so a noisy device can't spam the log
        self._last_overflow_log = 0.0
        # Guards start/stop: the GUI thread (mic switch) and the model-load
        # worker thread can call these concurrently.
        self._lock = threading.Lock()

    def list_devices(self):
        """Returns a list of input devices."""
        devices = []
        try:
            device_list = sd.query_devices()
            for i, dev in enumerate(device_list):
                if dev['max_input_channels'] > 0:
                    devices.append({'id': i, 'name': dev['name']})
        except Exception as e:
            logger.error(f"Error listing devices: {e}")
        return devices

    def set_device(self, device_index):
        self.device_index = device_index

    def start_stream(self):
        with self._lock:
            if self.is_recording:
                return

            def callback(indata, frames, time_info, status):
                if status:
                    # Log each distinct overflow at most once per 5 seconds
                    now = time.monotonic()
                    if now - self._last_overflow_log >= 5.0:
                        self._last_overflow_log = now
                        logger.warning(f"Audio status: {status}")
                self.audio_queue.put(indata.copy())

            try:
                device_info = sd.query_devices(self.device_index, 'input')
                self.input_channels = device_info['max_input_channels']

                logger.info(f"Opening stream on device: {device_info['name']}")
                logger.info(f"Device Info: Native Channels={self.input_channels}, Default Rate={device_info['default_samplerate']}")
                logger.info(f"Stream Config: Channels={self.input_channels}, Rate={self.sample_rate}Hz")

                self.stream = sd.InputStream(
                    device=self.device_index,
                    channels=self.input_channels,
                    samplerate=self.sample_rate,
                    callback=callback,
                    blocksize=self.chunk_size
                )
                self.stream.start()
                self.is_recording = True
                logger.info("Audio stream started.")
            except Exception as e:
                logger.error(f"Failed to start audio stream: {e}")
                self.is_recording = False
                if self.stream is not None:
                    try:
                        self.stream.close()
                    except Exception:
                        pass
                    self.stream = None

    def stop_stream(self):
        with self._lock:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.is_recording = False
        # Drop buffered audio so stale chunks don't leak into the next session
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Audio stream stopped.")

    def get_audio_chunk(self):
        try:
            data = self.audio_queue.get_nowait()
            # Downmix to Mono if necessary
            if data.ndim > 1 and data.shape[1] > 1:
                # Average across channels to get mono
                data = np.mean(data, axis=1)
            
            # Ensure float32 (sounddevice usually returns float32 but good to be safe)
            if data.dtype != np.float32:
                data = data.astype(np.float32)
                
            return data
        except queue.Empty:
            return None
