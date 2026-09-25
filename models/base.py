import threading
from abc import ABC, abstractmethod


# Cap buffered audio so a stuck/forgotten PTT can't grow memory without bound.
MAX_BUFFER_SECONDS = 120
MAX_BUFFER_SAMPLES = 16000 * MAX_BUFFER_SECONDS

# Chunks discarded while no model is loaded / PTT is not held: keeps the raw
# capture queue from growing unbounded (see MainWindow.process_audio_queue).
DRAIN_LIMIT = 512


class AudioBuffer:
    """Thread-safe mono float32 sample buffer shared by both STT engines.

    Appends accumulate while PTT is held; finalize() atomically takes the
    whole buffer so inference never runs under the lock, and oldest chunks
    are dropped once the cap is exceeded."""

    def __init__(self):
        self._lock = threading.Lock()
        self.chunks = []
        self.samples = 0

    def append(self, audio_data):
        if audio_data is None:
            return
        with self._lock:
            self.chunks.append(audio_data)
            self.samples += int(audio_data.shape[0])
            while self.samples > MAX_BUFFER_SAMPLES and len(self.chunks) > 1:
                self.samples -= int(self.chunks[0].shape[0])
                self.chunks.pop(0)

    def finalize(self):
        """Take the accumulated buffer (empty list when nothing was buffered)."""
        with self._lock:
            chunks = self.chunks
            self.chunks = []
            self.samples = 0
        return chunks

    def clear(self):
        with self._lock:
            self.chunks = []
            self.samples = 0


class STTModel(ABC):
    @abstractmethod
    def load(self):
        """Loads the model into memory."""
        pass

    @abstractmethod
    def unload(self):
        """Unloads the model to free resources."""
        pass

    @abstractmethod
    def process_audio(self, audio_data):
        """Processes a chunk of audio data.

        Args:
            audio_data: numpy array of audio samples

        Returns:
            str: Transcribed text, or None if no result causing an update yet.
        """
        pass

    @abstractmethod
    def finalize_transcription(self):
        """Called when PTT is released or phrase ends to force final transcription."""
        pass

    @abstractmethod
    def get_status(self):
        """Returns current status string (e.g., 'Loading', 'Ready', 'Error')."""
        pass
