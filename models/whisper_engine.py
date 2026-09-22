import numpy as np
import whisper
import torch
import threading
import gc
from core.logger import logger
from .base import STTModel

# Cap buffered audio so a stuck/forgotten PTT can't grow memory without bound
MAX_BUFFER_SECONDS = 120
MAX_BUFFER_SAMPLES = 16000 * MAX_BUFFER_SECONDS


class WhisperEngine(STTModel):
    def __init__(self, model_size="large-v3-turbo", device="cuda"):
        # "large-v3-turbo" is the explicit name in standard openai-whisper
        self.model_size = model_size
        self.device = device
        # Actual device the model ended up on (set by load(); may differ from
        # `self.device` when a CUDA request falls back to CPU)
        self.loaded_device = None
        self.model = None
        self.status = "Unloaded"
        # Mono float32 chunks; concatenated once in finalize_transcription()
        self.audio_buffer = []
        self._buffer_samples = 0
        self.lock = threading.Lock()

    def load(self):
        try:
            self.status = "Loading..."
            logger.info(f"Loading Standard Whisper model: {self.model_size} (device={self.device})")
            
            # Check availability of CUDA
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available. Falling back to CPU.")
                self.device = "cpu"

            self.model = whisper.load_model(self.model_size, device=self.device)
            self.loaded_device = self.device
            self.status = "Ready"
            logger.info("Whisper model loaded (OpenAI Standard).")
        except Exception as e:
            self.status = "Error"
            logger.error(f"Failed to load Whisper model: {e}")
            raise e

    def unload(self):
        if self.model:
            del self.model
            self.model = None
        
        # Cleanup torch memory
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error emptying CUDA cache: {e}")

        gc.collect()
        self.status = "Unloaded"
        logger.info("Whisper model unloaded.")

    def process_audio(self, audio_data):
        """Accumulates audio data."""
        # Match FasterWhisperEngine's contract: ignore audio while unloaded so a
        # stale buffer can't seed the first real PTT session.
        if self.model is None:
            return None
        with self.lock:
            if audio_data is not None:
                # Ensure float32 mono
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                audio_data = audio_data.flatten()
                self.audio_buffer.append(audio_data)
                self._buffer_samples += int(audio_data.shape[0])
                # Drop the oldest chunks once the buffer exceeds the cap
                while self._buffer_samples > MAX_BUFFER_SAMPLES and len(self.audio_buffer) > 1:
                    self._buffer_samples -= int(self.audio_buffer[0].shape[0])
                    self.audio_buffer.pop(0)
        return None

    def clear_buffer(self):
        with self.lock:
            self.audio_buffer = []
            self._buffer_samples = 0
            logger.debug("Whisper buffer cleared.")

    def finalize_transcription(self):
        """Transcribes the accumulated buffer (lock not held during inference)."""
        if self.model is None:
            return None

        with self.lock:
            if not self.audio_buffer:
                return None
            audio = np.concatenate(self.audio_buffer)
            self.audio_buffer = []
            self._buffer_samples = 0

        logger.info(f"Transcribing {len(audio)} samples...")
        try:
            # openai-whisper expects float32 tensor or numpy array
            result = self.model.transcribe(audio, fp16=(self.device == "cuda"), beam_size=5, language="en")
            text = result["text"].strip()
            logger.info(f"Transcription: {text}")
            return text or None
        except Exception as e:
            # Match FasterWhisperEngine's contract: log the failure and return
            # None instead of raising, so both engines behave the same to the GUI.
            logger.error(f"Standard Whisper Transcription Error: {e} (model type: {type(self.model).__name__})")
            return None

    def get_status(self):
        return self.status
