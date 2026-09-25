import numpy as np
import whisper
import torch
import gc
from core.logger import logger
from .base import STTModel, AudioBuffer


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
        self.audio_buffer = AudioBuffer()

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

        # Cleanup torch memory: the CUDA context is warm even when `device`
        # was rewritten to "cpu" by load()'s fallback, so key off loaded_device.
        if (self.loaded_device or self.device) == "cuda":
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
        if audio_data is not None:
            # Ensure float32 mono before it enters the shared buffer
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            self.audio_buffer.append(audio_data.flatten())
        return None

    def clear_buffer(self):
        self.audio_buffer.clear()
        logger.debug("Whisper buffer cleared.")

    def finalize_transcription(self):
        """Transcribes the accumulated buffer (lock not held during inference)."""
        if self.model is None:
            return None

        chunks = self.audio_buffer.finalize()
        if not chunks:
            return None
        audio = np.concatenate(chunks)

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
