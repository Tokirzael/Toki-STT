import gc

import numpy as np
from core.logger import logger
from .base import STTModel, AudioBuffer

# Import faster_whisper lazily or handle import errors if not installed?
# Requirement said we installed it.
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

# Raised by huggingface_hub when a repo is not present in the local cache
try:
    from huggingface_hub.errors import LocalEntryNotFoundError
except ImportError:  # very old huggingface_hub
    LocalEntryNotFoundError = FileNotFoundError

# Try to import torch for cache clearing if available
try:
    import torch
except ImportError:
    torch = None


class FasterWhisperEngine(STTModel):
    def __init__(self, model_id=None, device="auto"):
        """`device`: 'auto' (CUDA-first, CPU fallback), 'cuda', or 'cpu'.

        CPU loads are int8-quantized and keep VRAM free at the cost of speed."""
        if not model_id:
            # Default to the turbo model requested
            self.model_id = "deepdml/faster-whisper-large-v3-turbo-ct2"
        else:
            self.model_id = model_id

        if device not in ("auto", "cuda", "cpu"):
            raise ValueError(f"Unknown device '{device}' (expected auto/cuda/cpu)")
        self.device = device
        # Actual device the model ended up on (set by load(); None until then —
        # differs from `self.device` when 'auto' falls back to CPU)
        self.loaded_device = None

        self.model = None
        self.status = "Unloaded"

        # Buffering for PTT logic (shared implementation with WhisperEngine)
        self.audio_buffer = AudioBuffer()

    def load(self):
        if WhisperModel is None:
            self.status = "Error: Missing Lib"
            logger.error("faster-whisper library not found.")
            raise ImportError("faster-whisper library not found.")

        try:
            self.status = "Loading..."
            self.loaded_device = None  # set by _create_model on success
            logger.info(f"Loading Faster Whisper Model: {self.model_id} (device={self.device})")

            if self.device == "cpu":
                # Explicit CPU request: int8 in system RAM, no GPU touch.
                logger.info("Loading on CPU...")
                self.model = self._create_model("cpu", "int8")
                logger.info("Faster Whisper loaded on CPU.")
            elif self.device == "cuda":
                logger.info("Attempting to load on CUDA...")
                self.model = self._create_model("cuda", "float16")
                logger.info("Faster Whisper loaded on CUDA.")
            else:  # auto: try CUDA first, fall back to CPU
                try:
                    logger.info("Attempting to load on CUDA...")
                    self.model = self._create_model("cuda", "float16")
                    logger.info("Faster Whisper loaded on CUDA.")
                except Exception as cuda_error:
                    # The failure type isn't always reliable, so always try CPU and,
                    # if that fails too, surface the original error as the cause.
                    logger.warning(f"Failed to load on CUDA ({cuda_error}). Falling back to CPU.")
                    try:
                        self.model = self._create_model("cpu", "int8")
                        logger.info("Faster Whisper loaded on CPU.")
                    except Exception as cpu_error:
                        logger.error(f"CPU load also failed: {cpu_error} (original CUDA error: {cuda_error})")
                        raise cpu_error from cuda_error

            self.status = "Ready"

        except Exception as e:
            self.status = "Error"
            logger.error(f"Failed to load Faster Whisper Engine: {e}")
            raise e

    def _create_model(self, device, compute_type):
        """Create the WhisperModel from the local cache first (no network).

        Only when the model is not cached yet (first run) does this fall back
        to a download from the Hugging Face Hub.
        """
        try:
            self.model = WhisperModel(
                self.model_id, device=device, compute_type=compute_type,
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            logger.info(f"Model '{self.model_id}' not in local cache - downloading from Hugging Face Hub...")
            self.model = WhisperModel(
                self.model_id, device=device, compute_type=compute_type,
            )
        self.loaded_device = device
        return self.model

    def unload(self):
        if self.model:
            del self.model
            self.model = None
        self.audio_buffer.clear()

        if torch and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error emptying CUDA cache: {e}")

        gc.collect()
        self.status = "Unloaded"
        logger.info("Faster Whisper Engine Unloaded.")

    def process_audio(self, audio_data):
        """
        Accumulates audio while PTT is held. Transcription runs on PTT release
        via finalize_transcription().
        """
        if self.model is None:
            return None

        # Accumulate (AudioBuffer handles the mono/float32 cap bookkeeping)
        self.audio_buffer.append(audio_data)

        return None

    def clear_buffer(self):
        self.audio_buffer.clear()
        logger.debug("FasterWhisper buffer cleared.")

    def finalize_transcription(self):
        """Called when PTT is released."""
        if not self.model:
            return None

        chunks = self.audio_buffer.finalize()
        if not chunks:
            return None

        logger.info("Finalizing Faster Whisper Transcription...")

        # Concatenate buffer (chunks were mono float32 at append time)
        full_audio = np.concatenate(chunks)

        # Ensure flat array (mono)
        if len(full_audio.shape) > 1:
            full_audio = full_audio.flatten()

        logger.info(f"Audio buffer size: {full_audio.shape}, dtype: {full_audio.dtype}")

        # Transcribe
        try:
            # Faster Whisper transcribe returns a generator of segments
            segments, info = self.model.transcribe(full_audio, beam_size=5, language="en")

            # Log info about the transcription (e.g., detected language probability)
            # info is a TranscriptionInfo object
            logger.info(f"Transcription Info: language={info.language}, prob={info.language_probability}")

            text_segments = []
            for segment in segments:
                text_segments.append(segment.text)

            full_text = " ".join(text_segments).strip()

            if full_text:
                logger.info(f"Faster Whisper Result: {full_text}")
                return full_text
            else:
                logger.info("Faster Whisper Result: (Empty)")
                return None

        except Exception as e:
            logger.error(f"Transcription Error: {e} (model type: {type(self.model).__name__})")
            return None

    def get_status(self):
        return self.status
