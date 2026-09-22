from abc import ABC, abstractmethod

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
