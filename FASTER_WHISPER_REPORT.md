# Faster Whisper Implementation & Verification Report

## Overview
This report documents the requirements, verification process, and implementation details for integrating `faster-whisper` (powered by CTranslate2) into the Toki-STT application. The goal is to provide a high-performance, streaming-capable speech-to-text option while avoiding common configuration errors related to NVIDIA libraries (cuBLAS, cuDNN).

## Requirements

The `faster-whisper` library relies on `ctranslate2` for inference. To enable GPU acceleration, the following environment requirements must be met:

1.  **NVIDIA Driver**: Capable of supporting CUDA 12.x.
2.  **CUDA Toolkit**: **Version 12.x** is required for the pre-built Python wheels of `ctranslate2`.
    *   *Note:* Older installations might be using CUDA 11, which causes compatibility issues with recent `ctranslate2` wheels.
3.  **cuDNN**: **Version 8 for CUDA 12.x**.
    *   *Crucial:* Speech recognition models (like Whisper) use convolutional layers, which **explicitly require cuDNN**. Failing to have cuDNN 8 installed and in the system PATH is the most common cause of "cublas" or "cudnn" errors.

### Documentation Links
*   [CTranslate2 Installation Guide](https://opennmt.net/CTranslate2/installation.html)
*   [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
*   [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

## Common Errors & Solutions

### Error: `Could not load library libcudnn_ops_infer.so.8` (or similar)
**Cause:** cuDNN 8 is missing or not in the library path.
**Solution:**
1.  Download cuDNN 8 for CUDA 12.x.
2.  Extract the files.
3.  Copy the libraries (`bin/*` on Windows, `lib/*` on Linux) to your CUDA Toolkit installation directory (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`) OR add the extracted folder to your system `PATH` / `LD_LIBRARY_PATH`.

### Error: `CUDA driver version is insufficient for CUDA runtime version`
**Cause:** Your NVIDIA driver is too old to support the installed CUDA Toolkit (likely CUDA 12).
**Solution:** Update your NVIDIA GPU drivers to the latest version.

### Error: `Library cudart is not found`
**Cause:** The CUDA Toolkit is not installed or not found.
**Solution:** Install CUDA Toolkit 12.x.

## Verification Process

The application performs this check at load time in `FasterWhisperEngine.load()` (`models/faster_whisper_engine.py`):

1.  **Attempt GPU Load:** The application attempts to load the model with `device="cuda"`.
2.  **Fallback:** If `ctranslate2` raises an error (indicating missing libraries), the application catches the exception, logs the specific error, and automatically falls back to `device="cpu"`. If the CPU attempt fails as well, both errors are logged so the root cause is visible.
3.  **User Notification:** The user is notified via the logs and the UI status if GPU acceleration failed.

## Implementation Details

A new engine class `FasterWhisperEngine` has been added in `models/faster_whisper_engine.py`.

*   **Model ID:** Defaults to `deepdml/faster-whisper-large-v3-turbo-ct2`.
*   **Compute Type:** Uses `float16` for CUDA and `int8` for CPU to optimize performance.
*   **Error Handling:** Wraps the model loading process in a try/except block to catch `ValueError` or `RuntimeError` related to backend initialization.

## Application Integration

The GUI (`main_window.py`) has been updated to include a "Faster Whisper" radio button. When selected:
1.  The application unloads the current model.
2.  It attempts to load `FasterWhisperEngine`.
3.  If GPU loading fails, it falls back to CPU and logs the original CUDA error.
