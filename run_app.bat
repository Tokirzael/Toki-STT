@echo off
setlocal
cd /d "%~dp0"
echo [INFO] Toki-STT launcher...

:: ---------------------------------------------------------------------------
:: Fast path: if the venv exists and all required dependencies import cleanly,
:: skip the pip steps entirely (fast + works offline). Otherwise fall through
:: to the full setup below (one-shot installer behavior is preserved).
:: ---------------------------------------------------------------------------
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -c "import PySide6, whisper, faster_whisper, pynput, sounddevice, pythonosc, numpy, torch" >nul 2>&1
    if not errorlevel 1 (
        echo [INFO] Environment OK - skipping dependency installation...
        goto :run
    )
    echo [WARN] Existing .venv is missing dependencies - repairing...
) else (
    echo [INFO] No .venv found - running full setup...
)

:: ---------------------------------------------------------------------------
:: Full setup
:: ---------------------------------------------------------------------------

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.10+.
    pause
    exit /b 1
)

:: Create .venv if it doesn't exist
if not exist ".venv" (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
)

:: Activate venv
call .venv\Scripts\activate

:: Upgrade pip
echo [INFO] Checking dependencies...
python -m pip install --upgrade pip

:: Explicitly install PyTorch with CUDA 12.4
echo [INFO] Installing/Verifying PyTorch CUDA 12.4 ...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [WARNING] CUDA 12.4 install failed. Trying standard PyPI CPU/Default ...
    pip install torch torchaudio
    if errorlevel 1 (
        echo [ERROR] Failed to install PyTorch.
        pause
        exit /b 1
    )
)

:: Install other requirements
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: Final sanity check (openvr is optional: VR input is disabled without it)
python -c "import PySide6, whisper, faster_whisper, pynput, sounddevice, pythonosc, numpy, torch" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Required Python dependencies are missing or broken.
    pause
    exit /b 1
)

:run
echo [INFO] Starting Application...
.venv\Scripts\python.exe main.py
if errorlevel 1 (
    echo [ERROR] Application crashed.
    pause
)

endlocal
pause
