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
:: Locate a python.exe. Sets PYTHON to the executable used for everything
:: below: py launcher, PATH, then standard install directories (so a fresh
:: install whose PATH hasn't reached this window is still usable).
:: ---------------------------------------------------------------------------
set "PYTHON="
call :find_python
if defined PYTHON goto :have_python

echo [WARN] Python was not detected on this system. Attempting automatic install...
call :install_python
if defined PYTHON goto :have_python
:: Subroutine left PYTHON undefined: the failure path did not exit cleanly, so
echo [ERROR] Automatic Python install failed - see messages above for details.
pause
exit /b 1

:: ---------------------------------------------------------------------------
:: Auto-install. Download the official python.org installer once (pinned to
:: 3.13.5 - what this project is tested against; arch matched to the machine).
:: Pass 1: system-wide silent install via UAC ("Add to PATH" enabled, tests off).
:: If that doesn't land a working copy, pass 2: per-user silent install (no
:: elevation) into %LOCALAPPDATA%\Programs\Python. Detection scans the standard
:: directories, so this run keeps going without opening a new window; the PATH
:: update takes effect in future sessions.
:: ---------------------------------------------------------------------------
:install_python
:: Pick the installer for this machine's architecture (x86_64 default; arm64
:: fallback covers Snapdragon/ARM laptops).
set "ARCH=amd64"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "ARCH=arm64"
echo [INFO] Downloading official Python installer (%ARCH%) from python.org...
set "INSTALLER=%TEMP%\python_installer.exe"

if exist "%INSTALLER%" del /f /q "%INSTALLER%" >nul 2>&1

where curl.exe >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Fetching with curl...
    curl -L --fail -o "%INSTALLER%" https://www.python.org/ftp/python/3.13.5/%ARCH%.exe
) else (
    echo [INFO] Fetching with PowerShell...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing 'https://www.python.org/ftp/python/3.13.5/%ARCH%.exe' -OutFile '%INSTALLER%'"
)

if not exist "%INSTALLER%" (
    echo [ERROR] Downloading the Python installer failed - check your internet connection, then re-run this script.
    goto :install_failed
)

:: Pass 1: system-wide install with elevation
echo [INFO] Starting silent install - a UAC prompt may appear...
start "pyinst" /wait "%INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

call :find_python_all
if defined PYTHON ( del /f /q "%INSTALLER%" >nul 2>&1 & goto :eof )

:: Pass 2: per-user install, no elevation
echo [INFO] Trying a per-user install instead...
start "pyinst" /wait "%INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
del /f /q "%INSTALLER%" >nul 2>&1

:: Give the installer up to ~30 seconds, then re-check.
set /a "_W=0"
:install_wait
call :find_python_all
if defined PYTHON goto :eof
if %_W% GEQ 6 goto :install_failed
timeout /t 5 /nobreak >nul
set /a _W+=5
goto :install_wait

:install_failed
echo [ERROR] Python could not be installed automatically. Install it from https://www.python.org/downloads/ (tick 'Add to PATH') and re-run this script.
pause
exit /b 1

:: ---------------------------------------------------------------------------
:: Subroutine :find_python_all - same scan as :find_python, but starts from an
:: undefined PYTHON so a just-finished install is always picked up fresh.
:: ---------------------------------------------------------------------------
:find_python_all
set "PYTHON="
call :find_python
goto :eof

:: ---------------------------------------------------------------------------
:: Full setup
:: ---------------------------------------------------------------------------
:have_python
%PYTHON% --version
echo [INFO] Creating/updating virtual environment...
if not exist ".venv" %PYTHON% -m venv .venv

call .venv\Scripts\activate >nul 2>&1

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

:: ---------------------------------------------------------------------------
:: Subroutine :find_python - sets PYTHON to the first python.exe found in:
::   1. py launcher           2. PATH
::   3. %LOCALAPPDATA%\Programs\Python  (per-user installs, newest dir first)
::   4. Program Files         5. Program Files (x86)
:: Leaves PYTHON undefined if nothing is found. No setlocal inside this
:: routine: a goto out of it while an endlocal'd scope is open kills the script.
:: ---------------------------------------------------------------------------
:find_python
if defined PYTHON goto :eof

:: C:\Windows\System32\py.exe is a shim on every Win10/11 box even with no
:: Python installed - so "it exists" is not enough; the probe must actually
:: launch an interpreter (exit 0 from executing code).
py -c "import sys" >nul 2>&1 && set "PYTHON=py" && goto :eof

python -c "import sys" >nul 2>&1 && set "PYTHON=python" && goto :eof

:: dir /b /ad /o-n lists matching dirs newest-name-first (newest version wins)
for /f "delims=" %%D in ('dir /b /ad /o-n "%LOCALAPPDATA%\Programs\Python\Python3*" 2^>nul') do if exist "%LOCALAPPDATA%\Programs\Python\%%~D\python.exe" ( set "PYTHON=%LOCALAPPDATA%\Programs\Python\%%~D\python.exe" & goto :eof )

for /f "delims=" %%D in ('dir /b /ad /o-n "%ProgramFiles%\Python3*" 2^>nul') do if exist "%ProgramFiles%\%%~D\python.exe" ( set "PYTHON=%ProgramFiles%\%%~D\python.exe" & goto :eof )

for /f "delims=" %%D in ('dir /b /ad /o-n "%ProgramFiles(x86)%\Python3*" 2^>nul') do if exist "%ProgramFiles(x86)%\%%~D\python.exe" ( set "PYTHON=%ProgramFiles(x86)%\%%~D\python.exe" & goto :eof )

goto :eof
