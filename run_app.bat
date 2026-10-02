@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo [INFO] Toki-STT launcher...

:: Ship the helper probes to a space-free location BEFORE any first use (the
:: fast-path device gate invokes them bare: "python stt_probe.py ..."). A failed
:: copy is harmless: every call site falls back gracefully to no GPU info.
copy /y "%~dp0stt_probe.py" "%TEMP%\stt_probe.py" >nul 2>&1
copy /y "%~dp0stt_pip.py" "%TEMP%\stt_pip.py" >nul 2>&1
copy /y "%~dp0phonon_check.py" "%TEMP%\phonon_check.py" >nul 2>&1

:: ===========================================================================
:: STEP 1 - Pick an interpreter (PYTHON_USED = how to invoke it: a "py -3.NN"
:: launcher form, bare "python", or a python.exe path). Preference order:
::   1. py launcher pinned per-minor: 3.12 first (reference cu124 pairing),
::      then 3.13 (cu126), then
::      3.9 / 3.10 / 3.11
::   2. bare "python" on PATH if its minor is supported
::   3. standard install dirs, newest name first: %LOCALAPPDATA%\Programs\Python,
::      Program Files, Program Files (x86), C:\PythonNN - same version filter
:: Supported = CPython 3.9-3.12 or 3.13 ONLY - those are the versions PyTorch
:: publishes Windows CUDA wheels for (verified on download.pytorch.org: nothing
:: at all for 3.8 and 3.14+). Unsupported interpreters are skipped on purpose;
:: a machine with none of them falls to auto-install, which pins a known-good
:: version. No guessing.
:: NOTE: values set inside "call"ed subroutines are only visible via !var! /
:: if defined - the script below uses those forms exclusively for PYTHON_USED.
:: ===========================================================================
set "PYTHON_USED="
call :find_python_any

if not defined PYTHON_USED goto :install_python
goto :have_python

:install_python
echo [WARN] No compatible Python (3.9-3.13) detected on this system. Attempting automatic install...
set "PY_PIN_VER=3.12.7"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "ARCH=arm64" else set "ARCH=amd64"
echo [INFO] Downloading official Python %PY_PIN_VER% installer (%ARCH%) from python.org...
set "INSTALLER=%TEMP%\python_installer.exe"

if exist "%INSTALLER%" del /f /q "%INSTALLER%" >nul 2>&1

where curl.exe >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Fetching with curl...
    curl -L --fail -o "%INSTALLER%" https://www.python.org/ftp/python/%PY_PIN_VER%/%ARCH%.exe
) else (
    echo [INFO] Fetching with PowerShell...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing 'https://www.python.org/ftp/python/%PY_PIN_VER%/%ARCH%.exe' -OutFile '%INSTALLER%'"
)

if not exist "%INSTALLER%" (
    echo [ERROR] Downloading the Python installer failed - check your internet connection, then re-run this script.
    pause
    exit /b 1
)

:: Pass 1: system-wide silent install via UAC ("Add to PATH", tests off).
echo [INFO] Starting silent install - a UAC prompt may appear...
start "pyinst" /wait "%INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

set "PYTHON_USED="
call :find_python_any
if defined PYTHON_USED goto :installed_ok

:: Pass 2: per-user silent install, no elevation.
echo [INFO] Trying a per-user install instead...
start "pyinst" /wait "%INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
del /f /q "%INSTALLER%" >nul 2>&1

:: Give the installer up to ~30 seconds, then re-check.
set /a "_W=0"
:install_wait
call :find_python_any
if defined PYTHON_USED goto :installed_ok
if %_W% GEQ 6 goto :install_failed
timeout /t 5 /nobreak >nul
set /a _W+=5
goto :install_wait

:installed_ok
del /f /q "%INSTALLER%" >nul 2>&1
goto :have_python

:: Normalize PYTHON_USED to a real absolute exe path (handles "py -3.12", bare
:: "python", and dir-scan hits alike) so venv markers compare stably across runs.
:: The probe is invoked with NO spaces ("py -3.12 stt_probe.py") and writes its
:: result to a file, sidestepping every cmd nested-quoting trap. If resolution
:: fails we keep the found form as-is: the worst case is a one-time venv rebuild.
:have_python
"!PYTHON_USED!" "%~dp0stt_probe.py" > %TEMP%\stt_py_real.txt 2>nul
set /p _REAL=<%TEMP%\stt_py_real.txt 2>nul
if defined _REAL set "PYTHON_USED=!_REAL!"
!PYTHON_USED! --version
echo [INFO] Using interpreter: !PYTHON_USED!

:: ===========================================================================
:: STEP 2 - Venv gate. .venv\.venv_marker records the REAL venv exe that built
:: it (canonical absolute path); a mismatch forces a clean rebuild so we never
:: pip-install across interpreters. An unsupported-venv (e.g. leftover Python
:: 3.14) also triggers a rebuild. A missing marker (pre-upgrade installs) is
:: tolerated: if that venv's interpreter is supported and all imports pass, it
:: runs - and the marker gets written on first success to normalize it.
:: ===========================================================================
if not exist ".venv\Scripts\python.exe" goto :create_venv

".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (sys.version_info.major,sys.version_info.minor) in ((3,9),(3,10),(3,11),(3,12),(3,13)) else 1)" >nul 2>&1
if not errorlevel 1 goto :venv_supported
echo [WARN] Existing .venv uses an unsupported Python (no CUDA wheels exist for it) - rebuilding with !PYTHON_USED! ...
rmdir /s /q ".venv" >nul 2>&1
goto :create_venv

:: Compare marker against the (already-normalized) interpreter directly; no
:: python.exe needed at read time, and a stored dead path simply fails over to
:: rebuild - safe behavior.
:venv_supported
set "VENV_MARKER="
if not exist ".venv\.venv_marker" goto :marker_missing_ok
set /p VENV_MARKER=<".venv\.venv_marker" 2>nul
if not defined VENV_MARKER goto :marker_missing_ok

if "!VENV_MARKER!" neq "!PYTHON_USED!" (
    echo [WARN] Existing .venv was built by a different Python - rebuilding...
    rmdir /s /q ".venv" >nul 2>&1
    goto :create_venv
)

:marker_missing_ok
if not exist ".venv\.venv_marker" echo [INFO] Pre-existing .venv without build marker - validating in place...

:: Import check (fast path). On success, make sure the marker exists so future
:: runs skip this validation entirely. (Marker = base interpreter PYTHON_USED,
:: NOT the venv python - comparing against the venv exe would mismatch and force
:: a rebuild on every launch.)
:: Device gate for pre-existing environments: a box that once ran an older wheel
:: set on a Blackwell card passes every import yet throws on its first CUDA op.
:: When nvidia-smi saw a GPU, require torch to actually execute one; failing it
:: routes into :deps_tail, which re-picks the index (cu128) and reinstalls torch
:: in place - self-healing without deleting .venv. Boxes with no NVIDIA driver
:: skip this gate (their setup may be CPU-only). Written flat on purpose: the
:: errorlevel test must sit directly after its command, and labels/comments do
:: not survive inside (...) blocks.
.venv\Scripts\python.exe -c "import PySide6, faster_whisper, fermion, pynput, sounddevice, pythonosc, numpy, torch" >nul 2>&1
if errorlevel 1 goto :venv_needs_repair

:: No nvidia-smi / no NVIDIA GPU: gpumajfile writes "0" (never fails silently),
:: and cudevfile ALWAYS exits 0 - both results arrive as TEXT tokens, so this
:: whole gate is plain comparisons with zero exit-code plumbing. The files are
:: rewritten every run; a stale one can never count. Repairs trigger ONLY when
:: nvidia-smi actually saw hardware: on a driverless box torch has no CUDA
:: context by definition and would otherwise loop into "repair" forever. On any
:: mismatch we repair in place via :deps_tail (re-picks the index, reinstalls).
".venv\Scripts\python.exe" stt_probe.py gpumajfile %TEMP%\stt_gpucc.txt >nul 2>nul
set /p GPU_MAJ=<"%TEMP%\stt_gpucc.txt" 2>nul

:: cudevfile via the venv python EXPLICITLY (never bare "python": PATH may hold
:: a stray interpreter that has no torch at all - this gate must test THE venv).
".venv\Scripts\python.exe" stt_probe.py cudevfile %TEMP%\stt_cudev.txt >nul 2>nul
set "CU_DEV="
if exist "%TEMP%\stt_cudev.txt" set /p CU_DEV=<"%TEMP%\stt_cudev.txt" 2>nul

:: GPU present, no CUDA context -> wheel build vs card/driver mismatch. Flat
:: single-line ifs: each is its own parse unit, so every variable read here is
:: the value written a line above (no block-level early expansion to fight).
if not defined GPU_MAJ goto :venv_ready
if !GPU_MAJ! LSS 1 goto :venv_ready
if "!CU_DEV!"=="ok" goto :venv_ready
goto :venv_needs_repair

:venv_ready

> ".venv\.venv_marker" echo !PYTHON_USED!
echo [INFO] Environment OK - skipping dependency installation...
goto :run

:venv_needs_repair
echo [WARN] Existing .venv is missing dependencies or its PyTorch cannot execute on this GPU (wheel build predates the card) - repairing in place...
goto :deps_tail

:create_venv
echo [INFO] Creating virtual environment with !PYTHON_USED! ...
!PYTHON_USED! -m venv .venv
if errorlevel 1 (
    echo [ERROR] Virtual environment creation failed.
    pause
    exit /b 1
)
:: Marker = the real base interpreter that built this venv (normalized above), so
:: it compares exactly against next run's PYTHON_USED - never the venv python,
:: which lives inside .venv and would mismatch on every subsequent launch.
> ".venv\.venv_marker" echo !PYTHON_USED!

:: ===========================================================================
:: STEP 3 - Dependencies (shared tail: full setup AND in-place repair)
:: The torch wheel index is chosen by BOTH axes, whichever is stricter:
::   a. Python minor of the VENV interpreter (wherever wheels land):
::      cu124 = cp39-cp315, cu126/cu128 = cp39+ - verified 2026-09 on
::      download.pytorch.org; nothing exists for 3.8, so the interpreter gate
::      above already excludes it.
::   b. The GPU itself (stt_probe.py gpucc -> nvidia-smi compute capability).
::      Blackwell cards (RTX 50 series = cc 12.x) have NO kernels in cu124 or
::      cu126 wheels - those builds predate sm_120, so torch loads but every
::      CUDA op throws. They need the cu128 index. This is exactly why a fresh
::      install on an RTX 50xx card picked up "working" wheels that could not
::      run the model; the launcher now detects it before installing.
:: The installed torch must then PASSIVE-verify (torch.version.cuda non-empty)
:: and ACTIVELY verify (a real CUDA tensor runs on device 0 - stt_probe.py
:: cudadevice), so a wheel/driver/arch mismatch is caught here, not mid-run.
:: ===========================================================================
:deps_tail
call .venv\Scripts\activate >nul 2>&1

set "_VENV_PY=%~dp0.venv\Scripts\python.exe"
    > ".venv\.venv_marker" echo !PYTHON_USED!

".venv\Scripts\python.exe" stt_probe.py minor > %TEMP%\stt_minor.txt 2>nul
set "PY_MINOR="
set /p PY_MINOR=<"%TEMP%\stt_minor.txt" 2>nul
set "TORCH_INDEX=cu124"
if "%PY_MINOR%"=="13" set "TORCH_INDEX=cu126"

:: GPU axis: nvidia-smi compute capability (highest across all cards). Same
:: token contract as the fast-path gate above: gpumajfile writes one integer
:: ("0" when no NVIDIA box) - plain numeric compare, no exit-code plumbing.
set "GPU_MAJ="
".venv\Scripts\python.exe" stt_probe.py gpumajfile %TEMP%\stt_gpucc.txt >nul 2>nul
set /p GPU_MAJ=<"%TEMP%\stt_gpucc.txt" 2>nul

if defined GPU_MAJ if !GPU_MAJ! GEQ 1 (echo [INFO] Detected NVIDIA compute capability major: !GPU_MAJ!) else (echo [WARN] nvidia-smi not found or no NVIDIA GPU - falling back to a CUDA build anyway; the device check below will verify.)

:: Major >= 12 = Blackwell (RTX 50 series): cu124/cu126 wheels have NO kernels
:: for sm_12x, so those builds load fine but every CUDA op throws. cu128 has
:: cp39-cp314 wheels - it covers every supported Python minor at once, and it is
:: the newest index we pin. (TORCH_INDEX below is read via %...% on its own
:: line = immediate expansion, which is safe; only (...) blocks expand early.)
if defined GPU_MAJ if !GPU_MAJ! GEQ 12 set "TORCH_INDEX=cu128"

echo [INFO] Python 3.!PY_MINOR! -^> PyTorch wheel index: %TORCH_INDEX%

:: Pick pip. Newer pip aborts with WinError 448 when PATH contains an entry on
:: an untrusted volume (here: cua-driver under AppData). stt_pip.py, copied to
:: %TEMP% by this script at startup (and again here - the deps tail can also be
:: reached after a .venv rebuild where %TEMP% was cleared), strips those entries
:: and the temp area for the child process only; plain pip is used if absent.
copy /y "%~dp0stt_pip.py" "%TEMP%\stt_pip.py" >nul 2>&1
set "PIPCMD=pip"
if exist "%TEMP%\stt_pip.py" set "PIPCMD=python %TEMP%\stt_pip.py"
:: Upgrade pip (plain: metadata-only ops don't touch the untrusted check)
echo [INFO] Checking dependencies...
python -m pip install --upgrade pip

echo [INFO] Installing/Verifying PyTorch %TORCH_INDEX% ...
!PIPCMD! install torch torchaudio --index-url https://download.pytorch.org/whl/%TORCH_INDEX%
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch from the %TORCH_INDEX% index.
    echo [INFO] This Python version has no CUDA wheels for this project - remove it and re-run, and the launcher will auto-install a known-good Python.
    pause
    exit /b 1
)

:: Other requirements (requirements.txt carries a cu124 extra-index; harmless on 3.13)
!PIPCMD! install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

set "TORCH_CU="
for /f "tokens=*" %%c in ('python -c "import torch;print(torch.version.cuda)"') do set "TORCH_CU=%%c"
echo [INFO] torch version.cuda = %TORCH_CU%
if not defined TORCH_CU (
    echo [ERROR] The installed PyTorch has no CUDA runtime. Re-run this script; if it persists, report the lines above.
    pause
    exit /b 1
)

:: Active check: a real CUDA context must come up on device 0. version.cuda is
:: only a build-time string - an old wheel build (e.g. cu124 on a Blackwell card)
:: carries one, yet every kernel launch throws. This gate catches it BEFORE the
:: app starts, not mid-run. cudevfile writes "ok" or "fail:<reason>" as a token;
:: %CU_DEV% below is read via delayed expansion (device names contain spaces).
set "CU_DEV="
python stt_probe.py cudevfile %TEMP%\stt_cudev.txt >nul 2>nul
if exist "%TEMP%\stt_cudev.txt" set /p CU_DEV=<"%TEMP%\stt_cudev.txt" 2>nul

if not "!CU_DEV!"=="ok" (
    echo [ERROR] PyTorch claims CUDA %TORCH_CU% but cannot execute on this GPU: !CU_DEV!
    echo [INFO] The wheel build predates this card's architecture, or the NVIDIA driver is out of date.
    if defined GPU_MAJ echo [INFO] Detected compute capability major !GPU_MAJ! - a matching torch wheel set is required - delete .venv and re-run, or update your NVIDIA driver.
    pause
    exit /b 1
)
echo [INFO] CUDA device online: !CU_DEV!

python -c "import PySide6, faster_whisper, fermion, pynput, sounddevice, pythonosc, numpy, torch" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Required Python dependencies are missing or broken.
    pause
    exit /b 1
)

:: ===========================================================================
:: STEP 4 - Phonon-2 (CPU engine) model setup.
:: The app drives a local `fermion serve` loopback server (spawned by the app
:: itself on first "Load Model"); this step only makes sure the one-time
:: ~164 MB model (FermionResearch/Phonon-2) is on the machine AND that a real
:: transcription works. phonon_check.py is the file-token helper copied to
:: %TEMP% at startup (same contract as stt_probe.py: ALWAYS exits 0, writes
:: one token). Missing model -> downloaded from huggingface.co now, so the
:: user's first PTT is not stuck on a 164 MB transfer mid-utterance.
:: Soft-fail: Faster Whisper keeps working even if this step fails; the error
:: is logged for the user, and the next launch retries.
:: ===========================================================================
set "PHONON_TOKEN="
python "%TEMP%\phonon_check.py" check %TEMP%\stt_phonon_model.txt >nul 2>nul
if exist "%TEMP%\stt_phonon_model.txt" set /p PHONON_TOKEN=<"%TEMP%\stt_phonon_model.txt"
if "%PHONON_TOKEN%"=="ok" goto :run

echo [INFO] Phonon-2 model not verified on this machine - prefetching (one-time ~164 MB download)...
python "%TEMP%\phonon_check.py" prefetch %TEMP%\stt_phonon_model.txt >nul 2>nul
set "PHONON_TOKEN="
if exist "%TEMP%\stt_phonon_model.txt" set /p PHONON_TOKEN=<"%TEMP%\stt_phonon_model.txt"
if not "%PHONON_TOKEN%"=="ok" (
    echo [WARN] Phonon-2 model setup did not complete: !PHONON_TOKEN!
    echo [WARN] The app will still start; pick Faster Whisper, or check your connection and re-run.
) else (
    echo [INFO] Phonon-2 model ready on this machine.
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

:: ===========================================================================
:: Subroutines - entered via "call :label", always ending in "goto :eof".
:: No setlocal inside any of them: a goto out while an endlocal'd scope is open
:: kills the script.
:: ===========================================================================

:: :find_python_any - sets PYTHON_USED to the first SUPPORTED interpreter found,
:: or leaves it undefined if none exists. Never falls through past its label.
:find_python_any
if defined PYTHON_USED goto :eof

set "_PYL=py -3.12"
call %_PYL% -c "import sys" >nul 2>&1 && for /f "delims=" %%v in ('call %_PYL% -c "import sys;print(sys.executable)"') do set "PYTHON_USED=%%~v"

if defined PYTHON_USED goto :eof
set "_PYL=py -3.13"
call %_PYL% -c "import sys" >nul 2>&1 && for /f "delims=" %%v in ('call %_PYL% -c "import sys;print(sys.executable)"') do set "PYTHON_USED=%%~v"

if defined PYTHON_USED goto :eof
set "_PYL=py -3.9"
call %_PYL% -c "import sys" >nul 2>&1 && for /f "delims=" %%v in ('call %_PYL% -c "import sys;print(sys.executable)"') do set "PYTHON_USED=%%~v"

if defined PYTHON_USED goto :eof
set "_PYL=py -3.10"
call %_PYL% -c "import sys" >nul 2>&1 && for /f "delims=" %%v in ('call %_PYL% -c "import sys;print(sys.executable)"') do set "PYTHON_USED=%%~v"

if defined PYTHON_USED goto :eof
set "_PYL=py -3.11"
call %_PYL% -c "import sys" >nul 2>&1 && for /f "delims=" %%v in ('call %_PYL% -c "import sys;print(sys.executable)"') do set "PYTHON_USED=%%~v"

if defined PYTHON_USED goto :eof
python -c "import sys; raise SystemExit(0 if (sys.version_info.major,sys.version_info.minor) in ((3,9),(3,10),(3,11),(3,12),(3,13)) else 1)" >nul 2>&1 && for /f "delims=" %%v in ('python -c "import sys;print(sys.executable)"') do set "PYTHON_USED=%%~v"

for /f "delims=" %%D in ('dir /b /ad /o-n "%LOCALAPPDATA%\Programs\Python\Python3*" 2^>nul') do call :check_py_dir "%LOCALAPPDATA%\Programs\Python\%%~D"
if defined PYTHON_USED goto :eof

for /f "delims=" %%D in ('dir /b /ad /o-n "%ProgramFiles%\Python3*" 2^>nul') do call :check_py_dir "%ProgramFiles%\%%~D"
if defined PYTHON_USED goto :eof

for /f "delims=" %%D in ('dir /b /ad /o-n "%ProgramFiles(x86)%\Python3*" 2^>nul') do call :check_py_dir "%ProgramFiles(x86)%\%%~D"
if defined PYTHON_USED goto :eof

for /f "delims=" %%D in ('dir /b /ad /o-n "C:\Python3*" 2^>nul') do call :check_py_dir "C:\%%~D"

goto :eof

:: :install_failed - guidance + hard stop (leaves PYTHON_USED undefined).
:install_failed
echo [ERROR] Python could not be installed automatically. Install it from https://www.python.org/downloads/ (tick 'Add to PATH') and re-run this script.
pause
exit /b 1

:: :check_py_dir <candidate dir> - if the dir holds a supported python.exe, set
:: PYTHON_USED to it; otherwise return with it untouched.
:check_py_dir
set "_PYDIR=%~1"
if not exist "%_PYDIR%\python.exe" goto :eof
"%_PYDIR%\python.exe" -c "import sys; raise SystemExit(0 if (sys.version_info.major,sys.version_info.minor) in ((3,9),(3,10),(3,11),(3,12),(3,13)) else 1)" >nul 2>&1
if errorlevel 1 goto :eof
set "PYTHON_USED=%_PYDIR%\python.exe"
goto :eof
