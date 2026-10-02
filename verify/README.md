# verify/ — functional test suite (no-launch safe)

Verification harness for Toki-STT. Everything here runs WITHOUT starting the app:
the GUI is constructed offscreen, the model loads from local cache on CPU, and no
audio is recorded. Safe to run at any time.

## 1. Functional suite (app code)

Run from the Toki-STT folder (the one containing `run_app.bat`), substituting
its path for `<Toki-STT>`:

```bat
cd /d <Toki-STT>
.venv\Scripts\python.exe verify\functional_verify.py
```

Exit code 0 + final `OVERALL: OK` = healthy. 20 checks covering:

- imports of every module (PySide6, faster-whisper, fermion, openvr, ...)
- ConfigManager: defaults, persistence round-trip, partial-file merge, corrupt-file recovery
- STT engine buffer contract (accumulate / dtype-coerce / clear / empty-finalize) on BOTH engines
- full PTT pipeline on a real model: load from cache → feed 3s of synthetic audio → transcribe → unload
- GUI offscreen: idle queue drain stays bounded at one chunk, volume meter live while idle,
  closeEvent unloads the model exactly once

Note: it patches `pynput.keyboard.Listener` out (headless sessions can't start global
keyboard hooks) and uses a fake QCloseEvent for the teardown check — everything else is real.

## 2. Launcher check (run_app.bat, no app launch)

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File <Toki-STT>\verify\nolaunch_verify.ps1
```

Runs a TEMPORARY copy of run_app.bat with the app-start line swapped for an echo —
the real `run_app.bat` is never modified. Expect `CMD_RC=0` and, in the log path it
prints: interpreter picked → environment OK / deps verified → `CUDA device online`
→ `[NO_LAUNCH_VARIANT]`.
