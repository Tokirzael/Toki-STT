# Toki-STT — Push-To-Talk Whisper Transcription for VRChat

Desktop/VR speech-to-text utility: hold a PTT key (keyboard or SteamVR controller), speak, release — the utterance is transcribed locally with Whisper and posted to your VRChat chatbox over OSC. No cloud audio, no API keys.

## Quick Start

```bat
run_app.bat
```

That's it. First run installs Python 3.10+, a local `.venv`, CUDA 12.4 PyTorch (falls back to CPU build automatically), and everything in `requirements.txt`. Subsequent runs detect the healthy environment and skip straight to launch — works offline once set up.

In the window: pick an engine, click **Load Model**, press PTT.

## STT Engines

| Radio button | Engine | VRAM (loaded) | Notes |
|---|---|---|---|
| Whisper Turbo (PTT) | `openai-whisper` large-v3-turbo, fp16 on CUDA | ~4 GB | Falls back to CPU automatically if no CUDA |
| Faster Whisper (PTT) | faster-whisper CT2 int8 turbo (`deepdml/faster-whisper-large-v3-turbo-ct2`) | ~2 GB | Local-cache-first; downloads from HF only on first run |

Both transcribe the **accumulated PTT buffer** on key release. Check **"Use System RAM/CPU"** to keep all VRAM free at the cost of speed (Whisper runs on CPU, Faster Whisper loads int8 in system RAM). Models load/unload manually — nothing is loaded until you ask.

## Controls

- **Keyboard PTT**: default `Ctrl+R`. Rebind with **Record PTT** (Esc cancels; some keys are blocked so you can't lock yourself out of the UI).
- **VR PTT**: bind any SteamVR controller button with **Record VR PTT**. Requires SteamVR running at launch or re-pressing Record. Optional per-controller filtering is applied automatically from the recorded device.
- Both inputs share one logical PTT: pressing keyboard + VR simultaneously is debounced to a single utterance, and overlapping transcriptions queue in order instead of dropping text.

## VRChat OSC Output

- Default endpoint `127.0.0.1:9000` (standard VRChat). Change IP/port with **Apply OSC**.
- Long messages are split into 144-character chunks at 5-second intervals so VRChat doesn't truncate them; chatbox input is rate-limited to one message per 1.5 s.
- The typing indicator (`/chatbox/typing`) goes out on PTT press/release. After ~10 s of silence the chatbox contents are cleared (same behavior as a real user finishing their message).

## Configuration

Stored in `settings.json` at the project root (created with defaults on first run; mic selection, engine choice, PTT bindings, and OSC endpoint persist here). If the file is ever corrupt it is backed up to `settings.json.corrupt` and healthy defaults are rewritten automatically.

Audio capture runs continuously from app start (the input-level meter needs it) at 16 kHz; your microphone's native channel count is downmixed to mono in the engine, not the OS. Buffered PTT audio is capped at 2 minutes so a stuck key can't grow memory without bound.

## Launching Headless / From Task Scheduler

`stt_launch_task.bat` (→ `stt_launch_task.ps1`) kills any previous instance and starts the app with `pythonw.exe` under a 60-second re-entry lock — for startup tasks or remote restarts. The GUI enforces its own single-instance guard via a temp-dir `QLockFile`, so double-launches from anywhere are safe (the second instance exits cleanly).

## Logging

Console + rotating file (`debug.log`, 5 × 5 MB) at the project root, full DEBUG detail in the file for post-mortems.

## Project Layout

```
main.py                     Entry: QApplication + single-instance guard
core/                       audio capture (sounddevice), input (pynput/OpenVR),
                            OSC client, config, logging
gui/main_window.py          The whole UI (PySide6)
models/base.py              STTModel interface
models/whisper_engine.py    openai-whisper engine
models/faster_whisper_engine.py  faster-whisper CT2 engine
scripts/                    Legacy exploration scripts for the old Kroko engine —
                            not part of the app, safe to delete
run_app.bat                 Self-healing installer + launcher (the only entry point you need)
stt_launch_task.{bat,ps1}   Headless relaunch helper
```

## Requirements

- Windows with a microphone; NVIDIA GPU optional but strongly recommended
- SteamVR only needed for VR PTT — the app runs fine without it
- Python 3.10+ on PATH (installed by `run_app.bat`'s full-setup path if missing)
