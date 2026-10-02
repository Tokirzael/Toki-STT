# Toki-STT — Push-To-Talk Transcription for VRChat

Desktop/VR speech-to-text utility: hold a PTT key (keyboard or SteamVR controller), speak, release — the utterance is transcribed locally with Whisper and posted to your VRChat chatbox over OSC. No cloud audio, no API keys.

## Quick Start

```bat
run_app.bat
```

That's it. First run finds a suitable Python (3.9–3.13, auto-installing a pinned 3.12 if none exists), creates a local `.venv`, installs PyTorch from the CUDA wheel index that matches BOTH your Python minor and GPU architecture (cu124/cu126 normally; cu128 for RTX 50-series Blackwell cards, which have no kernels in older builds — detected via `nvidia-smi` before anything is installed), then everything in `requirements.txt`. On a box without an NVIDIA GPU it installs the same CUDA builds and verifies with a device probe at startup. Subsequent runs detect the healthy environment and skip straight to launch — works offline once set up (a wheel/driver mismatch on an existing `.venv` self-repairs in place).

In the window: pick an engine, click **Load Model**, press PTT.

## STT Engines

| Radio button | Engine | Footprint | Notes |
|---|---|---|---|
| Faster Whisper (PTT) | faster-whisper CT2 float16 turbo (`deepdml/faster-whisper-large-v3-turbo-ct2`) on CUDA | ~2 GB VRAM | GPU only. Local-cache-first; downloads from HF only on first run |
| Phonon-2 (PTT) | FermionResearch/Phonon-2 (~164 MB) via a local `fermion serve` loopback server | CPU / RAM, English only | No GPU needed. The first "Load Model" starts a detached local server if one isn't running; `run_app.bat` prefetches the model on first setup |

Both transcribe the **accumulated PTT buffer** on key release. Models load/unload manually — nothing is loaded until you ask. Unloading Phonon-2 keeps the local server running (it's fast to reuse and outlives the app on purpose).

## Controls

- **Keyboard PTT**: default `V` (yours is whatever you last recorded — it persists in `settings.json`). Rebind with **Record PTT** (Esc cancels; some keys are blocked so you can't lock yourself out of the UI).
- **VR PTT**: bind any SteamVR controller button with **Record VR PTT**. Requires SteamVR running at launch or re-pressing Record. Optional per-controller filtering is applied automatically from the recorded device.
- Both inputs share one logical PTT: pressing keyboard + VR simultaneously is debounced to a single utterance, and overlapping transcriptions queue in order instead of dropping text.

## VRChat OSC Output

- Default endpoint `127.0.0.1:9000` (standard VRChat). Change IP/port with **Apply OSC**.
- Long messages are split into 144-character chunks at 5-second intervals so VRChat doesn't truncate them; chatbox input is rate-limited to one message per 1.5 s.
- The typing indicator (`/chatbox/typing`) goes out on PTT press/release. After ~10 s of silence the chatbox contents are cleared (same behavior as a real user finishing their message).

## Configuration

Stored in `settings.json` at the project root (created with defaults on first run; mic selection, engine choice, PTT bindings, and OSC endpoint persist here). If the file is ever corrupt it is backed up to `settings.json.corrupt` and healthy defaults are rewritten automatically.

Audio capture runs continuously from app start (the input-level meter needs it) at 16 kHz; your microphone's native channel count is downmixed to mono in the engine, not the OS. Buffered PTT audio is capped at 2 minutes so a stuck key can't grow memory without bound.

## Logging

Console + rotating file (`debug.log`, 5 × 5 MB) at the project root, full DEBUG detail in the file for post-mortems.

## Project Layout

```
main.py                     Entry: QApplication + single-instance guard
core/                       audio capture (sounddevice), input (pynput/OpenVR),
                            OSC client, config, logging
gui/main_window.py          The whole UI (PySide6)
models/base.py              STTModel interface + shared AudioBuffer (PTT accumulation)
models/faster_whisper_engine.py  faster-whisper CT2 engine (GPU only)
models/phonon_engine.py     Phonon-2 engine: drives a local `fermion serve` loopback server
run_app.bat                 Self-healing installer + launcher (the only entry point you need)
stt_probe.py / stt_pip.py   Launcher helpers (interpreter/GPU probes; pip wrapper that
                            works around untrusted-mount PATH issues on this box)
phonon_check.py             Launcher helper: verifies/prefetches the Phonon-2 model (file token)
stt_launch_task.{bat,ps1}   Headless relaunch helper
```

## Requirements

- Windows with a microphone; NVIDIA GPU optional but strongly recommended
- SteamVR only needed for VR PTT — the app runs fine without it

## License

Copyright (c) 2026 Toki Ir'zael. Free software, offered as a gift.

You are free to:

- use it, in whole or in part, for any purpose that is not for profit;
- study, modify, and improve it;
- share your copies with anyone, free of charge, provided this license
  travels with the code.

You may not:

- sell it, or bundle it with a paid product or service;
- use it commercially (in exchange for money or other profit) without
  the author's express written permission.

Provided "as is", without warranty of any kind; the author is not liable
for any damages arising from its use.
