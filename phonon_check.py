"""Phonon-2 model check + prefetch helper for run_app.bat.

Usage (from the launcher, via the venv python):
    python phonon_check.py check <out_file>
        Writes one token to <out_file>: "ok" (model cached) or "missing".
    python phonon_check.py prefetch <out_file>
        Ensures the FermionResearch/Phonon-2 speech model is on this machine
        (one-time ~164 MB download from huggingface.co, cached under
        %USERPROFILE%\\.cache\\fermion). Then runs a throwaway transcription of
        0.1 s of silence through `python -m fermion.cli transcribe` to confirm
        the whole chain works. Writes "ok" or "fail:<reason>".

Exit code is always 0 in token mode - the launcher reads the file token and
does plain string comparisons (same contract as stt_probe.py).
"""
import json
import os
import subprocess
import sys
import tempfile
import wave


def _models_json():
    r = subprocess.run(
        [sys.executable, "-m", "fermion.cli", "models", "--json"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        return None, f"models rc={r.returncode}: {r.stderr.strip()[:120]}"
    try:
        return json.loads(r.stdout), None
    except Exception as e:
        return None, f"bad json: {e}"


def _phonon2_entry(models):
    for m in models.get("speech", []):
        if m.get("model", "").lower().endswith("phonon-2") or \
           "phonon-2" in (m.get("aliases") or []):
            return m
    return None


def write_token(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    out_file = sys.argv[2] if len(sys.argv) > 2 else None
    if not out_file:
        print(__doc__.strip().splitlines()[1])
        sys.exit(2)

    try:
        if mode == "check":
            models, err = _models_json()
            if err:
                write_token(out_file, f"fail:{err}")
                return
            m = _phonon2_entry(models)
            if m is None:
                write_token(out_file, "fail:phonon-2 not in model list")
                return
            write_token(out_file, "ok" if m.get("installed") else "missing")

        elif mode == "prefetch":
            models, err = _models_json()
            if err:
                write_token(out_file, f"fail:{err}")
                return
            m = _phonon2_entry(models)
            if m is None:
                write_token(out_file, "fail:phonon-2 not in model list")
                return
            if not m.get("installed"):
                # 0.1 s of silence: triggers the one-time download, then a
                # full decode of real (empty) audio end-to-end.
                wav = os.path.join(tempfile.gettempdir(), "stt_phonon_silence.wav")
                with wave.open(wav, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    w.writeframes(b"\x00\x00" * 1600)
                r = subprocess.run(
                    [sys.executable, "-m", "fermion.cli", "transcribe",
                     "phonon-2", wav],
                    capture_output=True, text=True, timeout=1800,
                )
                if r.returncode != 0:
                    write_token(out_file,
                                f"fail:download/transcribe rc={r.returncode}: {r.stderr.strip()[:160]}")
                    return
            write_token(out_file, "ok")
        else:
            write_token(out_file, f"fail:unknown mode {mode}")
    except Exception as e:
        write_token(out_file, f"fail:{type(e).__name__}: {str(e)[:120]}")
    finally:
        sys.exit(0)  # token mode: batch never chains errorlevels


if __name__ == "__main__":
    main()
