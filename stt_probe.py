r"""Probe helper for the Toki-STT launcher (stdlib only).

Shipped next to run_app.bat. Prints
single-line facts about interpreters / the GPU so run_app.bat can read them
WITHOUT any nested quoting in batch:

    stt_probe.py            -> prints sys.executable of THIS python
                               (= the venv python when run from .venv\Scripts)
    stt_probe.py <path>     -> prints os.path.realpath(<path>) (space-safe)
    stt_probe.py minor      -> prints "3.10" / "3.12" / "3.13" ... of THIS python
    stt_probe.py torchcuda  -> prints torch.version.cuda of THIS python
                               (empty line + exit 1 if torch is missing/CPU)
    stt_probe.py gpucc      -> highest NVIDIA compute capability on the box via
                               nvidia-smi, e.g. "8.9" / "12.0" (empty line +
                               exit 1 when no NVIDIA driver is present). Drives
                               the torch wheel-index choice: Blackwell cards
                               (compute cap >= 12.x, RTX 50 series) ship NO
                               kernels in cu124/cu126 wheels and need cu128+.
    stt_probe.py gpumajfile <path> -> writes ONLY the highest MAJOR compute
                               capability as integer text ("8", "9", "12") to
                               <path>, or the single character "0" when no NVIDIA
                               driver/GPU is present - so the launcher needs NO
                               exit-code plumbing: it reads one token and compares.
                               Prints nothing; the file is rewritten every run so
                               stale content from a dead driver never lingers.
    stt_probe.py cudadevice -> prints the online CUDA device name of THIS python
                               (empty line + exit 1 when torch can't use a GPU);
                               proves the installed wheel build actually runs on
                               this card, not just that it claims a CUDA runtime.
    stt_probe.py cudevfile <path> -> file twin of cudadevice: writes "ok" on a
                               successful device query, "fail:<reason>" otherwise;
                               ALWAYS exits 0 and prints nothing, so batch logic
                               reads exactly one text token instead of chaining
                               exit codes through commands that clobber them.
                               File modes without <path> print usage and exit 2 -
                               they must never fall through to the realpath mode,
                               which would leave a stale file readable as valid.

The launcher invokes it by RELATIVE path, e.g.:
    .venv\Scripts\python.exe stt_probe.py minor
so the command line never contains spaces and no quoting is needed.
"""
import os
import shutil
import subprocess
import sys


def _nvidia_smi_exe():
    exe = shutil.which("nvidia-smi")
    if not exe:
        # Driver installs to System32 by default; check explicitly in case it is
        # missing from this process's PATH.
        sysroot = os.environ.get("SystemRoot", r"C:\Windows")
        cand = os.path.join(sysroot, "System32", "nvidia-smi.exe")
        if os.path.isfile(cand):
            exe = cand
    return exe


def _gpu_caps():
    """(stdout_text, rc) from nvidia-smi compute-cap query; rc != 0 when no
    NVIDIA driver/GPU is present or the call fails."""
    exe = _nvidia_smi_exe()
    if not exe:
        return "", 1
    try:
        out = subprocess.run(
            [exe, "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "", 1
    return (out.stdout or ""), out.returncode


def _best_cc(caps_text):
    """Highest "major*100+minor" across all GPUs, or None when unparseable."""
    caps = []
    for line in caps_text.splitlines():
        t = line.strip()
        if "." not in t:
            continue
        try:
            major, minor = (int(x) for x in t.split(".")[:2])
        except ValueError:
            continue
        caps.append(major * 100 + minor)
    return max(caps) if caps else None


def _cuda_device_query():
    """(ok, detail) for THIS python's torch: ok=True + device name when a CUDA
    context comes up on device 0; ok=False otherwise (detail carries the reason
    for diagnostics; empty string is fine)."""
    try:
        import torch
    except Exception:
        return False, "torch missing"
    try:
        if not torch.cuda.is_available():
            # Surface the real reason once (driver / wheel-arch mismatch).
            torch.zeros(1, device="cuda")
        name = torch.cuda.get_device_name(0)
    except Exception as exc:
        return False, type(exc).__name__ + ": " + str(exc)[:200]
    if not name:
        return False, "empty device name"
    return True, name


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    # File-token modes require <path>; refuse to fall through to the realpath
    # handler (which would treat the mode name as a path and, worse, leave any
    # stale output file readable by batch logic as a valid answer).
    if arg in ("gpumajfile", "cudevfile") and len(sys.argv) < 3:
        print("usage: stt_probe.py %s <output-path>" % arg)
        return 2

    if arg == "minor":
        print(sys.version_info[1])
        return 0

    if arg == "gpucc":
        caps_text, rc = _gpu_caps()
        best = _best_cc(caps_text)
        if rc != 0 or best is None:
            print("")
            return 1
        # "M.m" with a single minor digit - matches nvidia-smi's own format.
        print("%d.%d" % (best // 100, best % 100))
        return 0

    if arg == "gpumajfile":
        caps_text, rc = _gpu_caps()
        best = _best_cc(caps_text)
        # Text token the launcher reads with set /p. "0" when no NVIDIA box -
        # one token covers both detection AND status, so batch does a plain
        # numeric compare (GEQ 12) with zero exit-code plumbing. Always rc 0:
        # writing the file IS the result.
        major = str(best // 100) if best is not None and rc == 0 else "0"
        try:
            with open(sys.argv[2], "w", encoding="ascii") as f:
                f.write(major)
        except OSError:
            return 1
        return 0

    if arg in ("cudadevice", "cudevfile"):
        result = _cuda_device_query()
        ok, detail = result
        if arg == "cudevfile":
            try:
                with open(sys.argv[2], "w", encoding="ascii") as f:
                    f.write("ok" if ok else ("fail:" + detail))
            except OSError:
                return 1
            # Always 0 on purpose: the TEXT is the status; batch must not have
            # to chain exit codes through commands that overwrite them.
            return 0
        print(detail)
        return 0 if ok else 1

    if arg == "torchcuda":
        try:
            import torch
        except Exception:
            print("")
            return 1
        cu = torch.version.cuda or ""
        print(cu)
        return 0 if cu else 1

    if arg is None:
        print(sys.executable)
        return 0

    # positional path argument -> realpath (single token, no spaces in output)
    print(os.path.realpath(arg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
