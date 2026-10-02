r"""pip runner for the Toki-STT launcher (stdlib only).

Two faces of a Windows quirk make plain `pip install` fail on some machines:
when a volume is flagged "untrusted" in MountPointTracker, newer pip aborts
any install with WinError 448 - triggered by PATH entries on that tree AND by
temp dirs that land under it. Both are fixed HERE so no other file changes:

1. The child gets a PATH with exactly the flagged entries removed.
2. The child gets fresh temp dirs + pip cache under %USERPROFILE%\.stt-tmp,
   which is always on a trusted local volume.

The launcher ALWAYS invokes this as:
    <venv-python> "%TEMP%\stt_pip.py" <pip args...>
and we spawn our OWN `sys.executable -m pip`, so the venv interpreter runs pip
no matter what "python" resolves to on PATH (some machines carry extra
interpreters that must never see these installs). It is a no-op shim where
nothing is flagged, safe to ship in every copy of the project.
"""
import os
import subprocess
import sys


def _flagged(entry):
    if "Cua" not in entry:
        return False
    low = entry.lower()
    return (os.sep + "cua-driver") in low or ("/cua-driver/") in low


def main():
    args = sys.argv[1:]

    env = dict(os.environ)
    for var in ("Path", "PATH"):
        cur = env.get(var)
        if not cur:
            continue
        kept, dropped = [], []
        for e in cur.split(os.pathsep):
            (dropped if _flagged(e) else kept).append(e)
        if dropped:
            env[var] = os.pathsep.join(kept)
            print("[stt-pip] PATH entry removed for pip:", dropped, flush=True)

    # Trusted temp area under the user profile - never under AppData.
    tmpdir = os.path.join(os.environ.get("USERPROFILE", "."), ".stt-tmp")
    try:
        os.makedirs(tmpdir, exist_ok=True)
        env["TMP"] = tmpdir
        env["TEMP"] = tmpdir
        env["PIP_CACHE_DIR"] = os.path.join(tmpdir, "pip-cache")
    except OSError:
        pass

    if not args:
        return 0

    # spawnve (NOT spawnv): the sanitized environment must be passed explicitly.
    # sys.executable == the venv python that ran us - immune to PATH drift.
    #
    # NOTE: os.spawnve was the original implementation but it is SPACE-HOSTILE:
    # on Windows it re-parses a space-joined command line and does not quote
    # the executable, so a venv python at a space-containing path (e.g.
    # "...\Toki-STT-main (2)\.venv\Scripts\python.exe") is mangled into
    # "C:\\...\\pf: can't open file 'C:\\...\\sim\\python.exe'". subprocess.call
    # builds the command line correctly (list2cmdline quotes each arg) and
    # still passes the sanitized env explicitly. Proven end-to-end on a
    # space-path python.
    return subprocess.call([sys.executable, "-m", "pip"] + args, env=env)


if __name__ == "__main__":
    sys.exit(main())
