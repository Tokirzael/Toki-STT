# No-launch verification of run_app.bat. Makes a TEMPORARY sibling copy with the
# app-start line swapped for an echo, runs IT from the project root (so %~dp0 and
# all relative paths resolve exactly as in real use), then deletes the copy.
# The real run_app.bat is never touched - no restore step, no crash window.
$ErrorActionPreference = 'Stop'
$root  = Split-Path -Parent $PSScriptRoot        # repo root (script lives in verify\)
$tmp   = Join-Path $root '__nolaunch_verify.bat'
$logf  = Join-Path $env:TEMP 'nolaunch_result.log'

$c  = [IO.File]::ReadAllText((Join-Path $root 'run_app.bat'))
$n  = ".venv\Scripts\python.exe main.py`r`n"
if (-not $c.Contains($n)) { Write-Output "NEEDLE_NOT_FOUND"; exit 1 }
$c2 = $c.Replace($n, "echo [NO_LAUNCH_VARIANT] app-start line replaced (verification run)`r`n")
try {
    [IO.File]::WriteAllText($tmp, $c2)
    Push-Location $root
    cmd /c "__nolaunch_verify.bat > `"$logf`" 2>&1 < NUL" | Out-Null
    $rc = $LASTEXITCODE
    Pop-Location
} finally {
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
}

Write-Output "CMD_RC=$rc"
Write-Output "LOG=$logf"
if ($rc -eq 0) { Write-Output 'NO_LAUNCH_OK' } else { Write-Output 'NO_LAUNCH_FAILED' }
