# Self-cleaning launcher with a 60s re-entry guard (prevents double-launch races)
$lock = "C:\STT\.launch_lock"
if (Test-Path $lock) {
  $age = ((Get-Date) - (Get-Item $lock).LastWriteTime).TotalSeconds
  if ($age -lt 60) { Write-Output ("LOCKED, skipping (last launch {0}s ago)" -f [int]$age); exit 0 }
}
Set-Content -Path $lock -Value ((Get-Date).ToString("s"))
$stt = Get-CimInstance Win32_Process -Filter "name like 'python%'" | Where-Object { $_.CommandLine -match '\.venv.*main\.py' }
foreach ($p in $stt) { Stop-Process -Id $p.ProcessId -Force; Write-Output ("KILLED {0}" -f $p.ProcessId) }
Start-Sleep -Seconds 2
$p = Start-Process -FilePath "C:\STT\.venv\Scripts\pythonw.exe" -ArgumentList "main.py" -WorkingDirectory "C:\STT" -PassThru
Write-Output ("STARTED {0}" -f $p.Id)
