# Chạy demo website. Dùng:  .\web\run_demo.ps1
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
Start-Job { Start-Sleep 4; Start-Process "http://127.0.0.1:8000" } | Out-Null
& "$root\.venv\Scripts\python.exe" "$root\web\server.py"
