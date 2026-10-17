# Run the service and expose it through an ngrok static domain; restarts either process if it stops.
# Usage: .\serve_public.ps1 -Domain your-domain.ngrok-free.dev
param(
    [Parameter(Mandatory = $true)][string]$Domain,
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force runs | Out-Null

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$ngrok = (Get-Command ngrok -ErrorAction SilentlyContinue).Source
if (-not $ngrok) {
    $ngrok = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter ngrok.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $ngrok) { throw "ngrok not found: winget install Ngrok.Ngrok" }

function Start-Server {
    Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", $Port, "--workers", "1" `
        -RedirectStandardOutput "runs\server.out.log" -RedirectStandardError "runs\server.err.log" -WindowStyle Hidden -PassThru
}
function Start-Tunnel {
    Start-Process -FilePath $ngrok -ArgumentList "http", "--url=https://$Domain", "$Port", "--log=stdout" `
        -RedirectStandardOutput "runs\ngrok.out.log" -RedirectStandardError "runs\ngrok.err.log" -WindowStyle Hidden -PassThru
}

$server = Start-Server
$tunnel = Start-Tunnel
Write-Host "service: http://localhost:$Port   public: https://$Domain   (Ctrl+C to stop)"
try {
    while ($true) {
        Start-Sleep -Seconds 10
        if ($server.HasExited) { Write-Host "$(Get-Date -f s) server exited ($($server.ExitCode)), restarting"; $server = Start-Server }
        if ($tunnel.HasExited) { Write-Host "$(Get-Date -f s) tunnel exited ($($tunnel.ExitCode)), restarting"; $tunnel = Start-Tunnel }
    }
}
finally {
    foreach ($p in @($server, $tunnel)) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force } }
}
