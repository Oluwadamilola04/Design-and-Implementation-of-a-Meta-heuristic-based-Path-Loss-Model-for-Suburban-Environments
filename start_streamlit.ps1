$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$streamlit = Join-Path $env:USERPROFILE ".conda\envs\pathloss\Scripts\streamlit.exe"

if (-not (Test-Path -LiteralPath $streamlit)) {
    Write-Error "Streamlit was not found in the pathloss environment: $streamlit"
}

Write-Host "Starting Path Loss Streamlit UI..."
Write-Host "Open http://127.0.0.1:8501 after Streamlit finishes loading."
Write-Host "Keep this terminal open while using the app."

& $streamlit run app.py --server.address 127.0.0.1 --server.port 8501
