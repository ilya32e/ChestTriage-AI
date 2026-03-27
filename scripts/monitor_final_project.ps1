$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Write-Host "Services actifs:"
Get-NetTCPConnection -LocalPort 5000,5001,8501 -State Listen -ErrorAction SilentlyContinue |
    Select-Object LocalPort, OwningProcess, State |
    Format-Table -AutoSize

Write-Host ""
Write-Host "Dernieres lignes du runner:"
Get-Content .\artifacts\logs\project_runner.out.log -Tail 80

Write-Host ""
Write-Host "Dernieres lignes d'erreur:"
Get-Content .\artifacts\logs\project_runner.err.log -Tail 80
