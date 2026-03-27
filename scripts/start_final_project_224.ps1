$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

New-Item -ItemType Directory -Force -Path ".\artifacts\logs" | Out-Null

Get-NetTCPConnection -LocalPort 5000,8501 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }

Get-CimInstance Win32_Process |
    Where-Object {
        $_.CommandLine -like '*run_real_project.ps1*' -or
        $_.CommandLine -like '*run_final_project.ps1*' -or
        $_.CommandLine -like '*run_final_project_224.ps1*'
    } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Start-Sleep -Seconds 2

$mlflow = Start-Process python `
    -ArgumentList '-m','mlflow','ui','--backend-store-uri','file:./mlruns','--host','127.0.0.1','--port','5000' `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput '.\artifacts\logs\mlflow.out.log' `
    -RedirectStandardError '.\artifacts\logs\mlflow.err.log' `
    -PassThru

$runner = Start-Process powershell `
    -ArgumentList '-ExecutionPolicy','Bypass','-File','.\\scripts\\run_final_project_224.ps1' `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput '.\artifacts\logs\project_runner.out.log' `
    -RedirectStandardError '.\artifacts\logs\project_runner.err.log' `
    -PassThru

Start-Sleep -Seconds 5

Write-Output "MLFLOW_PID=$($mlflow.Id)"
Write-Output "RUNNER_PID=$($runner.Id)"
