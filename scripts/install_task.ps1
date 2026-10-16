<#
.SYNOPSIS
    Registers (or removes) a Windows Task Scheduler job that refreshes the
    pipeline unattended.

.DESCRIPTION
    The job runs the project's own virtualenv interpreter directly rather than
    a bare "python", because Task Scheduler starts with a minimal PATH that does
    not include conda/py launcher shims. It runs whether or not a user is
    logged on, and retries twice if the machine is busy at the trigger time.

.EXAMPLE
    .\scripts\install_task.ps1 -Time 02:30
    .\scripts\install_task.ps1 -Uninstall
    .\scripts\install_task.ps1 -AtLogon
#>

[CmdletBinding()]
param(
    [string]$Time = "03:00",
    [ValidateSet("Daily", "Weekly", "AtLogon")]
    [string]$Frequency = "Daily",
    [switch]$AtLogon,
    [switch]$Force,
    [switch]$SkipShap,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$TaskName = "StartupIntelligence-PipelineRefresh"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed task '$TaskName'."
    }
    else {
        Write-Host "Task '$TaskName' is not registered."
    }
    return
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtualenv interpreter not found at $Python. Create it first: python -m venv .venv"
}

$arguments = "`"$ProjectRoot\run_scheduled.py`""
if ($Force)     { $arguments += " --force" }
if ($SkipShap)  { $arguments += " --skip-shap" }

$action = New-ScheduledTaskAction -Execute $Python -Argument $arguments -WorkingDirectory $ProjectRoot

if ($AtLogon) {
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $trigger.Delay = "PT5M"
}
elseif ($Frequency -eq "Weekly") {
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At $Time
}
else {
    $trigger = New-ScheduledTaskTrigger -Daily -At $Time
}

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 15) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Refreshes models, database, charts and docs for Startup Intelligence AI." `
    -Force | Out-Null

Write-Host "Registered '$TaskName'."
Write-Host "  Interpreter : $Python"
Write-Host "  Arguments   : $arguments"
Write-Host "  Logs        : $(Join-Path $ProjectRoot 'logs\scheduled_run.log')"
Write-Host "  Run now     : Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Remove      : .\scripts\install_task.ps1 -Uninstall"