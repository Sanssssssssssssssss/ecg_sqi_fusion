param(
    [Parameter(Mandatory=$true)]
    [string]$Target,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$RunnerArgs
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$PythonArgs = @("run_reproduce.py", "--target", $Target) + $RunnerArgs
python @PythonArgs
exit $LASTEXITCODE
