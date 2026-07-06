param(
    [Parameter(Mandatory=$true)]
    [string]$Target,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$RunnerArgs
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $Root

$Venv = Join-Path $Root "reproduce\work\.venv"
$Python = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    python -m venv $Venv
}

$Marker = Join-Path $Venv ".ecg_sqi_fusion_installed"
try {
    & $Python -c "import src, pandas, numpy" *>$null
    $ImportOk = ($LASTEXITCODE -eq 0)
} catch {
    $ImportOk = $false
}
if ($ImportOk -and -not (Test-Path $Marker)) {
    Set-Content -Path $Marker -Value (Get-Date).ToString("o")
}
if (-not (Test-Path $Marker)) {
    & $Python -m pip install --disable-pip-version-check -e .
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Set-Content -Path $Marker -Value (Get-Date).ToString("o")
}

$PythonArgs = @("reproduce/run_reproduce.py", "--target", $Target) + $RunnerArgs
& $Python @PythonArgs
exit $LASTEXITCODE
