param(
    [int]$NumJobs = 4,
    [int]$BlasThreads = 1,
    [string]$CondaEnv = "neuro",
    [string]$PythonExe = "python",
    [string]$ProjectDir = "",
    [string]$OutputDir = "FinalPaper\results\normalization_grid",
    [string]$RunName = "",
    [string]$GridArgs = "",
    [switch]$NoConda,
    [switch]$NoExit,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ($NumJobs -lt 1) {
    throw "NumJobs must be >= 1."
}
if ($BlasThreads -lt 1) {
    throw "BlasThreads must be >= 1."
}
if ([string]::IsNullOrWhiteSpace($GridArgs)) {
    throw "Pass grid-search arguments with -GridArgs, for example: -GridArgs '--j-values 0.5 1.0 --g-values 1.0 2.0 --seed-count 5'"
}
if ([string]::IsNullOrWhiteSpace($ProjectDir)) {
    $ProjectDir = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
} else {
    $ProjectDir = (Resolve-Path -LiteralPath $ProjectDir).Path
}
if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "grid_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$generatedDir = Join-Path $PSScriptRoot "generated"
$logDir = Join-Path $generatedDir "logs"
New-Item -ItemType Directory -Force -Path $generatedDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$psExe = $PSHOME
if (-not [string]::IsNullOrWhiteSpace($psExe)) {
    $candidate = Join-Path $psExe "powershell.exe"
    if (Test-Path -LiteralPath $candidate) {
        $psExe = $candidate
    } else {
        $candidate = Join-Path $psExe "pwsh.exe"
        if (Test-Path -LiteralPath $candidate) {
            $psExe = $candidate
        } else {
            $psExe = "powershell.exe"
        }
    }
} else {
    $psExe = "powershell.exe"
}

function ConvertTo-SingleQuotedLiteral([string]$Value) {
    return "'" + ($Value -replace "'", "''") + "'"
}

$quotedProjectDir = ConvertTo-SingleQuotedLiteral $ProjectDir
$quotedCondaEnv = ConvertTo-SingleQuotedLiteral $CondaEnv
$quotedPythonExe = ConvertTo-SingleQuotedLiteral $PythonExe
$quotedOutputDir = ConvertTo-SingleQuotedLiteral $OutputDir

Write-Host "ProjectDir: $ProjectDir"
Write-Host "OutputDir:  $OutputDir"
Write-Host "RunName:    $RunName"
Write-Host "NumJobs:    $NumJobs"
Write-Host "BLAS vars:  OPENBLAS/OMP/MKL = $BlasThreads"
Write-Host "GridArgs:   $GridArgs"

for ($jobIndex = 0; $jobIndex -lt $NumJobs; $jobIndex++) {
    $jobName = "{0}_job{1:000}_of_{2:000}" -f $RunName, $jobIndex, $NumJobs
    $jobScript = Join-Path $generatedDir ($jobName + ".ps1")
    $logPath = Join-Path $logDir ($jobName + ".log")
    $quotedLogPath = ConvertTo-SingleQuotedLiteral $logPath

    $condaBlock = @"
if (-not `$NoCondaForJob) {
    conda activate $quotedCondaEnv
    if (`$LASTEXITCODE -ne 0) {
        throw "Conda activation failed for environment $CondaEnv."
    }
}
"@
    if ($NoConda) {
        $condaBlock = ""
    }

    $jobBody = @"
`$ErrorActionPreference = "Stop"
`$NoCondaForJob = `$$($NoConda.IsPresent.ToString().ToLowerInvariant())
Set-Location -LiteralPath $quotedProjectDir

`$env:OPENBLAS_NUM_THREADS = "$BlasThreads"
`$env:OMP_NUM_THREADS = "$BlasThreads"
`$env:MKL_NUM_THREADS = "$BlasThreads"

Start-Transcript -Path $quotedLogPath -Append | Out-Null
try {
    Write-Host "Job $jobIndex/$NumJobs started at `$(Get-Date)"
    Write-Host "Working directory: `$(Get-Location)"
    Write-Host "OPENBLAS_NUM_THREADS=`$env:OPENBLAS_NUM_THREADS"
    Write-Host "OMP_NUM_THREADS=`$env:OMP_NUM_THREADS"
    Write-Host "MKL_NUM_THREADS=`$env:MKL_NUM_THREADS"
$condaBlock
    `$cmd = "$PythonExe FinalPaper\grid_search_normalization_topology.py $GridArgs --output-dir $OutputDir --job-index $jobIndex --num-jobs $NumJobs"
    Write-Host `$cmd
    Invoke-Expression `$cmd
    if (`$LASTEXITCODE -ne 0) {
        throw "Grid job exited with code `$LASTEXITCODE."
    }
    Write-Host "Job $jobIndex/$NumJobs finished at `$(Get-Date)"
} finally {
    Stop-Transcript | Out-Null
}
"@

    Set-Content -LiteralPath $jobScript -Value $jobBody -Encoding UTF8
    Write-Host "Prepared $jobScript"

    if (-not $DryRun) {
        $windowArgs = @("-ExecutionPolicy", "Bypass")
        if ($NoExit) {
            $windowArgs += "-NoExit"
        }
        $windowArgs += @("-File", $jobScript)
        Start-Process -FilePath $psExe -ArgumentList $windowArgs -WorkingDirectory $ProjectDir
    }
}

$aggregateScript = Join-Path $generatedDir ($RunName + "_aggregate_and_analyze.ps1")
$aggregateBody = @"
`$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $quotedProjectDir
`$env:OPENBLAS_NUM_THREADS = "$BlasThreads"
`$env:OMP_NUM_THREADS = "$BlasThreads"
`$env:MKL_NUM_THREADS = "$BlasThreads"
$(if ($NoConda) { "" } else { "conda activate $quotedCondaEnv" })

$PythonExe FinalPaper\grid_search_normalization_topology.py --output-dir $OutputDir --aggregate-only
$PythonExe FinalPaper\analyze_normalization_grid.py --results-dir $OutputDir
"@
Set-Content -LiteralPath $aggregateScript -Value $aggregateBody -Encoding UTF8

Write-Host ""
Write-Host "Generated aggregate/analysis helper:"
Write-Host "  $aggregateScript"
Write-Host ""
if ($DryRun) {
    Write-Host "Dry run only: no job windows were launched."
} else {
    Write-Host "Launched $NumJobs PowerShell job window(s)."
}
