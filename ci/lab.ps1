# SPDX-License-Identifier: AGPL-3.0-or-later
# Cycle headless du laboratoire engram : /lab-run sur un protocole PRE-ENREGISTRE.
# Usage : pwsh -File ci/lab.ps1 experiments/EXP-AAAA-MM-JJ-slug.md [-Yolo]
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Protocol,
    [switch]$Yolo
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Protocol -PathType Leaf)) {
    [Console]::Error.WriteLine("Erreur : fichier introuvable : $Protocol")
    exit 2
}
$hasStatus = Select-String -LiteralPath $Protocol -Pattern '^Statut : PRE-ENREGISTRE$' -Quiet
if (-not $hasStatus) {
    [Console]::Error.WriteLine("Erreur : $Protocol ne porte pas 'Statut : PRE-ENREGISTRE' — un cycle headless exige un protocole approuvé par le PI.")
    exit 3
}

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path 'lab-logs' | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path 'lab-logs' "lab-$stamp.json"

$allowedTools = 'Read,Glob,Grep,Write,Edit,Bash(git *),Bash(.venv/Scripts/python *),Bash(.venv\Scripts\python *),Bash(python *),Bash(pytest *)'

$claudeArgs = @('-p', "/lab-run '$Protocol'", '--output-format', 'json', '--max-turns', '150')
if ($Yolo) {
    $claudeArgs += '--dangerously-skip-permissions'
} else {
    $claudeArgs += @('--permission-mode', 'acceptEdits', '--allowedTools', $allowedTools)
}

Write-Host "Labo engram — protocole : $Protocol — log : $log"
& claude @claudeArgs | Out-File -FilePath $log -Encoding utf8
$code = $LASTEXITCODE
Write-Host "claude terminé avec le code $code — sortie dans $log"
exit $code
