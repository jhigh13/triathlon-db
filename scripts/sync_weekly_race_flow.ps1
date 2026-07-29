param(
    [switch]$SkipLocalBuild,
    [switch]$SkipRestore,
    [switch]$SkipDashboardSync
)

$ErrorActionPreference = 'Stop'

function Import-DotEnvFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        return
    }

    foreach ($rawLine in Get-Content $Path) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith('#')) {
            continue
        }

        $parts = $line -split '=', 2
        if ($parts.Count -ne 2) {
            continue
        }

        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if ($value.StartsWith('"') -and $value.EndsWith('"') -and $value.Length -ge 2) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        elseif ($value.StartsWith("'") -and $value.EndsWith("'") -and $value.Length -ge 2) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        if ($name) {
            Set-Item -Path "Env:$name" -Value $value
        }
    }
}

function Add-SslModeRequire {
    param(
        [string]$Uri
    )

    if ([string]::IsNullOrWhiteSpace($Uri)) {
        return $Uri
    }

    if ($Uri -match '^postgresql(\+[^:]+)?://') {
        # Keep local connections unchanged. SSL enforcement is only needed for remote poolers.
        $dbHost = $null
        if ($Uri -match '^postgresql(?:\+[^:]+)?://(?:[^@/]+@)?([^:/?]+)') {
            $dbHost = $matches[1]
        }
        if ($dbHost -and ($dbHost -in @('localhost', '127.0.0.1'))) {
            return $Uri
        }

        if ($Uri -match 'sslmode=') {
            return $Uri
        }

        if ($Uri.Contains('?')) {
            return "${Uri}&sslmode=require"
        }

        return "${Uri}?sslmode=require"
    }

    return $Uri
}

function Assert-Success {
    param([string]$Description)

    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Convert-ToLibpqUri {
    param(
        [string]$Uri
    )

    if ([string]::IsNullOrWhiteSpace($Uri)) {
        return $Uri
    }

    # pg_dump/pg_restore use libpq URIs and do not understand SQLAlchemy driver suffixes.
    if ($Uri -match '^postgresql\+[^:]+://') {
        return ($Uri -replace '^postgresql\+[^:]+://', 'postgresql://')
    }

    return $Uri
}

$triathlonRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $triathlonRoot
$dashboardRoot = Join-Path $workspaceRoot 'PodiumDashboard'

Import-DotEnvFile (Join-Path $triathlonRoot '.env')
Import-DotEnvFile (Join-Path $dashboardRoot '.env')

$localDbUri = Add-SslModeRequire $env:DB_URI
$remoteDbUri = Add-SslModeRequire $env:TRIATHLON_DATABASE_URL
$localDbUriLibpq = Convert-ToLibpqUri $localDbUri
$remoteDbUriLibpq = Convert-ToLibpqUri $remoteDbUri

if (-not $SkipLocalBuild) {
    Write-Host 'Running local triathlon-db build...'
    Push-Location $triathlonRoot
    try {
        & .\.venv\Scripts\python.exe .\tri_analysis\build_database.py
        Assert-Success 'build_database.py'
    }
    finally {
        Pop-Location
    }

    # Recompute event points so computed_weekly_rankings reflects new race results.
    Write-Host 'Recomputing event points from updated race results...'
    Push-Location $triathlonRoot
    try {
        # Piped via stdin (python -) rather than passed as a -c argument or a temp
        # script file: PowerShell mangles (strips) embedded double quotes when
        # passing multi-line here-strings as arguments to native executables, and
        # running a temp .py file elsewhere breaks the tri_analysis package import
        # (sys.path[0] becomes the temp file's directory instead of the CWD).
        # load_dotenv() is given an explicit path rather than relying on its
        # find_dotenv() stack-walking, which raises AssertionError when the code
        # is executed via stdin (there is no caller frame to inspect).
        @'
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), '.env'))
from tri_analysis.ranking_points import compute_event_points_batch
from tri_analysis.database import get_engine
engine = get_engine()
n = compute_event_points_batch(engine)
print(f"compute_event_points_batch: {n:,} rows written.")
'@ | & .\.venv\Scripts\python.exe -
        Assert-Success 'compute_event_points_batch'
    }
    finally {
        Pop-Location
    }
}

if (-not $SkipRestore) {
    if ([string]::IsNullOrWhiteSpace($localDbUri)) {
        throw 'DB_URI is not set.'
    }
    if ([string]::IsNullOrWhiteSpace($remoteDbUri)) {
        throw 'TRIATHLON_DATABASE_URL is not set.'
    }

    function Find-PgTool {
        param([string]$Name)

        $cmd = Get-Command $Name -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd.Source
        }

        # Fall back to common PostgreSQL install locations when not on PATH.
        $candidates = @(Get-ChildItem -Path 'C:\Program Files\PostgreSQL', 'C:\Program Files (x86)\PostgreSQL' -Directory -ErrorAction SilentlyContinue |
            Sort-Object { [int]($_.Name -as [int]) } -Descending |
            ForEach-Object { Join-Path $_.FullName "bin\$Name.exe" } |
            Where-Object { Test-Path $_ })

        if ($candidates.Count -gt 0) {
            return $candidates[0]
        }

        throw "$Name not found on PATH or under a standard PostgreSQL install directory. Install PostgreSQL client tools or add its 'bin' directory to PATH."
    }

    $pgDump = Find-PgTool 'pg_dump'
    $psql = Find-PgTool 'psql'
    $dumpPath = Join-Path ([System.IO.Path]::GetTempPath()) ("triathlon-db-{0}.sql" -f ([guid]::NewGuid().ToString('N')))

    try {
        Write-Host 'Dumping local triathlon-db to a temporary SQL file...'
        & $pgDump --format=plain --no-owner --no-privileges --clean --if-exists --dbname $localDbUriLibpq --file $dumpPath
        Assert-Success 'pg_dump'

        # PostgreSQL 17 emits transaction_timeout; older Supabase/Postgres targets reject it.
        Write-Host 'Restoring SQL dump into triathlon-db Supabase...'
        Get-Content $dumpPath |
            Where-Object { $_ -notmatch '^SET transaction_timeout = 0;$' } |
            & $psql --set ON_ERROR_STOP=1 --dbname $remoteDbUriLibpq
        Assert-Success 'psql restore'
    }
    finally {
        if (Test-Path $dumpPath) {
            Remove-Item $dumpPath -Force
        }
    }
}

if (-not $SkipDashboardSync) {
    Write-Host 'Syncing PodiumDashboard race cache from triathlon-db Supabase...'
    Push-Location $dashboardRoot
    try {
        # podium_cli.py prints unicode checkmark/cross glyphs; PYTHONUTF8 forces
        # UTF-8 stdout so it doesn't crash under the default Windows cp1252 console.
        $env:PYTHONUTF8 = '1'
        try {
            & .\venv\Scripts\python.exe .\podium_cli.py sync-races --mapped-only
            Assert-Success 'podium_cli.py sync-races'
        }
        finally {
            Remove-Item Env:\PYTHONUTF8 -ErrorAction SilentlyContinue
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host 'Weekly race flow complete.'