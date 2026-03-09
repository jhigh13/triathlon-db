# Para Standards Batch Report Generator
# Generates reports for all para categories with auto-selected first USA athlete
# Run from the triathlon-db repo root:  .\para_triathlon_analysis\scripts\generate_reports.ps1

$PYTHON = ".venv/Scripts/python.exe"
$CATEGORIES = @(
    "PTWC Men",
    "PTS2 Men",
    "PTS3 Men",
    "PTS4 Men",
    "PTS5 Men",
    "PTVI Men",
    "PTWC Women",
    "PTS2 Women",
    "PTS3 Women",
    "PTS4 Women",
    "PTS5 Women",
    "PTVI Women"
)

$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$BASE_OUTPUT_DIR = "para_triathlon_analysis\output\para_standards_batch_$TIMESTAMP"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Para Standards Batch Report Generator" -ForegroundColor Cyan
Write-Host "Timestamp: $TIMESTAMP" -ForegroundColor Cyan
Write-Host "Output Directory: $BASE_OUTPUT_DIR" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Step 1: Calculate position metrics for latest events
Write-Host "[Step 1/3] Calculating position metrics for latest events..." -ForegroundColor Yellow
try {
    & $PYTHON -m tri_analysis.metrics --latest-events
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] Position metrics calculated`n" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Position metrics calculation had issues (continuing anyway)`n" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[WARNING] Could not calculate position metrics: $($_.Exception.Message)`n" -ForegroundColor Yellow
}

# Step 2: Generate HTML reports
Write-Host "[Step 2/3] Generating HTML reports for all categories...`n" -ForegroundColor Yellow

$SUCCESS_COUNT = 0
$FAILED_CATEGORIES = @()

foreach ($CATEGORY in $CATEGORIES) {
    $SAFE_NAME = $CATEGORY -replace ' ', '_'
    $CATEGORY_OUTPUT = "$BASE_OUTPUT_DIR\$SAFE_NAME"
    
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Processing: $CATEGORY" -ForegroundColor Yellow
    Write-Host "Output: $CATEGORY_OUTPUT" -ForegroundColor Gray
    
    try {
        & $PYTHON -m para_triathlon_analysis.para_standards `
            --category "$CATEGORY" `
            --since "2021-01-01" `
            --no-png `
            --auto-select-first `
            --outdir "$CATEGORY_OUTPUT"
        
        if ($LASTEXITCODE -eq 0) {
            $SUCCESS_COUNT++
            Write-Host "[SUCCESS] $CATEGORY" -ForegroundColor Green
        } else {
            $FAILED_CATEGORIES += $CATEGORY
            Write-Host "[FAILED] $CATEGORY (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        }
    }
    catch {
        $FAILED_CATEGORIES += $CATEGORY
        Write-Host "[ERROR] $CATEGORY - $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "HTML Report Generation Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Successful: $SUCCESS_COUNT / $($CATEGORIES.Count)" -ForegroundColor Green

if ($FAILED_CATEGORIES.Count -gt 0) {
    Write-Host "Failed Categories:" -ForegroundColor Red
    foreach ($CAT in $FAILED_CATEGORIES) {
        Write-Host "  - $CAT" -ForegroundColor Red
    }
}

# Step 3: Convert HTML reports to PDF
Write-Host "`n[Step 3/3] Converting HTML reports to PDF..." -ForegroundColor Yellow

$PDF_SCRIPT = "para_triathlon_analysis/scripts/convert_to_pdf.py"
if (Test-Path $PDF_SCRIPT) {
    try {
        & $PYTHON $PDF_SCRIPT "$BASE_OUTPUT_DIR"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[SUCCESS] PDF conversion complete" -ForegroundColor Green
        } else {
            Write-Host "[INFO] PDF conversion had issues - you can manually convert via browser print" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[INFO] Could not auto-convert to PDF: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "You can manually convert by opening each report.html in Chrome/Edge and printing to PDF" -ForegroundColor Gray
    }
} else {
    Write-Host "[INFO] convert_reports_to_pdf.py not found - skipping PDF conversion" -ForegroundColor Yellow
    Write-Host "You can manually convert by opening each report.html in Chrome/Edge and printing to PDF" -ForegroundColor Gray
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Batch Processing Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All reports saved to: $BASE_OUTPUT_DIR" -ForegroundColor Cyan
