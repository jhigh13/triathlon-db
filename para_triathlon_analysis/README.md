# Para Triathlon Standards Analysis

Tools to generate para triathlon standards reports — benchmark USA athletes against Paris 2024 medalists across all 12 para categories.

## Folder Structure
```
para_triathlon_analysis/
├── README.md              # This file
├── __init__.py            # Package marker
├── para_standards.py      # Main analysis module (2300+ lines)
├── scripts/
│   ├── generate_reports.ps1   # Batch runner for all 12 categories
│   └── convert_to_pdf.py      # HTML → PDF via headless Chrome/Edge
└── output/                    # Generated reports (timestamped batches)
    └── para_standards_batch_YYYYMMDD_HHMMSS/
        ├── PTS2_Men/  (dataset.csv + report.html)
        └── ...        (12 category dirs total)
```

## Quick Start

### 1. Generate All Reports (HTML + CSV)

From the repo root:
```powershell
.\para_triathlon_analysis\scripts\generate_reports.ps1
```

This will:
- Calculate position metrics for latest events
- Generate reports for all 12 para categories (Men: PTWC, PTS2-5, PTVI; Women: PTWC, PTS2-5, PTVI)
- Auto-select the first (most active) USA athlete in each category
- Automatically convert HTML reports to PDF
- Save outputs to timestamped directory: `para_triathlon_analysis/output/para_standards_batch_YYYYMMDD_HHMMSS/`
- Each category gets its own subdirectory with `report.html`, `report.pdf`, and `dataset.csv`

### 2. Manual PDF Conversion (If Needed)

If automated PDF conversion fails, you can manually convert:

For each category:
1. Open `report.html` in Chrome or Edge
2. Press `Ctrl+P` or click Print
3. Select "Save as PDF" as printer
4. Click Save

You can also run PDF conversion separately:

```powershell
python para_triathlon_analysis/scripts/convert_to_pdf.py para_triathlon_analysis/output/para_standards_batch_YYYYMMDD_HHMMSS
```

## Generated Files

Each category directory contains:
- **`report.html`** - Interactive HTML report with charts and benchmarks
- **`report.pdf`** - PDF version suitable for email/printing
- **`dataset.csv`** - Raw data used for the report

## Report Contents

Each report includes:
- **Selected USA Athlete** - Top USA competitor by event participation
- **Benchmark Athletes**:
  - Paris 2024 medalists (positions 1-3)
  - Additional top contenders specific to each category
- **Time-Factor Audit** (PTVI/PTWC only) - Transparency for handicap timing
- **Benchmarks**:
  - Win Gold: Compare vs Paris gold medalist only
  - Gold Contention: Weighted comparison (0.5/0.3/0.2 for medalists, 0.15 for additional contenders)
- **Performance Metrics**:
  - Swim pace, bike speed, run pace
  - Transitions (T1, T2)
  - Total time (placing frame for PTVI/PTWC)
  - Composite score (unitless normalized advantage)
- **Projection Table** - Gap analysis showing improvements needed

## Manual Report Generation

To generate a single category with custom athlete selection:

```powershell
python -m para_triathlon_analysis.para_standards --category "PTVI Men" --since 2021-01-01 --no-png
```

Or auto-select first athlete:

```powershell
python -m para_triathlon_analysis.para_standards --category "PTVI Men" --since 2021-01-01 --no-png --auto-select-first
```

## Available Options

```
--category "Category Name"      # Required: e.g., "PTVI Men", "PTWC Women"
--since YYYY-MM-DD              # Start date for historical data (default: 2021-01-01)
--usa-athlete-name "Name"       # Optional: specific USA athlete name
--auto-select-first             # Auto-select first athlete (no prompt)
--no-png                        # Skip PNG chart generation (HTML only)
--outdir path/to/output         # Custom output directory
--no-factor-normalization       # Disable time-factor handling (PTVI/PTWC)
```

## Troubleshooting

### PTS3 Women Uses Pontevedra 2023
- PTS3 Women was not a medal event at Paris 2024
- Instead uses 2023 World Para Championships in Pontevedra as benchmark
- Top 3: Elise Marc, Kenia Yesenia Villalobos Vargas, Sanne Koopman

### PDF Conversion Fails
- Requires Chrome or Edge installed
- Script will exit with error if PDFs cannot be created
- Fallback: Manual conversion via browser print (Ctrl+P → Save as PDF)

### Missing Athletes in Additional Benchmarks
- Warning messages indicate athlete names don't match database exactly
- Reports still generate with found athletes
- Check athlete names in database if updates needed

## Categories Covered

**Men**: PTWC, PTS2, PTS3, PTS4, PTS5, PTVI  
**Women**: PTWC, PTS2, PTS3*, PTS4, PTS5, PTVI  

*PTS3 Women will fail (not a Paris medal event)

## Output Example

```
tri_analysis/outputs/para_standards_batch_20260128_142005/
├── PTWC_Men/
│   ├── report.html
│   ├── report.pdf
│   └── dataset.csv
├── PTS2_Men/
│   ├── report.html
│   ├── report.pdf
│   └── dataset.csv
├── ...
└── PTVI_Women/
    ├── report.html
    ├── report.pdf
    └── dataset.csv
```

## Email Distribution

PDFs are optimized for email attachments:
- Clean layout with summary tables upfront
- Interactive Plotly charts (HTML version)
- CSV data for further analysis
- Typical PDF size: 200-400 KB per report
