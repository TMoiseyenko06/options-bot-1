# run_pipeline.ps1 — Run the full training pipeline on Windows.
# Usage: .\run_pipeline.ps1
# For morning signal only: $env:DATABENTO_API_KEY="your_key"; python dashboard/morning_signal.py

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== Step 1: Parse DBN files ===" -ForegroundColor Cyan
python data_pipeline/parse_dbn.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Step 2: Fetch yfinance data ===" -ForegroundColor Cyan
python data_pipeline/fetch_yfinance.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Step 3: Merge raw data ===" -ForegroundColor Cyan
python data_pipeline/merge.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Step 4: Engineer features + labels ===" -ForegroundColor Cyan
python features/engineer.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Step 5: Train XGBoost model ===" -ForegroundColor Cyan
python models/train.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Step 6: Evaluate model ===" -ForegroundColor Cyan
python models/evaluate.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Step 7: Run backtest ===" -ForegroundColor Cyan
python backtest/engine.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Step 8: Grid search parameter sweep ===" -ForegroundColor Cyan
python grid_search/sweep.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Pipeline complete ===" -ForegroundColor Green
Write-Host "To run the morning signal dashboard:"
Write-Host '  $env:DATABENTO_API_KEY="your_key"; python dashboard/morning_signal.py'
