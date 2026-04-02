#!/usr/bin/env bash
# run_pipeline.sh — Run the full training pipeline end-to-end.
# Usage: bash run_pipeline.sh
# For morning signal only: python dashboard/morning_signal.py

set -e
cd "$(dirname "$0")"

echo "=== Step 1: Parse DBN files ==="
python data_pipeline/parse_dbn.py

echo ""
echo "=== Step 2: Fetch yfinance data ==="
python data_pipeline/fetch_yfinance.py

echo ""
echo "=== Step 3: Merge raw data ==="
python data_pipeline/merge.py

echo ""
echo "=== Step 4: Engineer features + labels ==="
python features/engineer.py

echo ""
echo "=== Step 5: Train XGBoost model ==="
python models/train.py

echo ""
echo "=== Step 6: Evaluate model ==="
python models/evaluate.py

echo ""
echo "=== Step 7: Run backtest ==="
python backtest/engine.py

echo ""
echo "=== Step 8: Grid search parameter sweep ==="
python grid_search/sweep.py

echo ""
echo "=== Pipeline complete ==="
echo "To run the morning signal dashboard:"
echo "  DATABENTO_API_KEY=<your_key> python dashboard/morning_signal.py"
