#!/usr/bin/env bash
# run_morning_signal.sh — Run the morning signal dashboard.
#
# Designed to be called by cron at 9:00am ET (Mon–Fri).
# Logs to /var/log/options-bot/morning_signal.log
#
# Cron setup (run: crontab -e):
#   If your system clock is UTC (common on NAS/servers):
#     0 13 * * 1-5 /mnt/user/RadvestMain/Tim/options-bot-1/run_morning_signal.sh
#   If your system clock is already ET:
#     0 9  * * 1-5 /mnt/user/RadvestMain/Tim/options-bot-1/run_morning_signal.sh
#   To check your system timezone: timedatectl | grep "Time zone"

set -e
cd "$(dirname "$0")"

LOG_DIR="/var/log/options-bot"
LOG_FILE="$LOG_DIR/morning_signal.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Create log dir if it doesn't exist
mkdir -p "$LOG_DIR"

echo "" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"
echo "[$DATE] Starting morning signal" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"

# Use the python from the venv if it exists, else system python
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
else
    PYTHON="python3"
fi

echo "[$DATE] Using python: $PYTHON" >> "$LOG_FILE"

$PYTHON dashboard/morning_signal.py >> "$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done" >> "$LOG_FILE"
