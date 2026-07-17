#!/usr/bin/env bash
# Runs on every Codespace start/attach (postStartCommand).
# Starts `python3 bot.py` in the background so you don't have to run it
# by hand - logs go to bot.log, PID tracked in bot.pid so restarts don't
# leave duplicate bots polling Telegram (which causes conflicting getUpdates
# errors).
set -uo pipefail

cd "$(dirname "$0")/.."

PID_FILE="bot.pid"
LOG_FILE="bot.log"

# If a bot from a previous start is still alive, kill it first so we never
# end up with two instances polling the same bot token at once.
if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "==> Stopping previous bot instance (PID $OLD_PID)..."
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
  fi
fi

if [ -z "${API_ID:-}" ] || [ -z "${API_HASH:-}" ] || [ -z "${BOT_TOKEN:-}" ]; then
  echo "=================================================================="
  echo " Skipping auto-start: API_ID / API_HASH / BOT_TOKEN not set."
  echo " Add them as Codespaces secrets (repo Settings > Secrets and"
  echo " variables > Codespaces), then rebuild/restart the Codespace."
  echo "=================================================================="
  exit 0
fi

echo "==> Starting bot.py in background (logs: $LOG_FILE)..."
nohup python3 bot.py > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
disown

sleep 2
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "==> Bot started (PID $(cat "$PID_FILE")). Tail logs with: tail -f bot.log"
else
  echo "==> Bot failed to start - check bot.log:"
  tail -n 40 "$LOG_FILE" || true
fi
