#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

LOG_DIR="outputs/logs/deepseek_full_canonical_alpha"
mkdir -p "$LOG_DIR"

STATUS_FILE="$LOG_DIR/repaired_evaluation_status.tsv"
FAIL_FILE="$LOG_DIR/repaired_evaluation_failures.txt"
: > "$STATUS_FILE"
: > "$FAIL_FILE"
printf "timestamp\tdataset\tevent\tstatus\n" >> "$STATUS_FILE"

datasets=(
  "bird.california_schools"
  "bird.debit_card_specializing"
  "bird.student_club"
  "galois.fortune"
  "spider.apartment_rentals"
  "spider.bike_1"
  "spider.flight_4"
  "spider.soccer_1"
  "spider.wine_1"
)

for ds in "${datasets[@]}"; do
  printf "%s\t%s\tstart\t\n" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$ds" >> "$STATUS_FILE"
  uv run redd evaluate \
    --config configs/examples/deepseek_full_canonical_alpha.yaml \
    --exp full_canonical \
    --dataset "$ds" \
    --json > "$LOG_DIR/repaired_evaluation_${ds}.log" 2>&1
  status=$?
  printf "%s\t%s\tend\t%s\n" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$ds" "$status" >> "$STATUS_FILE"
  if [ "$status" -ne 0 ]; then
    printf "%s\t%s\n" "$ds" "$status" >> "$FAIL_FILE"
  fi
done
