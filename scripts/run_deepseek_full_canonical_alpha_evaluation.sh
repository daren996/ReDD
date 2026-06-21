#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

set -a
source .env
set +a

export LITELLM_LOG=ERROR
export LITELLM_LOG_LEVEL=ERROR
export LITELLM_SET_VERBOSE=False
export PYTHONUNBUFFERED=1
export REDD_LLM_USAGE_LOG=logs/deepseek_full_canonical_alpha/eval_llm_usage.jsonl

CONFIG=configs/examples/deepseek_full_canonical_alpha.yaml
EXP=full_canonical
LOGDIR=logs/deepseek_full_canonical_alpha

DATASETS=(
  bird.california_schools
  bird.debit_card_specializing
  bird.student_club
  cuad.cuad
  examples.single_doc_demo
  fda.fda
  galois.fortune
  galois.premierleague
  quest.lcr
  quest.swde_movie
  quest.swde_university
  quest.wikitext
  spider.apartment_rentals
  spider.bike_1
  spider.college_2
  spider.flight_4
  spider.soccer_1
  spider.wine_1
)

mkdir -p "$LOGDIR"
echo -e "timestamp\tdataset\tevent\tstatus" > "$LOGDIR/evaluation_status.tsv"
: > "$LOGDIR/evaluation_failures.txt"
: > "$LOGDIR/evaluation_run_all.log"

for ds in "${DATASETS[@]}"; do
  safe=${ds//[^A-Za-z0-9_.-]/_}
  echo -e "$(date -u +%Y-%m-%dT%H:%M:%SZ)\t$ds\tstart\t" | tee -a "$LOGDIR/evaluation_status.tsv" >> "$LOGDIR/evaluation_run_all.log"
  uv run redd evaluate --config "$CONFIG" --exp "$EXP" --dataset "$ds" --json >> "$LOGDIR/evaluation_${safe}.log" 2>&1
  status=$?
  echo -e "$(date -u +%Y-%m-%dT%H:%M:%SZ)\t$ds\tend\t$status" | tee -a "$LOGDIR/evaluation_status.tsv" >> "$LOGDIR/evaluation_run_all.log"
  if [ "$status" -ne 0 ]; then
    echo "$ds status=$status" >> "$LOGDIR/evaluation_failures.txt"
  fi
done
