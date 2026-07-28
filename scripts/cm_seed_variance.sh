#!/usr/bin/env bash
# Run one distill config N times with different seeds, to get an error bar.
#
# Every headline in this track was a single sample until 2026-07-28: 2/8 vs 4/8,
# 7/25 vs 10/25, 5/8 vs 1/8. The first three-draw run (CM-B.0b-repro) measured a
# spread of 1 on an 8-fixture test, which is what finally made those comparisons
# interpretable — and showed the "proven" 200q/48tok recipe to be reproducibly
# worse than CM-A.2's 120q/32tok.
#
# Each draw: distill -> eval -> archive cartridge and results, so no draw can
# clobber another. A failed distill skips its eval rather than scoring the
# previous draw's stale cartridge.
#
# Config comes from the environment, so the caller states it explicitly and it
# lands in the RECIPE banner of every log. Nothing is defaulted silently here.
#
# Usage:
#   CM_TAG=b0d CM_SEEDS="1 2 3" \
#   CM_MODEL_QUERIES=0 CM_NQUERIES=120 CM_MAX_ANSWER_TOKENS=32 \
#     nohup caffeinate -dimsu bash scripts/cm_seed_variance.sh [WAIT_PID] &
set -uo pipefail

cd /Users/ecaterina/Developer/libucks || exit 1

WAIT_PID="${1:-}"
BUCKET="${CM_BUCKET:-bc6b90e2}"
TAG="${CM_TAG:?set CM_TAG, e.g. b0d — it names the archived cartridges and results}"
SEEDS="${CM_SEEDS:?set CM_SEEDS, e.g. \"1 2 3\"}"
KV=/Users/ecaterina/Developer/test-repos/echoswarm/.libucks/kv_cache
CART="$KV/$BUCKET.cartridge.safetensors"
LOG="cm_${TAG}.log"

# Passed through to the distiller by name so the banner records them.
export CM_MODEL_QUERIES CM_NQUERIES CM_MAX_ANSWER_TOKENS

log() { echo "[$TAG $(date '+%H:%M:%S')] $*"; }

if [[ -n "$WAIT_PID" ]]; then
  log "waiting for PID $WAIT_PID ..."
  while ps -p "$WAIT_PID" >/dev/null 2>&1; do sleep 60; done
  log "PID $WAIT_PID exited"
fi

log "config: bucket=$BUCKET seeds=[$SEEDS] model_queries=${CM_MODEL_QUERIES:-unset}" \
    "nqueries=${CM_NQUERIES:-unset} max_answer_tokens=${CM_MAX_ANSWER_TOKENS:-unset}"

for SEED in $SEEDS; do
  log "draw seed=$SEED — distilling"
  if CM_SEED="$SEED" uv run python scripts/cm_distill_buckets.py \
       --buckets "$BUCKET" --force >> "$LOG" 2>&1; then
    log "draw seed=$SEED distilled — evaluating (default 25-fixture set)"
    uv run python scripts/cm_eval_cartridge.py --buckets "$BUCKET" \
      --tag "${TAG}_s${SEED}" >> "$LOG" 2>&1

    # Second, better-powered read on the same cartridge. Separate fixture set =
    # separate denominator, so these scores are never mixed with the 25-set.
    log "draw seed=$SEED — evaluating extension set (16 fixtures)"
    uv run python scripts/cm_eval_cartridge.py --buckets "$BUCKET" \
      --fixtures tests/eval/fixtures/echoswarm_qa_bc6b90e2_ext.json \
      --tag "${TAG}ext_s${SEED}" >> "$LOG" 2>&1

    cp "$CART" "$KV/$BUCKET.cartridge.${TAG}_s${SEED}.bak"
    log "draw seed=$SEED archived"
  else
    log "draw seed=$SEED FAILED to distill — skipping its evals, not scoring a stale cartridge"
  fi
done

log "=== all draws done ==="
grep -hE "cartridge latent-alone grounding|fixture set:" "$LOG" || true
echo
echo "Analyse with:"
echo "  uv run python scripts/cm_variance_report.py --glob 'echoswarm_cartridge_A2_${TAG}_s*.json'"
echo "  uv run python scripts/cm_variance_report.py --glob 'echoswarm_cartridge_A2_${TAG}ext_s*.json'"
