#!/usr/bin/env bash
# CM-B.0b-repro — three independent draws of the SAME config, to finally put an
# error bar on this track.
#
# Every headline so far (2/8 vs 4/8, 7/25 vs 10/25, 5/8 vs 1/8) is one sample
# against another with unmeasured variance. Nothing in the pipeline was seeded
# until 2026-07-28, so run-to-run spread has never been observed even once.
#
# Draw 1 is the already-running unseeded job (an independent draw, just not
# replayable). Draws 2 and 3 are seeded so they can be replayed exactly.
#
# Each draw: distill -> eval -> archive the cartridge and the results JSON, so
# the next draw cannot clobber them. A failed distill skips its eval rather
# than scoring a stale cartridge from the previous draw.
set -uo pipefail

cd /Users/ecaterina/Developer/libucks || exit 1

WAIT_PID="${1:-}"
BUCKET=bc6b90e2
KV=/Users/ecaterina/Developer/test-repos/echoswarm/.libucks/kv_cache
CART="$KV/$BUCKET.cartridge.safetensors"

export CM_MODEL_QUERIES=0
export CM_NQUERIES=200
export CM_MAX_ANSWER_TOKENS=48

log() { echo "[seed-variance $(date '+%H:%M:%S')] $*"; }

if [[ -n "$WAIT_PID" ]]; then
  log "waiting for running distill (PID $WAIT_PID) ..."
  while ps -p "$WAIT_PID" >/dev/null 2>&1; do sleep 60; done
  log "PID $WAIT_PID exited"
fi

# ---- draw 1: score whatever the already-running unseeded job produced ----
if [[ -f "$CART" ]]; then
  log "draw 1 (unseeded) — evaluating"
  uv run python scripts/cm_eval_cartridge.py --buckets "$BUCKET" --tag s0_unseeded \
    >> cm_seed_variance.log 2>&1
  cp "$CART" "$KV/$BUCKET.cartridge.s0_unseeded.bak"
  log "draw 1 archived"
else
  log "draw 1 SKIPPED — no cartridge at $CART (the run failed?)"
fi

# ---- draws 2 and 3: seeded, replayable ----
for SEED in 1 2; do
  log "draw $((SEED + 1)) — distilling with CM_SEED=$SEED"
  if CM_SEED=$SEED uv run python scripts/cm_distill_buckets.py \
       --buckets "$BUCKET" --force >> cm_seed_variance.log 2>&1; then
    log "draw $((SEED + 1)) distilled — evaluating"
    uv run python scripts/cm_eval_cartridge.py --buckets "$BUCKET" --tag "s${SEED}" \
      >> cm_seed_variance.log 2>&1
    cp "$CART" "$KV/$BUCKET.cartridge.s${SEED}.bak"
    log "draw $((SEED + 1)) archived"
  else
    log "draw $((SEED + 1)) FAILED to distill — skipping its eval, not scoring a stale cartridge"
  fi
done

log "=== all draws done ==="
grep -hE "cartridge latent-alone grounding" cm_seed_variance.log || true
