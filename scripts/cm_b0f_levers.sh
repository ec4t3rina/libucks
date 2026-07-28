#!/usr/bin/env bash
# CM-B.0f — the two objections that survive the bug audit.
#
# CM-B.0e found cartridge - random = +0 on the trustworthy fixture set. Most of
# the defects found in the audit bias DOWNWARD (verbatim truncation, unreachable
# fixtures, template padding, last-epoch save), and cartridge-minus-random is a
# within-run contrast where nearly all of them cancel: all three arms share the
# fixtures, metric, decoder and model. Two objections survive, and neither has
# ever been tested:
#
#   best   every run in this track shipped the LAST epoch. CM-B.0d's KL rose at
#          epoch 2 in all three seeds, so the promoted cartridge was repeatedly
#          not the best one trained.
#   p384   P=128 has never been varied. The prefix may simply lack the capacity
#          to hold the bucket's identifiers.
#
# One seed each, both scored against all three floor arms. If the cartridge still
# fails to beat a random prefix under both, the negative stands with its plausible
# objections closed rather than merely unexamined.
set -uo pipefail
cd /Users/ecaterina/Developer/libucks || exit 1

WAIT_PID="${1:-}"
B=bc6b90e2
KV=/Users/ecaterina/Developer/test-repos/echoswarm/.libucks/kv_cache
EXT=tests/eval/fixtures/echoswarm_qa_bc6b90e2_ext.json

# Held fixed across both arms — CM-A.2's config, the best-scoring one on record.
export CM_MODEL_QUERIES=0
export CM_NQUERIES=120
export CM_MAX_ANSWER_TOKENS=32
export CM_SEED=1

log() { echo "[b0f $(date '+%H:%M:%S')] $*"; }

if [[ -n "$WAIT_PID" ]]; then
  log "waiting for PID $WAIT_PID ..."
  while ps -p "$WAIT_PID" >/dev/null 2>&1; do sleep 60; done
  log "PID $WAIT_PID exited"
fi

run_arm() {
  local tag="$1"; shift
  log "$tag — distilling with $*"
  if env "$@" uv run python scripts/cm_distill_buckets.py \
       --buckets "$B" --force >> cm_b0f.log 2>&1; then
    log "$tag — floor arms (orig 8)"
    uv run python scripts/cm_floor.py --buckets "$B" \
      --tag "${tag}_orig8" >> cm_b0f.log 2>&1
    log "$tag — floor arms (ext 16)"
    uv run python scripts/cm_floor.py --buckets "$B" --fixtures "$EXT" \
      --tag "${tag}_ext16" >> cm_b0f.log 2>&1
    cp "$KV/$B.cartridge.safetensors" "$KV/$B.cartridge.${tag}.bak"
    log "$tag — done and archived"
  else
    log "$tag — DISTILL FAILED, skipping its floor arms rather than scoring a stale cartridge"
  fi
}

run_arm b0f_best CM_KEEP_BEST=1
run_arm b0f_p384 CM_PREFIX_LEN=384

log "=== b0f done ==="
grep -hE "RECIPE|PROMOTED|best epoch WAS|fixtures from|floor:|random:|cartridge:|cartridge -|VERDICT" cm_b0f.log || true
