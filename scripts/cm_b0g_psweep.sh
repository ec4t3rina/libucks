#!/usr/bin/env bash
# CM-B.0g — confirm P=384 across seeds, then sweep P further.
#
# CM-B.0f found P was the binding constraint, not the recipe: at P=384 the
# cartridge scored 3/8 (floor 0) and 9/16 (floor 4) — c-floor of +3 and +5 —
# against +0.67 and +0.33 for three seeds at P=128. Initial KL fell 5.68 -> 3.63
# from the capacity change alone. Every negative in this track, Phase 4-C
# included, was measured at P=64 or P=128.
#
# That was ONE draw, and single draws have been wrong repeatedly here, so the
# confirmation comes first. best-epoch promotion is on throughout: the P=384 draw
# shipped its WORST epoch (4.7843 when epoch 2 was 3.3194) because KEEP_BEST was
# off, so there is free headroom, and keeping it on makes the sweep arms mutually
# comparable.
#
# INTERPRETIVE LIMIT, worth stating before the numbers arrive: bc6b90e2's verbatim
# is ~1009 tokens, so P=128 is 7.9x compression, P=384 is 2.6x, P=512 is 2.0x and
# P=768 is only 1.3x. As P approaches seq_len the cartridge stops being a
# compression of the cache and becomes a reparameterisation of it — at which point
# training-free KV-cache pruning would be the simpler answer. P=768 is therefore an
# UPPER BOUND / sanity check, not a proposal.
#
# A failed arm skips its own floor evals and the chain continues, so one OOM at
# high P cannot cost the rest.
set -uo pipefail
cd /Users/ecaterina/Developer/libucks || exit 1

WAIT_PID="${1:-}"
B=bc6b90e2
KV=/Users/ecaterina/Developer/test-repos/echoswarm/.libucks/kv_cache
EXT=tests/eval/fixtures/echoswarm_qa_bc6b90e2_ext.json

# CM-A.2's config, held fixed. Only P and the seed vary.
export CM_MODEL_QUERIES=0
export CM_NQUERIES=120
export CM_MAX_ANSWER_TOKENS=32
export CM_KEEP_BEST=1

log() { echo "[b0g $(date '+%H:%M:%S')] $*"; }

if [[ -n "$WAIT_PID" ]]; then
  log "waiting for PID $WAIT_PID ..."
  while ps -p "$WAIT_PID" >/dev/null 2>&1; do sleep 60; done
  log "PID $WAIT_PID exited"
fi

run_arm() {
  local tag="$1" p="$2" seed="$3"
  log "$tag — distilling P=$p seed=$seed keep_best=1"
  if CM_PREFIX_LEN="$p" CM_SEED="$seed" uv run python scripts/cm_distill_buckets.py \
       --buckets "$B" --force >> cm_b0g.log 2>&1; then
    log "$tag — floor arms (orig 8)"
    uv run python scripts/cm_floor.py --buckets "$B" \
      --tag "${tag}_orig8" >> cm_b0g.log 2>&1
    log "$tag — floor arms (ext 16)"
    uv run python scripts/cm_floor.py --buckets "$B" --fixtures "$EXT" \
      --tag "${tag}_ext16" >> cm_b0g.log 2>&1
    cp "$KV/$B.cartridge.safetensors" "$KV/$B.cartridge.${tag}.bak"
    log "$tag — done and archived"
  else
    log "$tag — DISTILL FAILED (OOM at high P?); skipping its floor arms, continuing"
  fi
}

# 1+2: confirm P=384 across seeds, with best-epoch on.
run_arm b0g_p384_s1 384 1
run_arm b0g_p384_s2 384 2
run_arm b0g_p384_s3 384 3

# 3: does it keep scaling?
run_arm b0g_p512_s1 512 1
run_arm b0g_p768_s1 768 1

log "=== b0g done ==="
grep -hE "RECIPE|PROMOTED|best epoch WAS|FAILED|fixtures from|floor:|random:|cartridge:|cartridge - floor" cm_b0g.log || true
echo
echo "Analyse:"
echo "  uv run python scripts/cm_variance_report.py --glob 'echoswarm_floor_b0g_p384_s*_orig8.json'"
