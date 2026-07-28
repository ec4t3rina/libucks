#!/usr/bin/env bash
# Run the three floor arms against EVERY archived cartridge of one config.
#
# The first floor run used whatever happened to be in place (b0d seed 3) and gave
# cartridge - random = +0 on the original 8 fixtures. Single draws have been
# misleading all through this track — a one-fixture difference was twice read as
# signal — so the contrast has to hold across seeds before it is quoted.
#
# cm_floor.py reads <bucket>.cartridge.safetensors, so each archived draw is
# swapped into place in turn. The original file is saved first and restored on
# exit, including on interrupt, so a crash cannot leave the wrong cartridge
# installed under the canonical name.
set -uo pipefail

cd /Users/ecaterina/Developer/libucks || exit 1

BUCKET="${CM_BUCKET:-bc6b90e2}"
TAG="${CM_TAG:?set CM_TAG, e.g. b0d — selects <bucket>.cartridge.<TAG>_s<N>.bak}"
SEEDS="${CM_SEEDS:?set CM_SEEDS, e.g. \"1 2 3\"}"
KV=/Users/ecaterina/Developer/test-repos/echoswarm/.libucks/kv_cache
CART="$KV/$BUCKET.cartridge.safetensors"
KEEP="$KV/$BUCKET.cartridge.INPLACE-BEFORE-FLOOR.bak"
EXT=tests/eval/fixtures/echoswarm_qa_bc6b90e2_ext.json

log() { echo "[floor-all $(date '+%H:%M:%S')] $*"; }

[[ -f "$CART" ]] || { log "FATAL: no cartridge at $CART"; exit 1; }
cp "$CART" "$KEEP"
log "saved in-place cartridge -> $(basename "$KEEP")"

restore() {
  if [[ -f "$KEEP" ]]; then
    cp "$KEEP" "$CART" && log "restored original in-place cartridge"
  fi
}
trap restore EXIT INT TERM

for SEED in $SEEDS; do
  SRC="$KV/$BUCKET.cartridge.${TAG}_s${SEED}.bak"
  if [[ ! -f "$SRC" ]]; then
    log "seed $SEED SKIPPED — no archive at $(basename "$SRC")"
    continue
  fi
  cp "$SRC" "$CART"
  log "seed $SEED installed — running floor (orig 8)"
  uv run python scripts/cm_floor.py --buckets "$BUCKET" \
    --tag "${TAG}s${SEED}_orig8" >> cm_floor_all.log 2>&1
  log "seed $SEED — running floor (ext 16)"
  uv run python scripts/cm_floor.py --buckets "$BUCKET" --fixtures "$EXT" \
    --tag "${TAG}s${SEED}_ext16" >> cm_floor_all.log 2>&1
  log "seed $SEED done"
done

log "=== all seeds done ==="
grep -hE "fixtures from|floor:|random:|cartridge:|cartridge -" cm_floor_all.log || true
