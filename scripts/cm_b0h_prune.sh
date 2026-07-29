#!/usr/bin/env bash
# CM-B.0h — does distillation beat slicing the real KV cache?
#
# CM-B.0g drove KL to 1.09 at P=768 and grounding did NOT follow: P=768 scored
# 2/8 while P=512 scored 4/8 on a 3x worse KL. That decoupling says the cartridge
# is fitting the 120 self-study queries rather than learning the document. If so,
# a training-free slice of the REAL cache — which contains the identifiers
# outright and never has to relearn them — should be competitive.
#
# `cartridge - kv_first` is the number that matters: kv_first is provably
# identical to init_from_extracted_kv, so that delta is exactly what the gradient
# steps add. It has never been measured in this project.
#
# TWO SETTINGS, deliberately:
#   p384_s2  the best-scoring P=384 draw (9/16, 4/8). 2.6x compression, so
#            selection actually has to discard most of the document.
#   p768_s1  the best-KL draw. 768 of ~1009 positions, i.e. barely compressed —
#            kv_first here is nearly the full cache and should be a STRONG
#            baseline. If distillation cannot beat it, the mechanism is the
#            problem, not the budget.
#
# The in-place cartridge is saved and restored under a trap, so an interrupt
# cannot leave the wrong one installed under the canonical name.
set -uo pipefail
cd /Users/ecaterina/Developer/libucks || exit 1

B=bc6b90e2
KV=/Users/ecaterina/Developer/test-repos/echoswarm/.libucks/kv_cache
CART="$KV/$B.cartridge.safetensors"
KEEP="$KV/$B.cartridge.INPLACE-BEFORE-PRUNE.bak"
EXT=tests/eval/fixtures/echoswarm_qa_bc6b90e2_ext.json

log() { echo "[b0h $(date '+%H:%M:%S')] $*"; }

[[ -f "$CART" ]] || { log "FATAL: no cartridge at $CART"; exit 1; }
cp "$CART" "$KEEP"
log "saved in-place cartridge -> $(basename "$KEEP")"
restore() { [[ -f "$KEEP" ]] && cp "$KEEP" "$CART" && log "restored in-place cartridge"; }
trap restore EXIT INT TERM

for TAG in b0g_p384_s2 b0g_p768_s1; do
  SRC="$KV/$B.cartridge.${TAG}.bak"
  if [[ ! -f "$SRC" ]]; then
    log "$TAG SKIPPED — no archive"
    continue
  fi
  cp "$SRC" "$CART"
  log "$TAG installed — pruning arms (orig 8)"
  uv run python scripts/cm_kv_prune.py --buckets "$B" \
    --tag "${TAG}_orig8" >> cm_b0h.log 2>&1
  log "$TAG — pruning arms (ext 16)"
  uv run python scripts/cm_kv_prune.py --buckets "$B" --fixtures "$EXT" \
    --tag "${TAG}_ext16" >> cm_b0h.log 2>&1
  log "$TAG done"
done

log "=== b0h done ==="
grep -hE "compression|floor:|kv_first:|kv_last:|kv_stride:|kv_norm:|cartridge:|VERDICT|Note:" cm_b0h.log || true
