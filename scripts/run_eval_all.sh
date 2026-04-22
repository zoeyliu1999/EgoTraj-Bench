#!/bin/bash
# -------------------------------------------------------
# Evaluate BiFlow on all T2FPV-ETH folds
# Usage:
#   # A) release checkpoints (default, from checkpoints/T2FPV-*)
#   bash scripts/run_eval_all.sh --source release
#
#   # B) training outputs (provide fold directories)
#   bash scripts/run_eval_all.sh --source results --ckpt_base results/biflow_t2fpv_k20
# -------------------------------------------------------

set -e

GPU="${GPU:-0}"
SOURCE="${SOURCE:-release}"   # release | results
CKPT_BASE=""
RELEASE_ROOT="${RELEASE_ROOT:-checkpoints}"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --source) SOURCE="$2"; shift 2 ;;
        --ckpt_base) CKPT_BASE="$2"; shift 2 ;;
        --release_root) RELEASE_ROOT="$2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

if [ "$SOURCE" != "release" ] && [ "$SOURCE" != "results" ]; then
    echo "Error: --source must be one of: release, results"
    exit 1
fi

if [ "$SOURCE" = "results" ] && [ -z "$CKPT_BASE" ]; then
    echo "Error: --ckpt_base is required when --source results"
    echo "Usage: bash scripts/run_eval_all.sh --source results --ckpt_base <path>"
    exit 1
fi

T2FPV_FOLDS=("eth" "hotel" "univ" "zara1" "zara2")

for FOLD in "${T2FPV_FOLDS[@]}"; do
    echo ""
    echo "=============================="
    echo "  Evaluating fold: $FOLD"
    echo "=============================="

    if [ "$SOURCE" = "release" ]; then
        RELEASE_NAME="T2FPV-${FOLD}"
        bash scripts/run_eval.sh \
            --fold_name "$FOLD" \
            --release_root "$RELEASE_ROOT" \
            --release_name "$RELEASE_NAME" \
            --gpu "$GPU" \
            $EXTRA_ARGS
    else
        # Compatibility: support both <ckpt_base>/<fold>/ and <ckpt_base>/<fold>/ckpt/
        CKPT_DIR="${CKPT_BASE}/${FOLD}"
        if [ -d "${CKPT_BASE}/${FOLD}/ckpt" ]; then
            CKPT_DIR="${CKPT_BASE}/${FOLD}/ckpt"
        fi
        if [ ! -d "$CKPT_DIR" ]; then
            echo "[SKIP] $CKPT_DIR not found"
            continue
        fi
        bash scripts/run_eval.sh \
            --ckpt_dir "$CKPT_DIR" \
            --fold_name "$FOLD" \
            --gpu "$GPU" \
            $EXTRA_ARGS
    fi
done

echo ""
echo "=== All T2FPV folds evaluated ==="
