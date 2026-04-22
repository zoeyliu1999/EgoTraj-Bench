#!/bin/bash
# -------------------------------------------------------
# Train BiFlow on EgoTraj-Bench
# Usage:
#   bash scripts/run_train.sh                       # EgoTraj-TBD (default)
#   bash scripts/run_train.sh --fold_name eth        # T2FPV-ETH
# -------------------------------------------------------

set -e

# ---------- defaults ----------
GPU="${GPU:-0}"
FOLD_NAME="${FOLD_NAME:-tbd}"
CFG="${CFG:-}"
EXTRA_ARGS=""

# ---------- parse named env overrides ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --fold_name) FOLD_NAME="$2"; shift 2 ;;
        --cfg) CFG="$2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ---------- auto-select config & data ----------
if [ "$FOLD_NAME" = "tbd" ]; then
    CFG="${CFG:-cfg/biflow_k20.yml}"
    DATA_SOURCE="original"
    DATA_DIR="./data/egotraj"
else
    CFG="${CFG:-cfg/biflow_t2fpv_k20.yml}"
    DATA_SOURCE="original_bal"
    DATA_DIR="./data/t2fpv"
fi

echo "=== BiFlow Training ==="
echo "  fold_name : $FOLD_NAME"
echo "  cfg       : $CFG"
echo "  data_dir  : $DATA_DIR"
echo "  gpu       : $GPU"
echo "========================"

python scripts/train_biflow.py \
    --cfg "$CFG" \
    --fold_name "$FOLD_NAME" \
    --data_source "$DATA_SOURCE" \
    --data_dir "$DATA_DIR" \
    --gpu "$GPU" \
    $EXTRA_ARGS
