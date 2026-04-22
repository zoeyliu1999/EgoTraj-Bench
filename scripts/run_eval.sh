#!/bin/bash
# -------------------------------------------------------
# Evaluate BiFlow on EgoTraj-Bench
# Usage (two checkpoint sources):
#   A) Training outputs under results/ (local experiments)
#      - Use --run_name (auto-resolve to results/<cfg_name>/<run_name>)
#      - Example:
#        bash scripts/run_eval.sh --fold_name zara2 --run_name train_v1_zara2_orig_bal_FM_SharedFuser_K20_EP150_BS32_LR0.0001
#
#   B) Release package under checkpoints/ (public/shared models)
#      - Use --release_name (auto-resolve to checkpoints/<release_name>)
#      - Example:
#        bash scripts/run_eval.sh --fold_name zara2 --release_name T2FPV-zara2
#
#   C) Any explicit path (highest priority)
#      - Use --ckpt_dir directly
#      - Example:
#        bash scripts/run_eval.sh --ckpt_dir results/biflow_t2fpv_k20/<run_name> --fold_name eth
#
#   D) TBD K=5 example (results source)
#      bash scripts/run_eval.sh --fold_name tbd --cfg_name biflow_k5 --run_name train_v1_tbd_orig_FM_SharedFuser_K5_EP150_BS32_LR0.0001
#
#   # mode behavior
#   #   (default)       -> eval_biflow.py --mode best (load checkpoint_best.pt)
#   #   --mode best     -> load checkpoint_best.pt
#   #   --mode last     -> load checkpoint_last.pt
#   #   --mode <int>    -> load checkpoint_best_<int>.pt
# -------------------------------------------------------

set -e

# ---------- defaults ----------
GPU="${GPU:-0}"
FOLD_NAME="${FOLD_NAME:-tbd}"
CKPT_DIR=""
CFG_NAME=""
RUN_NAME=""
RESULTS_ROOT="${RESULTS_ROOT:-results}"
RELEASE_NAME=""
RELEASE_ROOT="${RELEASE_ROOT:-checkpoints}"
MODE="${MODE:-best}"
EXTRA_ARGS=""

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --fold_name) FOLD_NAME="$2"; shift 2 ;;
        --ckpt_dir) CKPT_DIR="$2"; shift 2 ;;
        --cfg_name) CFG_NAME="$2"; shift 2 ;;
        --run_name) RUN_NAME="$2"; shift 2 ;;
        --results_root) RESULTS_ROOT="$2"; shift 2 ;;
        --release_name) RELEASE_NAME="$2"; shift 2 ;;
        --release_root) RELEASE_ROOT="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ---------- infer cfg_name ----------
if [ -z "$CFG_NAME" ]; then
    if [ "$FOLD_NAME" = "tbd" ]; then
        # Auto-route TBD K=5 runs if run_name carries K5.
        if [ -n "$RUN_NAME" ] && [[ "$RUN_NAME" == *"_K5_"* ]]; then
            CFG_NAME="biflow_k5"
        else
            CFG_NAME="biflow_k20"
        fi
    else
        CFG_NAME="biflow_t2fpv_k20"
    fi
fi

# ---------- resolve ckpt_dir ----------
# priority: explicit --ckpt_dir > --release_name > --run_name
if [ -z "$CKPT_DIR" ] && [ -n "$RELEASE_NAME" ]; then
    CKPT_DIR="${RELEASE_ROOT}/${RELEASE_NAME}"
fi

if [ -z "$CKPT_DIR" ] && [ -n "$RUN_NAME" ]; then
    CKPT_DIR="${RESULTS_ROOT}/${CFG_NAME}/${RUN_NAME}"
fi

if [ -z "$CKPT_DIR" ]; then
    echo "Error: Must provide --fold_name, and one of --ckpt_dir / --release_name / --run_name."
    echo "Examples:"
    echo "  bash scripts/run_eval.sh --ckpt_dir results/${CFG_NAME}/<run_name> --fold_name ${FOLD_NAME}"
    echo "  bash scripts/run_eval.sh --fold_name ${FOLD_NAME} --release_name <release_name>"
    echo "  bash scripts/run_eval.sh --fold_name ${FOLD_NAME} --run_name <run_name>"
    exit 1
fi

# ---------- auto-select data ----------
if [ "$FOLD_NAME" = "tbd" ]; then
    DATA_SOURCE="original"
    DATA_DIR="./data/egotraj"
else
    DATA_SOURCE="original_bal"
    DATA_DIR="./data/t2fpv"
fi

echo "=== BiFlow Evaluation ==="
echo "  fold_name : $FOLD_NAME"
echo "  cfg_name  : $CFG_NAME"
echo "  release   : ${RELEASE_NAME:-<none>}"
echo "  run_name  : ${RUN_NAME:-<auto>}"
echo "  ckpt_dir  : $CKPT_DIR"
echo "  mode(raw) : $MODE"
echo "  mode(eff) : $MODE"
echo "  data_dir  : $DATA_DIR"
echo "  gpu       : $GPU"
echo "=========================="

python scripts/eval_biflow.py \
    --ckpt_dir "$CKPT_DIR" \
    --fold_name "$FOLD_NAME" \
    --data_source "$DATA_SOURCE" \
    --data_dir "$DATA_DIR" \
    --gpu "$GPU" \
    --mode "$MODE" \
    $EXTRA_ARGS
