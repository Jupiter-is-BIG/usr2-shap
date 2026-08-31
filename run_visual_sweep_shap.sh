#!/usr/bin/env bash
# Visual-severity sweep for usr2, full LRS3 test set,
# Permutation SHAP.
#
# Runs all severity levels for ONE distortion type in parallel,
# waits for that type to finish, then moves to the next type.
#
# Mirrors Dr-SHAP-AV/run_gb_sweep_omniavsr.sh, adapted to usr2's Hydra CLI
# (eval_shap.py + conf/config_shap.yaml) instead of Dr-SHAP-AV's argparse.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

GPUS=(0 1 2 3 4)
TYPES=(JPEG BW GB GNC CC)

BACKBONE=resnet_transformer_large
CKPT=/aa4825/models/usr2/large_high_resource_lrs3vox2.pth
TEST_CSV=/aa4825/data/labelslrs3_test_with_tags_counts_unigram1000_micro50.csv
LRS3_VIDEO_DIR=/ucappell/datasets/lrs3/lrs3_video_seg24s
LRS3_AUDIO_DIR=/ucappell/datasets/lrs3/lrs3_video_seg24s
OUT_DIR=output
WANDB_PROJECT=dr-shap-av-visual-usr2

BEAM_SIZE=30
CTC_WEIGHT=0.1
MAXLENRATIO=0.4

LOGDIR=logs/visual_sweep_usr2
mkdir -p "$LOGDIR" "$OUT_DIR"

COMMON_ARGS=(
  project_name="$WANDB_PROJECT"
  log_wandb=True

  model/backbone="$BACKBONE"
  model.pretrained_model_path="$CKPT"

  data.dataset.test_csv="$TEST_CSV"
  data.lrs3_video_dir="$LRS3_VIDEO_DIR"
  data.lrs3_audio_dir="$LRS3_AUDIO_DIR"

  decode.beam_size=$BEAM_SIZE
  decode.ctc_weight=$CTC_WEIGHT
  decode.maxlenratio=$MAXLENRATIO

  shap.compute_shap=true
  shap.shap_alg=permutation
  shap.num_samples_shap=2000
  shap.output_path_shap="$OUT_DIR"
)

JOBS=(
  "lvl1:1"
  "lvl2:2"
  "lvl3:3"
  "lvl4:4"
  "lvl5:5"
)

MAX_PARALLEL=${#GPUS[@]}

for type in "${TYPES[@]}"; do

  echo ""
  echo "============================================================"
  echo "Starting distortion type: $type"
  echo "============================================================"
  echo ""

  gpu_idx=0

  for job in "${JOBS[@]}"; do
    suffix="${job%%:*}"
    level="${job#*:}"

    gpu="${GPUS[$((gpu_idx % MAX_PARALLEL))]}"
    gpu_idx=$((gpu_idx + 1))

    exp_name="usr2_shap_permutation_viddist-${type}-${suffix}"

    echo "Launching $exp_name on GPU $gpu"
    echo "  Log: $LOGDIR/${exp_name}.log"

    CUDA_VISIBLE_DEVICES=$gpu python eval_shap.py \
      "${COMMON_ARGS[@]}" \
      data.vid_dist_type="$type" \
      data.vid_dist_level=$level \
      project_name="$exp_name" \
      > "$LOGDIR/${exp_name}.log" 2>&1 &

  done

  echo ""
  echo "Waiting for all $type runs to finish..."
  wait

  echo ""
  echo "============================================================"
  echo "Finished distortion type: $type"
  echo "============================================================"
  echo ""

done

echo "============================================================"
echo "ALL visual severity sweeps finished."
echo "Logs in $LOGDIR/"
echo "Shapley .npz files in $OUT_DIR/"
echo "============================================================"
