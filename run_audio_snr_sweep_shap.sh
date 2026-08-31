#!/usr/bin/env bash
# Audio-SNR sweep for usr2, full LRS3 test set, Permutation SHAP.
#
# One job per SNR level, all launched in parallel across GPUs 0-7 (8 SNR
# levels, 8 GPUs -> a single wave, unlike the visual sweep's nested
# type/level loop).
#
# Mirrors run_visual_sweep_shap.sh, but sweeps decode.snr_target (+
# data.noise_path, the babble noise file) instead of data.vid_dist_type/level.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

GPUS=(0 1 2 3 4 5 6 7)

BACKBONE=resnet_transformer_large
CKPT=/aa4825/models/usr2/large_high_resource_lrs3vox2.pth
TEST_CSV=/aa4825/data/labels/lrs3_test_with_tags_counts_unigram1000_micro50.csv
LRS3_VIDEO_DIR=/ucappell/datasets/lrs3/lrs3_video_seg24s
LRS3_AUDIO_DIR=/ucappell/datasets/lrs3/lrs3_video_seg24s
NOISE_PATH=/path/to/babble_noise.npy
OUT_DIR=output
WANDB_PROJECT=dr-shap-av-audio-usr2

BEAM_SIZE=30
CTC_WEIGHT=0.1
MAXLENRATIO=0.4

LOGDIR=logs/audio_snr_sweep_usr2
mkdir -p "$LOGDIR" "$OUT_DIR"

COMMON_ARGS=(
  project_name="$WANDB_PROJECT"
  log_wandb=True

  model/backbone="$BACKBONE"
  model.pretrained_model_path="$CKPT"

  data.dataset.test_csv="$TEST_CSV"
  data.lrs3_video_dir="$LRS3_VIDEO_DIR"
  data.lrs3_audio_dir="$LRS3_AUDIO_DIR"
  data.noise_path="$NOISE_PATH"

  decode.beam_size=$BEAM_SIZE
  decode.ctc_weight=$CTC_WEIGHT
  decode.maxlenratio=$MAXLENRATIO

  shap.compute_shap=true
  shap.shap_alg=permutation
  shap.num_samples_shap=2000
  shap.output_path_shap="$OUT_DIR"
)

# label:snr_target -- 9999 is the codebase's sentinel for "no noise" (see
# data/transforms.py::AddNoise, which short-circuits to clean passthrough
# only on an exact match to 9999).
JOBS=(
  "clean:9999"
  "10:10"
  "5:5"
  "2p5:2.5"
  "0:0.0"
  "m2p5:-2.5"
  "m5:-5"
  "m10:-10"
)

MAX_PARALLEL=${#GPUS[@]}

echo ""
echo "============================================================"
echo "Starting audio SNR sweep (${#JOBS[@]} levels, ${MAX_PARALLEL} GPUs)"
echo "============================================================"
echo ""

gpu_idx=0

for job in "${JOBS[@]}"; do
  suffix="${job%%:*}"
  snr="${job#*:}"

  gpu="${GPUS[$((gpu_idx % MAX_PARALLEL))]}"
  gpu_idx=$((gpu_idx + 1))

  exp_name="usr2_shap_permutation_snr-${suffix}"

  echo "Launching $exp_name on GPU $gpu"
  echo "  Log: $LOGDIR/${exp_name}.log"

  CUDA_VISIBLE_DEVICES=$gpu python eval_shap.py \
    "${COMMON_ARGS[@]}" \
    decode.snr_target=$snr \
    project_name="$exp_name" \
    > "$LOGDIR/${exp_name}.log" 2>&1 &

done

echo ""
echo "Waiting for all SNR runs to finish..."
wait

echo ""
echo "============================================================"
echo "ALL audio SNR sweeps finished."
echo "Logs in $LOGDIR/"
echo "Shapley .npz files in $OUT_DIR/"
echo "============================================================"
