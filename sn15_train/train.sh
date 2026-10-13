#!/usr/bin/env bash
# Style LoRA training for SenseNova-U1.5-8B-MoT on 1x A40.
# Usage: train.sh <A|B> [extra args, which override the defaults below]
#   A: base MoT with the 8-step V2 distill LoRA fused in (--preset_lora_path), style LoRA trained on top
#   B: plain base MoT; the 8-step LoRA is only stacked at inference (and while sampling)
# 560 images x 18 epochs = 10,080 steps; a checkpoint + 8-step samples every 500 steps
# (train_sn15.py = official train.py + sampling; see <output>/samples/step-N.jpg).
# Env overrides: METADATA (csv path), OUTPUT (checkpoint dir)
set -euo pipefail
VARIANT=${1:?usage: train.sh <A|B> [extra args]}
shift

ROOT=/root/source/test_SenseNova-U1.5
DS=$ROOT/DiffSynth-Studio
export DIFFSYNTH_MODEL_BASE_PATH=$DS/models
export DIFFSYNTH_SKIP_DOWNLOAD=true
LORA8=$DIFFSYNTH_MODEL_BASE_PATH/SenseNova/SenseNova-U1.5-8B-MoT-LoRAs/SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors

case $VARIANT in
  A) PRESET=(--preset_lora_path "$LORA8" --preset_lora_model dit) ;;
  B) PRESET=(--sample_lora8 "$LORA8") ;;
  *) echo "variant must be A or B" >&2; exit 1 ;;
esac

cd "$DS"
/root/venvs/sn15/bin/accelerate launch --num_processes 1 --num_machines 1 --mixed_precision no --dynamo_backend no \
  "$ROOT/sn15_train/train_sn15.py" \
  --dataset_base_path "$ROOT/sn15_train/data" \
  --dataset_metadata_path "${METADATA:-$ROOT/sn15_train/meta/train.csv}" \
  --data_file_keys image \
  --extra_inputs max_timestep_boundary \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --dataset_num_workers 2 \
  --model_id_with_origin_paths "SenseNova/SenseNova-U1.5-8B-MoT:model*.safetensors" \
  --tokenizer_config "SenseNova/SenseNova-U1.5-8B-MoT:./" \
  --lora_base_model dit \
  --lora_target_modules "q_proj_mot_gen,k_proj_mot_gen,v_proj_mot_gen,o_proj_mot_gen,mlp_mot_gen.gate_proj,mlp_mot_gen.up_proj,mlp_mot_gen.down_proj" \
  --lora_rank 32 \
  --learning_rate 1e-4 \
  --num_epochs 18 \
  --save_steps 500 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT:-$DS/models/train/sn15_style_$VARIANT}" \
  --use_gradient_checkpointing \
  --enable_csv_log \
  --enable_tensorboard_log \
  "${PRESET[@]}" "$@"
