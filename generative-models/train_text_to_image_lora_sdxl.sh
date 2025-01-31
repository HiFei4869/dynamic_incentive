#!/bin/bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
# export DATASET_NAME="HiFei4869/artbench_expressionism_200"
export TRAIN_DIR="cartoon_a"
export CHECKPOINT_PATH="./cartoon_LRL/checkpoint-9500"
export OUTPUT_DIR="./cartoon_LRL"
accelerate launch --num_processes 4 train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir="$TRAIN_DIR" --caption_column="caption" \
  --resolution=256 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=960 --checkpointing_steps=100 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=100 \
  --mixed_precision="fp16" \
  --seed=42 \
  --resume_from_checkpoint=$CHECKPOINT_PATH \
  --output_dir=$OUTPUT_DIR \
  --validation_prompt="Taylor Swift, detailed eyes and skin, araffed taylor swift performs at a concert with a microphone, golden inlays, taylor swift modeling, jeweled, 2019 trending photo, tn, toronto, sri lanka, posed, lalisa manobal, fashion icon, on, sparkling blue eyes, golden dress, smug look" \

## 空格隔开两个string
