#!/bin/bash
export CHECKPOINT_PATH="../generative-models/artbench_expressionism_200/checkpoint-9500"
export OUTPUT_DIR="../generative-models/artbench_expressionism_200"
export DATASET_DIR="../generative-models/artbench_expressionism_200_imagefolder"
export NUM_TRAIN_EPOCHS=96 #96
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"

# Execute the training script
# --enable_xformers_memory_efficient_attention \
accelerate launch grad_sim1.py \
    --enable_xformers_memory_efficient_attention \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path=$VAE_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_data_dir=$DATASET_DIR \
    --caption_column="caption" \
    --image_column="image" \
    --resolution=256 --random_flip \
    --resume_from_checkpoint=$CHECKPOINT_PATH \
    --checkpointing_steps=2 \
    --train_batch_size=1 \
    --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --num_train_epochs=$NUM_TRAIN_EPOCHS \
    --mixed_precision="fp16" \
    --seed=42

# python3 grad_sim1.py \
#     --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
#     --pretrained_vae_model_name_or_path=$VAE_NAME \
#     --output_dir=$OUTPUT_DIR \
#     --train_data_dir=$DATASET_NAME \
#     --caption_column="text" \
#     --resolution=256 --random_flip \
#     --resume_from_checkpoint=$CHECKPOINT_PATH \
#     --checkpointing_steps=2 \
#     --train_batch_size=1 \
#     --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
#     --num_train_epochs=$NUM_TRAIN_EPOCHS \
#     --mixed_precision="fp16" \
#     --seed=42