# echo "e_seed: $1"
# echo "K: $2"
# echo "Z: $3"


export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
# export DATASET_NAME="../../generative-models/artbench_expressionism_200_imagefolder"
export DATASET_NAME="../../generative-models/cartoon_a"
export OUTPUT_DIR="output_contribution/checkpoint-9500"

python grad_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=$DATASET_NAME \
  --enable_xformers_memory_efficient_attention \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt="a cute Sundar Pichai creature" --validation_epochs 5 \
  --checkpointing_steps=5000 \
  --output_dir=$OUTPUT_DIR \
  --e_seed=42 \
  --K=10 \
  --Z=32768