import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from typing import List, Tuple
from find_shap import *
import subprocess
from pathlib import Path
import math
import os
import sys

def is_folder_empty(folder_path):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
    
    return not any(os.scandir(folder_path))

def get_image_quality(state, group_list, episode):
    N_t, C_t, D_t, _, t_l = state
    n_checkpoint = 9500
    round = int(5 - t_l + 1)
    current_step = int(5 - t_l)
    train_dir = '../../generative-models/group_' + '+'.join(f'{i}' for i in group_list)

    output_dir = f"../rl_tem/{episode}" # episode
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if OUTPUT_DIR is empty and current_step is 0, copy the checkpoint
    if current_step == 0: #and not os.listdir(output_dir)
        checkpoint_src = "../../generative-models/artbench_expressionism_200/checkpoint-9500"
        subprocess.run(f"cp -r {checkpoint_src} {output_dir}", shell=True)

    checkpoint_path = f'../rl_tem/{episode}/checkpoint-{current_step * 500 + n_checkpoint}' # episode
    print(f'episode:{episode}, round:{round}, current_step:{current_step}, checkpointpath:{checkpoint_path}, train_dir:{train_dir}')
    # num_train_epochs = [(9500 + round * 500) * 2] / (N_t * 100)
    # num_train_epochs = math.ceil(num_train_epochs)  # Round up to ensure sufficient training
    if not os.path.exists(checkpoint_path):
        if is_folder_empty(output_dir):
            os.rmdir(output_dir)
        return 0
    numerator = (9500 + round * 500) * 2
    denominator = N_t * 100
    num_train_epochs = numerator / denominator
    num_train_epochs = math.ceil(num_train_epochs)  # Ensure it's a scalar

    # os.environ["MODEL_NAME"] = "stabilityai/stable-diffusion-xl-base-1.0"
    # os.environ["VAE_NAME"] = "madebyollin/sdxl-vae-fp16-fix"
    os.environ["TRAIN_DIR"] = train_dir
    os.environ["CHECKPOINT_PATH"] = checkpoint_path
    os.environ["OUTPUT_DIR"] = output_dir

    command = [
        "accelerate", "launch", "--num_processes", "2", "train_text_to_image_lora_sdxl.py",
        f"--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
        f"--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix",
        f"--train_data_dir={train_dir}", "--caption_column=caption",
        "--resolution=256", "--random_flip",
        "--train_batch_size=1", f"--num_train_epochs={num_train_epochs}",
        "--checkpointing_steps=100", "--learning_rate=1e-04",
        "--lr_scheduler=constant", "--lr_warmup_steps=100",
        "--mixed_precision=fp16", "--seed=42",
        f"--resume_from_checkpoint={checkpoint_path}",
        f"--output_dir={output_dir}",
        "--validation_prompt='a close up of a drawing of a man with glasses and a mustache, an ink drawing inspired by Stanisaw Tondos, reddit, shin hanga, man with glasses, portrait of sigmund freud, stanisaw'"
    ]

    # # Run the training command and hide the output
    # with subprocess.Popen(' '.join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
    #     out, err = proc.communicate()

    # if proc.returncode != 0:
    #     raise RuntimeError(f"Training failed with error: {err.decode()}")
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

        err = proc.stderr.read()
        if proc.returncode != 0:
            raise RuntimeError(f"Training failed with error: {err}")

    # inference script
    inference_command = f"python3 inference_rl.py --model_path rl_tem/{episode}/checkpoint-{current_step * 500 + n_checkpoint}" # episode
    print('Inferencing!')
    # Run the inference command and collect the output
    with subprocess.Popen(inference_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        out, err = proc.communicate()
    
    fid = float(out.decode())
    
    # Print the episode and inference output
    print(f"episode: {episode}")  # episode
    print(f"FID Output:\n{fid}")

    return [1/(fid+1e-6)]*100

state = np.array([3, 2, 80, 66, 5], dtype=np.float32)
fid = get_image_quality(state, [1,2,3,4,5,6,7,8], 1)