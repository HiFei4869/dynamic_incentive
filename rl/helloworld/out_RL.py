import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from typing import List, Tuple
import subprocess
import os
import shutil
import re
import csv
import torch
from datetime import datetime

def main(budget, mode):
    if mode == 0:
        budget_distribution = budget * np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    elif mode == 1:
        budget_distribution = budget * np.random.dirichlet(np.ones(5),size=1)[0]
    else: # greedy
        budget_distribution = budget * np.array([0.5, 0.25, 0.25, 0, 0])
    
    for round in range(5):
        # inner_train(1, budget_distribution[round])
        inner_train(2, budget_distribution[round])
    round_dir = f"../../generative-models/rl_logging/Artbench-G-L/2"
    if not os.path.exists(round_dir):
        os.makedirs(round_dir)
    fid_value = fid(round_dir)
    print(fid_value)

    current_date = datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
    output_dir = "./output_fid"
    output_file = os.path.join(output_dir, f"Artbench-G-L.txt")

    with open(output_file, "a") as file:
        file.write(f"output: {fid_value}\n")



def inner_train(round_num, budget):
    dataholders = ["a","b","c","d","e","f","9","10"] # ArtBench
    # dataholders = ["i","j","k","l","m","n","o","p"] # Portrait
    # dataholders = ["a","b","c","d","e","f","g","h"] # Cartoon
    mode = 2       # 0: inner RL; 1: inner random; 2: inner linear
    if mode == 0:  # contribution only
        budget_distribution = budget * np.array([0.41929144, 0.12463295, 0.07377037, 0.07244312, 0.06626723, 0.06589315, 0.08088747, 0.0968142])
        # budget_distribution = budget * np.array([1, 0, 0, 0, 0, 0, 0, 0])
        # budget_distribution = budget * np.array([0.12648772, 0.12831376, 0.11076288, 0.12974688, 0.12136942, 0.11740535, 0.1306551, 0.13525891])
        # [0.11461563, 0.13077076, 0.11106646, 0.12858988, 0.13041177, 0.12012152, 0.13342872, 0.13099532]
        # budget_distribution = budget * np.array([0.11741429, 0.13432065, 0.11099451, 0.12883137, 0.12621321, 0.11972708, 0.13195506, 0.13054389]) # portrait
        # budget_distribution = budget * np.array([0.12127227,0.13276975, 0.11028866, 0.1277392,  0.12527613, 0.11913971, 0.13001741, 0.13349685]) # p contribution
        # budget_distribution = budget * np.array([0.67003, 0.11964162, 0.034562, 0.03226982, 0.02734387, 0.02987239, 0.03762612, 0.04865424]) # cartoon
        # budget_distribution = budget * np.array([0.6494294, 0.08202955, 0.04455208, 0.04405867, 0.03689169, 0.03740072, 0.04700756, 0.05863034]) # p contribution
    elif mode == 1:
        budget_distribution = budget * np.random.dirichlet(np.ones(8),size=1)[0]
    else:
        budget_distribution = budget * np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
    price = np.array([65, 35, 60, 35, 55, 30, 50, 100]) # ArtBench
    data_length = [100, 50, 100, 60, 100, 50, 100, 200] # ArtBench
    # price = np.array([50, 90, 60, 110, 50, 85, 90, 170]) # Portrait
    # data_length = [50, 100, 50, 100, 60, 90, 100, 200] # Portrait
    # price = np.array([55, 75, 45, 65, 30, 40, 95, 160]) # Cartoon
    # data_length = [50, 80, 40, 60, 25, 30, 100, 150] # Cartoon
    determine = np.where(budget_distribution - price >= 0)[0]
    train_dir = []
    num_data = 0
    for i in determine:
        dir = "group_" + dataholders[i]
        # dir = "portrait_" + dataholders[i]
        # dir = "cartoon_" + dataholders[i]
        train_dir.append(dir) # the training directory of the current step
        num_data += data_length[i]
    if num_data > 0:
        launch_training(round_num, train_dir, num_data)
        return 1
    else:
        return 0
        
def launch_training(round_num, train_dir, num_data):
    round_dir = f"../../generative-models/rl_logging/Artbench-G-L/{round_num}"
    if not os.path.exists(round_dir):
        os.makedirs(round_dir)
        source_dir = "../../generative-models/artbench_expressionism_200/checkpoint-9500"
        target_dir = os.path.join(round_dir, "checkpoint-9500")
        shutil.copytree(source_dir, target_dir)
        checkpoint_path = f"{round_dir}/checkpoint-9500"
        last_checkpoint_number = 9500
    else:
        checkpoint_path, last_checkpoint_number = last_checkpoint(round_dir)
    
    print(f'last_checkpoint:{last_checkpoint_number}')
    print(f'num_data:{num_data}')
    print(f'start training round {round_num}')
    print(f'checkpoint_path:{checkpoint_path}')
    print(f'round_dir:{round_dir}')

    num_gpus = torch.cuda.device_count()

    # train_dir = f"../../generative-models/{train_dir}"
    train_dir = ' '.join([f"../../generative-models/{group}" for group in train_dir])
    print(f'train_dir:{train_dir}')

    os.environ["MODEL_NAME"] = "stabilityai/stable-diffusion-xl-base-1.0"
    os.environ["VAE_NAME"] = "madebyollin/sdxl-vae-fp16-fix"
    
    os.environ["TRAIN_DIR"] = train_dir
    os.environ["CHECKPOINT_PATH"] = checkpoint_path
    os.environ["OUTPUT_DIR"] = round_dir
    os.environ["TRAIN_EPOCH"] = str(int((last_checkpoint_number + 500) * num_gpus / num_data + 20))
    # os.environ["TRAIN_EPOCH"] = "100"
    os.environ["NUM_PROCESS"] = str(num_gpus)
    print(os.environ["TRAIN_EPOCH"])

    # num_process = 2/4/6/8
    command = [
        "accelerate", "launch", "--num_processes", os.environ["NUM_PROCESS"], "../../generative-models/train_text_to_image_lora_sdxl.py",
        "--pretrained_model_name_or_path", os.environ["MODEL_NAME"],
        "--pretrained_vae_model_name_or_path", os.environ["VAE_NAME"],
        "--train_data_dir", os.environ["TRAIN_DIR"],
        "--caption_column", "caption",
        "--resolution", "256",
        "--random_flip",
        "--train_batch_size", "1",
        "--num_train_epochs", os.environ["TRAIN_EPOCH"],
        "--checkpointing_steps", "100",
        "--learning_rate", "1e-04",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "100",
        "--mixed_precision", "fp16",
        "--seed", "42",
        "--resume_from_checkpoint", os.environ["CHECKPOINT_PATH"],
        "--output_dir", os.environ["OUTPUT_DIR"],
        "--validation_prompt", "painting of a house in a wooded area with trees and a mountain in the background, an art deco painting inspired by Alesso Baldovinetti, instagram, modernism, oil on canvas (1921), italian futurism style, cypresses and hills"
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

def last_checkpoint(round_dir):
    valid_numbers = {9500, 10000, 10500, 11000, 11500, 12000}
    
    # Compile a regex pattern
    pattern = re.compile(r"^checkpoint-(\d+)$")
    
    largest_checkpoint = None
    largest_number = -1
    
    for subfolder in os.listdir(round_dir):
        subfolder_path = os.path.join(round_dir, subfolder)
        if os.path.isdir(subfolder_path):
            match = pattern.match(subfolder)
            if match:
                checkpoint_number = int(match.group(1))
                if checkpoint_number in valid_numbers and checkpoint_number > largest_number:
                    largest_number = checkpoint_number
                    largest_checkpoint = subfolder_path
    if not largest_checkpoint:
        source_dir = "../../generative-models/artbench_expressionism_200/checkpoint-9500"
        target_dir = os.path.join(round_dir, "checkpoint-9500")
        shutil.copytree(source_dir, target_dir)
        largest_checkpoint = f"{round_dir}/checkpoint-9500"
        largest_number = 9500       
    return largest_checkpoint, largest_number

def fid(round_dir):
    # input round_dir, output the FID value (float)
    ###### inference ######
    model_path, _ = last_checkpoint(round_dir)

    jsonl_file_path = '../../generative-models/shap_bg_imagefolder/metadata.jsonl'
    import json
    from diffusers import DiffusionPipeline
    import subprocess
    import argparse

    # example: model_path = "./rl_track/3_1_0_0_0/checkpoint-10500"

    directory_path = os.path.dirname(model_path)
    model_identifier = os.path.basename(directory_path)

    output_dir = f"../../generative-models/rl_inference_output/{model_identifier}"
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.load_lora_weights(model_path)

    # output_dir = "./inference_output/3_1_0_0_0"
    # os.makedirs(output_dir, exist_ok=True)

    # Open the .jsonl file and read it line by line
    with open(jsonl_file_path, 'r') as file:
        i = 0
        for line in file:
            # Parse the JSON line into a dictionary
            data = json.loads(line)
            
            # Access the desired column (e.g., 'age')
            prompt = data.get('caption')  # Use .get() to avoid KeyError if 'age' is missing
            
            #prompt = "A naruto with green eyes and red legs."
            image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
            output_path = os.path.join(output_dir, f"generated_image_{i+1}.png")
            image.save(output_path)
            i += 1

    ###### fid ######
    command = [
    'python', '-m', 'pytorch_fid',
    '../../generative-models/shap_bg_imagefolder/images', output_dir,
    '--dims', '64',
    '--device', 'cuda:0'
    ]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the standard output and standard error
    print("FID:\n", result.stdout)

    try:
        fid_value = float(result.stdout.strip().split()[-1])
    except (IndexError, ValueError):
        print("Error: Could not extract FID value from output.")

    return fid_value
    
if __name__ == "__main__":
    # mode = 0: linear; mode = 1: random; mode = 2: greedy
    budget = 1000
    mode = 2
    main(budget, mode)



