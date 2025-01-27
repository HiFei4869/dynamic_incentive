## function 1 -- input: name of group (string), number of image in each group (list), name of base folder (string)
## function 2 -- input: name of folder (string), name of base folder (string); output: csv recording the copyright

# calculate_clip_mse_folder(sorted file name) -> list of clip score (sem score)
# (sorted file name) -> list of per score

import torch
import numpy as np
from PIL import Image
from clip_interrogator import Config, Interrogator
import os
import argparse
import csv
from tqdm import tqdm
from datasets import load_dataset
import re
import pandas as pd

import lpips
import torchvision.transforms as T


def extract_number(file_path: str) -> int:
    match = re.search(r'image_(\d+)\.jpg$', os.path.basename(file_path))
    if match:
        return int(match.group(1))
    return float('inf')



def calculate_sem_score_folder(local_image_folder, target_dir, need_sort):
    local_image_folder = f'../../generative-models/{local_image_folder}/images'
    files_local = [os.path.join(local_image_folder, f) for f in os.listdir(local_image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    files_target = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if need_sort:       
        files_local = sorted(files_local, key=extract_number)
        files_target = sorted(files_target, key=extract_number)
    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
    mse_loss = []
    for i, path in enumerate(files_local):
        img0 = Image.open(path).convert('RGB')
        img1 = Image.open(files_target[i]).convert('RGB')

        embedding_0 = ci.image_to_features(img0).cpu().numpy()
        embedding_1 = ci.image_to_features(img1).cpu().numpy()
        
        # Calculate MSE
        mse = np.mean((embedding_0 - embedding_1) ** 2)
        mse_loss.append(mse)
    
    return mse_loss

def calculate_per_score_folder(local_image_folder, target_dir, need_sort):
    loss_fn = lpips.LPIPS(net='alex',version='0.1')
    loss_fn.cuda()
    resize_transform = T.Resize((256, 256))
    local_image_folder = f'../../generative-models/{local_image_folder}/images'

    files_local = [os.path.join(local_image_folder, f) for f in os.listdir(local_image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    files_target = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if need_sort:       
        files_local = sorted(files_local, key=extract_number)
        files_target = sorted(files_target, key=extract_number)

    per_loss = []

    for i, path in enumerate(files_local):

        img0 = lpips.im2tensor(lpips.load_image(path))
        img1 = lpips.im2tensor(lpips.load_image(files_target[i]))

        # Resize images to the same size
        img0 = resize_transform(img0)
        img1 = resize_transform(img1)


        img0 = img0.cuda()
        img1 = img1.cuda()

        # Compute distance
        dist01 = loss_fn.forward(img0, img1)
        per_loss.append(dist01.detach().cpu().squeeze().numpy())

    return per_loss

def calculate_copyright_score_folder(local_image_folder, target_dir, need_sort):
    per_loss = calculate_per_score_folder(local_image_folder, target_dir, need_sort)
    sem_loss = calculate_sem_score_folder(local_image_folder, target_dir, need_sort)

    copyright_loss = []
    copyright_loss = sem_loss*500 + per_loss
    c = max(copyright_loss) - min(copyright_loss)
    copyright_loss_norm = 1 - (copyright_loss - min(copyright_loss))/c

    return copyright_loss_norm

def inference(target_dir):
    jsonl_file_path = f'../../generative-models/{target_dir}/metadata.jsonl'
    import json
    from diffusers import DiffusionPipeline
    import subprocess
    import argparse

    # example: model_path = "./rl_track/3_1_0_0_0/checkpoint-10500"
    model_path = "../../generative-models/artbench_expressionism_200/checkpoint-9500"
    model_identifier = f"{target_dir}_cp"

    output_dir = f"../../generative-models/rl_inference_output/{model_identifier}"
    os.makedirs(output_dir, exist_ok=True)

    dir = os.listdir(output_dir)
    if len(dir) == 0: # haven't make inference

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        pipe.to("cuda")
        pipe.load_lora_weights(model_path)


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
    return output_dir

def get_copyright(dataset):
    if dataset == "a":
        dataholders = ["a","b","c","d","e","f","9","10"] # ArtBench
        dataset_names = ["group_" + item for item in dataholders]
        data_root = "artbench"

    elif dataset == "p":
        dataholders = ["i","j","k","l","m","n","o","p"] # Portrait
        dataset_names = ["portrait_" + item for item in dataholders]
        data_root = "portrait"
    else:
        dataholders = ["a","b","c","d","e","f","g","h"] # Cartoon
        # dataholders = ["a","b"]
        dataset_names = ["cartoon_" + item for item in dataholders]
        data_root = "cartoon"

    copyright_group = []
    for local_image_folder in dataset_names:
        inference_output = inference(local_image_folder)
        need_sort = 1
        copyright_loss_norm = calculate_copyright_score_folder(local_image_folder, inference_output, need_sort)
        copyright_each = sum(copyright_loss_norm)
        copyright_group.append(copyright_each)
    copyright_group = [x/sum(copyright_group) for x in copyright_group]

    print(copyright_group)
    return copyright_group

if __name__ == "__main__":
    a = get_copyright("c")
# parser.add_argument('--dir0',type=str, default='../generative-models/group_1/images')
# parser.add_argument('--dir1',type=str, default='../generative-models/inference_output/fid_group_1_10500_256')
    # local_image_folder = '../generative-models/cartoon_a'


###################

