# import torch
# import numpy as np
# from PIL import Image
# from clip_interrogator import Config, Interrogator
# import os
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# import argparse
# import warnings
# warnings.filterwarnings("ignore")

# device = "cuda"
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('-p0','--path0', type=str, default='./naruto.png')
# parser.add_argument('-p1','--path1', type=str, default='./naruto_new.png')

# opt = parser.parse_args()

# # set the seed to ensure reproduction
# torch.manual_seed(43)

# ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))


# image_0 = Image.open(opt.path0).convert('RGB')
# image_1 = Image.open(opt.path1).convert('RGB')

# embedding_0 = ci.image_to_features(image_0).cpu().numpy()
# embedding_1 = ci.image_to_features(image_1).cpu().numpy()


# # Calculate MSE
# mse = np.mean((embedding_0 - embedding_1) ** 2)

# print("Sem_Loss:", mse)

############## Multiple images from huggingface ################
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

# def resize_image(image, size):
#     return image.resize((size, size))

# def calculate_clip_mse(hf_repo_name, local_image_folder, num_images, output_csv):
#     # Load the dataset from Hugging Face
#     dataset = load_dataset(hf_repo_name)
    
#     # Initialize CLIP Interrogator
#     ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
    
#     mse_losses = []

#     # Prepare CSV file
#     with open(output_csv, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Image Index", "Semantic Loss"])
        
#         for i in tqdm(range(num_images)):
#             # Load image from Hugging Face dataset
#             hf_image = dataset['train'][i]['image']
            
#             # Resize the Hugging Face image to 256x256
#             hf_image_resized = resize_image(hf_image, 256)
            
#             # Load corresponding local image
#             local_image_path = os.path.join(local_image_folder, f'generated_image_{i+1}.png')
#             local_image = Image.open(local_image_path).convert('RGB')
            
#             # Resize the local image to 256x256
#             local_image_resized = resize_image(local_image, 256)
            
#             # Compute CLIP embeddings
#             embedding_hf = ci.image_to_features(hf_image_resized).cpu().numpy()
#             embedding_local = ci.image_to_features(local_image_resized).cpu().numpy()
            
#             # Calculate MSE
#             mse = np.mean((embedding_hf - embedding_local) ** 2)
#             mse_losses.append(mse)
            
#             print(f'Image {i+1}/{num_images}: MSE Loss = {mse}')
            
#             # Write to CSV
#             writer.writerow([i+1, mse])

#     return mse_losses

def sort_files_by_number(directory):

    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def extract_number(file_path: str) -> int:
        match = re.search(r'image_(\d+)\.jpg$', os.path.basename(file_path))
        if match:
            return int(match.group(1))
        return float('inf')

    sorted_files = sorted(files, key=extract_number)
    return sorted_files


def calculate_clip_mse_folder(sorted_files0, sorted_files1):
    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
    mse_losses = []
    for i, path in enumerate(sorted_files0):
        print(path)
        img0 = Image.open(path).convert('RGB')
        img1 = Image.open(sorted_files1[i]).convert('RGB')

        embedding_0 = ci.image_to_features(img0).cpu().numpy()
        embedding_1 = ci.image_to_features(img1).cpu().numpy()
        
        # Calculate MSE
        mse = np.mean((embedding_0 - embedding_1) ** 2)
        mse_losses.append(mse)
    
    return mse_losses



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CLIP embedding MSE between images.")
    parser.add_argument('--hf_repo_name', type=str, default='HiFei4869/artbench_expressionism_200', help='Hugging Face repository name.')
    parser.add_argument('--local_image_folder', type=str, default='../generative-models/group_1/images', help='Path to local folder containing generated images.')
    parser.add_argument('--dir1', type=str, default='../generative-models/inference_output/fid_group_1_10500_256', help='Path to the inference output directory')
    parser.add_argument('--num_images', type=int, default=200, help='Number of images to process.')
    parser.add_argument('--output_csv', type=str, default='./sem_loss_1_10500.csv', help='Path to output CSV file.')
    args = parser.parse_args()

    # Set the seed to ensure reproduction
    torch.manual_seed(43)
    
    # Calculate CLIP MSE losses
    # losses = calculate_clip_mse(args.hf_repo_name, args.local_image_folder, args.num_images, args.output_csv)

    sorted_files0 = sort_files_by_number(args.local_image_folder)
    sorted_files1 = sort_files_by_number(args.dir1)
    
    losses = calculate_clip_mse_folder(sorted_files0, sorted_files1)
    df = pd.DataFrame(losses, columns=['sem_loss'])
    df.to_csv(args.output_csv, index=True)
    # Print summary
    print(f'\nAverage Semantic Loss: {sum(losses) / len(losses):.5f}')
