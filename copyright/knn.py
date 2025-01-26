import torch
import numpy as np
from PIL import Image
from clip_interrogator import Config, Interrogator
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import argparse
import warnings
import json
import pandas as pd
warnings.filterwarnings("ignore")

import re
import argparse


parser = argparse.ArgumentParser(description="output shapley values or copyright value by means of near neighbors")
parser.add_argument('--target_directory_path', type=str, help='Path to target directory')
parser.add_argument('--top_n_json', type=str, help='Path to store the nearest neighbor results')
parser.add_argument('--input_csv_path', type=str, help='Path to the input csv (shapley/copyright loss)')
parser.add_argument('--output_csv_path', type=str, help='Path to the output csv (shapley/copyright loss)')
args = parser.parse_args()

'''
python3 sem_test.py --target_directory_path ../generative-models/group_2/images --top_n_json sim_group_1_2.json --input_csv_path shap_group1_12000_s.csv --output_csv_path shap_group2_12000_s.csv
'''
target_directory_path = args.target_directory_path       #'../generative-models/group_2/images'
base_directory_path = '../generative-models/group_1/images'

top_n_similar_images_file_path = args.top_n_json         #'sim_group_1_2.json'

input_csv_path = args.input_csv_path                     #'shap_group1_12000_s.csv'
output_csv_path = args.output_csv_path                   #'shap_group2_12000_s.csv'

def get_numerical_value(filename):
    # map from 1-100 to 0-99
    return int(filename.split('_')[1].split('.')[0]) - 1

def eval_sim(directory_path, image_path):
    # set the seed to ensure reproduction
    torch.manual_seed(43)

    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

    mse_list = []

    entries = os.listdir(directory_path)
    sorted_entries = sorted(entries, key=get_numerical_value)

    for entry in sorted_entries:
        print(f'entry: {entry}')
        entry_path = os.path.join(directory_path, entry)

        image_0 = Image.open(image_path).convert('RGB')
        image_1 = Image.open(entry_path).convert('RGB')

        embedding_0 = ci.image_to_features(image_0).cpu().numpy()
        embedding_1 = ci.image_to_features(image_1).cpu().numpy()

        # Calculate MSE
        mse = 1 / (np.mean((embedding_0 - embedding_1) ** 2) + 1e-4)
        mse_list.append(mse)

    return mse_list

def compute_top_n_similar_images(target_directory_path, base_directory_path, n):
    """Compute MSE for all images in target folder w.r.t images in base folder."""
    result = {}

    target_images = os.listdir(target_directory_path)
    target_images_sorted = sorted(target_images, key=get_numerical_value)

    for target_image in target_images_sorted:
        target_image_path = os.path.join(target_directory_path, target_image)
        
        # Get MSE list for this target image against all base images
        mse_list = eval_sim(base_directory_path, target_image_path)
        
        # Get indices of the n highest values in mse_list
        top_n_indices = np.argsort(mse_list)[-n:][::-1]
        
        # Record the result for this target image
        result[target_image] = top_n_indices.tolist()

    return result

def save_top_n_similar_images(top_n_similar_images, file_path):
    """Save the top_n_similar_images dictionary to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(top_n_similar_images, f)

def load_top_n_similar_images(file_path):
    """Load the top_n_similar_images dictionary from a JSON file."""
    with open(file_path, 'r') as f:
        top_n_similar_images = json.load(f)
    return top_n_similar_images


# top_n_similar_images = compute_top_n_similar_images(target_directory_path, base_directory_path, 5)

# save_top_n_similar_images(top_n_similar_images, top_n_similar_images_file_path)

def compute_average_scores(input_csv_path, top_n_similar_images, output_csv_path):
    """Compute the average scores based on the top_n_similar_images."""
    scores_df = pd.read_csv(input_csv_path)
    
    # Ensure the column is named 'normalized'
    if 'normalized' not in scores_df.columns:
        raise ValueError("Input CSV must have a column named 'normalized'.")
    
    average_scores = []

    for target_image, indices in top_n_similar_images.items():
        # Retrieve the scores of the top_n similar images
        similar_scores = scores_df['normalized'].iloc[indices].values

        average_score = np.mean(similar_scores)

        average_scores.append(average_score)

    output_df = pd.DataFrame({'normalized': average_scores})

    output_df.to_csv(output_csv_path, index=False)



top_n_similar_images = load_top_n_similar_images(top_n_similar_images_file_path)

compute_average_scores(input_csv_path, top_n_similar_images, output_csv_path)

# Example usage:
# target_directory_path = '../generative-models/group_1/images'
# base_directory_path = '../generative-models/shap_bg'
# n = 5

# top_n_similar_images = compute_top_n_similar_images(target_directory_path, base_directory_path, n)
# print(top_n_similar_images)

############## Multiple images ################
# import torch
# import numpy as np
# from PIL import Image
# from clip_interrogator import Config, Interrogator
# import os
# import argparse
# import csv
# from tqdm import tqdm
# from datasets import load_dataset

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

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Calculate CLIP embedding MSE between images.")
#     parser.add_argument('--hf_repo_name', type=str, default='HiFei4869/artbench_expressionism_200', help='Hugging Face repository name.')
#     parser.add_argument('--local_image_folder', type=str, default='../generative-models/inference_output/artbench_expressionism_200_2', help='Path to local folder containing generated images.')
#     parser.add_argument('--num_images', type=int, default=200, help='Number of images to process.')
#     parser.add_argument('--output_csv', type=str, default='./semantic_loss_2.csv', help='Path to output CSV file.')
#     args = parser.parse_args()

#     # Set the seed to ensure reproduction
#     torch.manual_seed(43)
    
#     # Calculate CLIP MSE losses
#     losses = calculate_clip_mse(args.hf_repo_name, args.local_image_folder, args.num_images, args.output_csv)
    
#     # Print summary
#     print(f'\nAverage Semantic Loss: {sum(losses) / len(losses):.3f}')
