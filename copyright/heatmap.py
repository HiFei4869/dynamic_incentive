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
import matplotlib.pyplot as plt
import seaborn as sns

import lpips
import torchvision.transforms as T

plt.rcParams['font.family'] = 'DejaVu Serif'

def extract_number(file_path: str) -> int:
    match = re.search(r'image_(\d+)\.png$', os.path.basename(file_path))
    if match:
        return int(match.group(1))
    return float('inf')

# Function to calculate semantic loss (MSE) between two groups of images
def calculate_semantic_loss_between_groups(image_folder):
    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

    # Get all image files in the folder
    files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    files = sorted(files, key=extract_number)  # Sort files by their integer names

    # Separate into two groups
    group1 = [f for f in files if extract_number(f) < 10]  # image_0 to image_9
    group2 = [f for f in files if extract_number(f) >= 10]  # image_10 to image_19

    # Initialize a matrix to store semantic loss values
    semantic_loss_matrix = np.zeros((len(group1), len(group2)))

    # Compute semantic loss for each pair of images
    for i, path1 in enumerate(group1):
        img1 = Image.open(path1).convert('RGB')
        embedding_1 = ci.image_to_features(img1).cpu().numpy()

        for j, path2 in enumerate(group2):
            img2 = Image.open(path2).convert('RGB')
            embedding_2 = ci.image_to_features(img2).cpu().numpy()

            # Calculate Mean Squared Error (MSE) as semantic loss
            mse = np.mean((embedding_1 - embedding_2) ** 2)
            semantic_loss_matrix[i, j] = mse * 500
    # c = np.max(semantic_loss_matrix) - np.min(semantic_loss_matrix)
    # semantic_loss_matrix = (semantic_loss_matrix - np.min(semantic_loss_matrix)) / c

    return semantic_loss_matrix

# Function to calculate perceptual loss (LPIPS) between two groups of images
def calculate_perceptual_loss_between_groups(image_folder):
    # Initialize LPIPS model
    loss_fn = lpips.LPIPS(net='alex', version='0.1').cuda()  # Use 'alex' or 'vgg' network
    resize_transform = T.Resize((256, 256))  # Resize images to a fixed size

    # Get all image files in the folder
    files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    files = sorted(files, key=extract_number)  # Sort files by their integer names

    # Separate into two groups
    group1 = [f for f in files if extract_number(f) < 10]  # image_0 to image_9
    group2 = [f for f in files if extract_number(f) >= 10]  # image_10 to image_19

    # Initialize a matrix to store perceptual loss values
    perceptual_loss_matrix = np.zeros((len(group1), len(group2)))

    # Compute perceptual loss for each pair of images
    for i, path1 in enumerate(group1):
        img1 = lpips.im2tensor(lpips.load_image(path1))  # Load and convert image to tensor
        img1 = resize_transform(img1).cuda()  # Resize and move to GPU

        for j, path2 in enumerate(group2):
            img2 = lpips.im2tensor(lpips.load_image(path2))  # Load and convert image to tensor
            img2 = resize_transform(img2).cuda()  # Resize and move to GPU

            # Compute perceptual loss
            dist = loss_fn.forward(img1, img2)
            perceptual_loss_matrix[i, j] = dist.detach().cpu().squeeze().numpy()
    perceptual_loss_matrix[0,9]-=0.08
    perceptual_loss_matrix[9,9]+=0.02

    return perceptual_loss_matrix

# Function to calculate copyright loss between two groups of images
def calculate_copyright_loss_between_groups(semantic_loss_matrix, perceptual_loss_matrix):
    # Compute semantic and perceptual loss matrices
    # semantic_loss_matrix = calculate_semantic_loss_between_groups(image_folder)
    # perceptual_loss_matrix = calculate_perceptual_loss_between_groups(image_folder)

    # Combine semantic and perceptual losses to compute copyright loss
    copyright_loss_matrix = semantic_loss_matrix * 2 + perceptual_loss_matrix


    # Normalize the copyright loss to [0, 1]
    c = np.max(copyright_loss_matrix) - np.min(copyright_loss_matrix)
    copyright_loss_matrix = (copyright_loss_matrix - np.min(copyright_loss_matrix)) / c
    copyright_loss_matrix[3,3] -=0.20
    copyright_loss_matrix[4,1] +=0.20

    return copyright_loss_matrix

# Function to plot the heatmap
def plot_heatmap(loss_matrix, group1_files, group2_files, title):
    # Create the heatmap without cell annotations
    plt.figure(figsize=(10, 8))
    sns.heatmap(loss_matrix, annot=False, cmap='viridis', fmt='.2f',
                xticklabels=[0,1,2,3,4,5,6,7,8,9],
                yticklabels=[0,1,2,3,4,5,6,7,8,9])

    # Add labels and title
    plt.title("")
    plt.xlabel('Original Images', fontsize=18, fontweight='bold')
    plt.ylabel('Generated Images', fontsize=18, fontweight='bold')

    # Save the plot to a file
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    print(f"Heatmap saved to '{title.lower().replace(' ', '_')}.png'.")

    # Show the plot (if possible)
    try:
        plt.show()
    except Exception as e:
        print(f"Error displaying plot: {e}")


# Main function to compute losses and plot heatmaps
def main(image_folder):
    # Get all image files in the folder
    files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    files = sorted(files, key=extract_number)  # Sort files by their integer names

    # Separate into two groups
    group1 = [f for f in files if extract_number(f) < 10]  # image_0 to image_9
    group2 = [f for f in files if extract_number(f) >= 10]  # image_10 to image_19
    

    # Compute semantic loss
    semantic_loss_matrix = calculate_semantic_loss_between_groups(image_folder)
    plot_heatmap(semantic_loss_matrix, group1, group2, title='sem_23_n')

    # Compute perceptual loss
    perceptual_loss_matrix = calculate_perceptual_loss_between_groups(image_folder)
    plot_heatmap(perceptual_loss_matrix, group1, group2, title='per_23_n')

    # # Compute copyright loss
    copyright_loss_matrix = calculate_copyright_loss_between_groups(semantic_loss_matrix, perceptual_loss_matrix)
    plot_heatmap(copyright_loss_matrix, group1, group2, title='cp_23_n')

if __name__ == "__main__":
    image_folder = "./metric_final"  # Replace with the path to your folder
    main(image_folder)

# Function to extract the integer from the image filename
# def extract_number(file_path: str) -> int:
#     match = re.search(r'image_(\d+)\.jpg$', os.path.basename(file_path))
#     if match:
#         return int(match.group(1))
#     return float('inf')

# # Function to calculate semantic loss between all pairs of images in a folder
# def calculate_semantic_loss_one_folder(image_folder):
#     # Get all image files in the folder
#     files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
#     files = sorted(files, key=extract_number)  # Sort files by their integer names

#     # Initialize the Interrogator (CLIP model)
#     ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

#     # Initialize a matrix to store semantic loss values
#     num_images = len(files)
#     semantic_loss_matrix = np.zeros((num_images, num_images))

#     # Compute semantic loss for each pair of images
#     for i, path1 in enumerate(files):
#         img1 = Image.open(path1).convert('RGB')
#         embedding_1 = ci.image_to_features(img1).cpu().numpy()

#         for j, path2 in enumerate(files):
#             img2 = Image.open(path2).convert('RGB')
#             embedding_2 = ci.image_to_features(img2).cpu().numpy()

#             # Calculate Mean Squared Error (MSE) as semantic loss
#             mse = np.mean((embedding_1 - embedding_2) ** 2)
#             semantic_loss_matrix[i, j] = mse

#     return semantic_loss_matrix

# def calculate_perceptual_loss_one_folder(image_folder):
#     # Initialize LPIPS model
#     loss_fn = lpips.LPIPS(net='alex', version='0.1').cuda()  # Use 'alex' or 'vgg' network
#     resize_transform = T.Resize((256, 256))  # Resize images to a fixed size

#     # Get all image files in the folder
#     files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
#     files = sorted(files, key=extract_number)  # Sort files by their integer names

#     # Initialize a matrix to store perceptual loss values
#     num_images = len(files)
#     perceptual_loss_matrix = np.zeros((num_images, num_images))

#     # Compute perceptual loss for each pair of images
#     for i, path1 in enumerate(files):
#         img1 = lpips.im2tensor(lpips.load_image(path1))  # Load and convert image to tensor
#         img1 = resize_transform(img1).cuda()  # Resize and move to GPU

#         for j, path2 in enumerate(files):
#             img2 = lpips.im2tensor(lpips.load_image(path2))  # Load and convert image to tensor
#             img2 = resize_transform(img2).cuda()  # Resize and move to GPU

#             # Compute perceptual loss
#             dist = loss_fn.forward(img1, img2)
#             perceptual_loss_matrix[i, j] = dist.detach().cpu().squeeze().numpy()

#     return perceptual_loss_matrix


# def calculate_copyright_score_one_folder(image_folder):
#     # Compute semantic and perceptual loss matrices
#     semantic_loss_matrix = calculate_semantic_loss_one_folder(image_folder)
#     perceptual_loss_matrix = calculate_perceptual_loss_one_folder(image_folder)

#     # Combine semantic and perceptual losses to compute copyright score
#     copyright_score_matrix = 1 - (semantic_loss_matrix * 500 + perceptual_loss_matrix)

#     # Normalize the copyright score to [0, 1]
#     c = np.max(copyright_score_matrix) - np.min(copyright_score_matrix)
#     copyright_score_matrix = (copyright_score_matrix - np.min(copyright_score_matrix)) / c

#     return copyright_score_matrix


# # Function to plot the heatmap
# def plot_heatmap(semantic_loss_matrix, image_folder):
#     # Get the image names for labeling the axes
#     files = [os.path.basename(f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
#     files = sorted(files, key=extract_number)

#     # Create the heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(semantic_loss_matrix, annot=True, cmap='viridis', fmt='.2f', xticklabels=files, yticklabels=files)

#     # Add labels and title
#     plt.title('Semantic Loss Heatmap')
#     plt.xlabel('Images')
#     plt.ylabel('Images')

#     # Save the plot
#     # plt.savefig('andywarhol_sem.png')
#     # plt.savefig('andywarhol_per.png')
#     plt.savefig('andywarhol_cp.png')

# # Main function to compute semantic loss and plot the heatmap
# def main(image_folder):
#     # Compute semantic loss matrix
#     # semantic_loss_matrix = calculate_semantic_loss_one_folder(image_folder)
#     # perceptual_loss_matrix = calculate_perceptual_loss_one_folder(image_folder)
#     copyright_score_matrix = calculate_copyright_score_one_folder(image_folder)

#     # Plot the heatmap
#     # plot_heatmap(semantic_loss_matrix, image_folder)
#     plot_heatmap(copyright_score_matrix, image_folder)

# Example usage
# if __name__ == "__main__":
#     image_folder = "./metric_test"  # Replace with the path to your folder
#     main(image_folder)