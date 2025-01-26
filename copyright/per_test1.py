import torch
import argparse
import lpips
import os
import pandas as pd
import re
import torchvision.transforms as T

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default='./generated_image_32.png')
parser.add_argument('-p1','--path1', type=str, default='./original_32.png')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

parser.add_argument('--dir0',type=str, default='../generative-models/group_1/images')
parser.add_argument('--dir1',type=str, default='../generative-models/inference_output/fid_group_1_10500_256')
parser.add_argument('--output_csv',type=str, default='per_loss_1_10500.csv')

args = parser.parse_args()

torch.manual_seed(43)

loss_fn = lpips.LPIPS(net='alex',version=args.version)

if(args.use_gpu):
	loss_fn.cuda()

# Single images
# img0 = lpips.im2tensor(lpips.load_image(args.path0)) # RGB image from [-1,1]
# img1 = lpips.im2tensor(lpips.load_image(args.path1))
def sort_files_by_number(directory):

    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def extract_number(file_path: str) -> int:
        match = re.search(r'image_(\d+)\.jpg$', os.path.basename(file_path))
        if match:
            return int(match.group(1))
        return float('inf')

    sorted_files = sorted(files, key=extract_number)
    return sorted_files

sorted_files0 = sort_files_by_number(args.dir0)
sorted_files1 = sort_files_by_number(args.dir1)

# print(sorted_files0)
# print(sorted_files1)
resize_transform = T.Resize((256, 256))

per_loss = []
# Multiple images
for i, path in enumerate(sorted_files0):
    print(path)
    img0 = lpips.im2tensor(lpips.load_image(path))
    img1 = lpips.im2tensor(lpips.load_image(sorted_files1[i]))

    # Resize images to the same size
    img0 = resize_transform(img0)
    img1 = resize_transform(img1)

    if args.use_gpu:
        img0 = img0.cuda()
        img1 = img1.cuda()

    # Compute distance
    dist01 = loss_fn.forward(img0, img1)
    per_loss.append(dist01.detach().cpu().squeeze().numpy())
# print('Per_Loss: %.3f'%dist01)

df = pd.DataFrame(per_loss, columns=['per_loss'])

# Save the DataFrame to a CSV file
df.to_csv(args.output_csv, index=True)


################## From huggingface ##################
# import os
# import torch
# import lpips
# from datasets import load_dataset
# from PIL import Image
# import argparse
# from tqdm import tqdm
# import csv
# import numpy as np

# def calculate_perceptual_loss(hf_repo_name, local_image_folder, num_images, output_csv):
#     # Load the dataset from Hugging Face
#     dataset = load_dataset(hf_repo_name)
    
#     # Initialize LPIPS model
#     loss_fn = lpips.LPIPS(net='alex', version='0.1')

#     loss_fn.cuda()
    
#     perceptual_losses = []
    
#     with open(output_csv, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Image Index", "Perceptual Loss"])
        
#         for i in tqdm(range(num_images)):
#             # Load image from Hugging Face dataset
#             hf_image = dataset['train'][i]['image']
            
#             # Convert the image to a NumPy array and resize
#             hf_image_np = np.array(hf_image)
#             hf_image_resized = Image.fromarray(hf_image_np).resize((256, 256))
            
#             # Load corresponding local image
#             local_image_path = os.path.join(local_image_folder, f'generated_image_{i+1}.png')
#             local_image = Image.open(local_image_path).convert('RGB')
            
#             # Convert the local image to a NumPy array and resize
#             local_image_np = np.array(local_image)
#             local_image_resized = Image.fromarray(local_image_np).resize((256, 256))
            
#             # Convert images to LPIPS tensor format
#             hf_tensor = lpips.im2tensor(np.array(hf_image_resized))  # LPIPS expects images in tensor format [-1,1]
#             local_tensor = lpips.im2tensor(np.array(local_image_resized))

#             hf_tensor = hf_tensor.cuda()
#             local_tensor = local_tensor.cuda()
            

# 			# hf_tensor = hf_tensor.cuda()
# 			# local_tensor = local_tensor.cuda()
            
#             # Compute perceptual loss
#             dist = loss_fn(hf_tensor, local_tensor)
#             perceptual_loss = dist.item()
#             perceptual_losses.append(perceptual_loss)

            
#             # Write to CSV
#             writer.writerow([i+1, perceptual_loss])
    
#     return perceptual_losses

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Calculate perceptual loss between images.")
#     parser.add_argument('--hf_repo_name', type=str, default='HiFei4869/artbench_expressionism_200', help='Hugging Face repository name.')
#     parser.add_argument('--local_image_folder', type=str, default='../generative-models/inference_output/artbench_expressionism_200_2', help='Path to local folder containing generated images.')
#     parser.add_argument('--num_images', type=int, default=200, help='Number of images to process.')
#     parser.add_argument('--output_csv', type=str, default='./perceptual_loss_2.csv', help='Path to output CSV file.')
#     args = parser.parse_args()

#     # Set the seed to ensure reproduction
#     torch.manual_seed(43)
    
#     # Calculate perceptual losses
#     losses = calculate_perceptual_loss(args.hf_repo_name, args.local_image_folder, args.num_images, args.output_csv)
    
#     # Print summary
#     print(f'\nAverage Perceptual Loss: {sum(losses) / len(losses):.3f}')
