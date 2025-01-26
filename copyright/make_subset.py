# import os
# from datasets import load_dataset, Dataset
# import pandas as pd
# import json
# # # Load the original dataset
# # dataset = load_dataset("AterMors/wikiart_recaption")

# # # Select the first 100 examples
# # subset = dataset['train'].select(range(1000))

# # # Prepare the data for saving
# # images = subset['image']
# # captions = subset['text']

# # # Define the directory where the subset images will be saved
# # subset_dir = 'wikiart_subset_1000'
# # os.makedirs(subset_dir, exist_ok=True)

# # # Save images locally
# # image_paths = []
# # for i, image in enumerate(images):
# #     image_path = os.path.join(subset_dir, f'image_{i}.jpg')
# #     image.save(image_path)
# #     image_paths.append(image_path)

# # # Create a DataFrame for captions
# # df = pd.DataFrame({'image': image_paths, 'caption': captions})

# # # Save the DataFrame as a CSV file
# # csv_file_path = os.path.join(subset_dir, 'captions.csv')
# # df.to_csv(csv_file_path, index=False)

# # # Verify the CSV file
# # print(f"CSV file '{csv_file_path}' created successfully with {len(df)} entries.")

# from datasets import Dataset, Features, Value, Image

# # # Define the features of the dataset
# # features = Features({
# #     'image': Image(),
# #     'caption': Value('string')
# # })


# # # Create dataset from the DataFrame
# # data = {
# #     'image': image_paths,
# #     'caption': captions
# # }
# def load_image(example):
#     return {"image": Image.open(example["image"]).convert("RGB")}

# # Path to your metadata.jsonl file
# metadata_path = '../generative-models/artbench_expressionism_200_3/metadata.jsonl'

# # Read the metadata file
# data = []
# with open(metadata_path, 'r') as f:
#     for line in f:
#         data.append(json.loads(line.strip()))

# # Create the dataset
# dataset = Dataset.from_list(data)
# #dataset = Dataset.from_dict(data, features=features)

# # Define dataset info
# # dataset = dataset.cast_column("image", Image())

# # Save dataset locally
# dataset.save_to_disk(subset_dir)

# # Upload dataset to Hugging Face Hub
# dataset.push_to_hub("HiFei4869/artbench_expressionism_200_3")

from datasets import Dataset, Features, Value, Image
from huggingface_hub import HfApi, HfFolder
from PIL import Image as PILImage
import json
import os

# Function to load images
def load_image(example):
    # Construct the correct image path
    image_path = os.path.join(os.path.dirname(metadata_path), example["file_name"])
    return {"image": PILImage.open(image_path).convert("RGB")}

# Path to your metadata.jsonl file
metadata_path = '../generative-models/artbench_expressionism_200_3/metadata.jsonl'

# Read the metadata file
data = []
with open(metadata_path, 'r') as f:
    for line in f:
        data.append(json.loads(line.strip()))

# Create the dataset
dataset = Dataset.from_list(data)

# Apply the image loading function
dataset = dataset.map(load_image, remove_columns=["file_name"], batched=False)

# Define features for the dataset
features = Features({
    "image": Image(),
    "text": Value("string"),
})

# Cast dataset to the correct feature types
dataset = dataset.cast(features)

# Save the dataset locally
subset_dir = '../generative-models/artbench_expressionism_200_3/saved_dataset'
dataset.save_to_disk(subset_dir)

# Authenticate to the Hugging Face Hub
api = HfApi()
token = HfFolder.get_token()  # Ensure you have your token stored in ~/.huggingface/token

# Push dataset to the Hugging Face Hub
dataset.push_to_hub("HiFei4869/artbench_expressionism_200_3", token=token)



