from PIL import Image
from clip_interrogator import Config, Interrogator
import os
from datasets import Dataset, DatasetDict, Features, Value, Image as HfImage
from huggingface_hub import HfApi, HfFolder
import warnings
warnings.filterwarnings("ignore")

# Initialize the clip interrogator
ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k"))

# Function to load and preprocess images from a folder
def load_and_preprocess_images_from_folder(folder_path, limit=200):
    image_extensions = ('.jpg', '.jpeg')
    images = []
    image_names = []
    count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(image_path).convert('RGB')
                images.append(image)
                image_names.append(filename)
                count += 1
                if count >= limit:
                    break
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
    return images, image_names

folder_path = './artbench_expressionism_600'
images, image_names = load_and_preprocess_images_from_folder(folder_path, limit=200)
captions = [ci.interrogate_classic(image) for image in images]

# Ensure both arrays have the same length
if len(image_names) != len(captions):
    raise ValueError("The arrays image_names and captions must have the same length")

# Create a dictionary for the dataset
data = {
    "image": [os.path.join(folder_path, image_name) for image_name in image_names],
    "caption": captions
}

# Define the features of the dataset
features = Features({
    'image': HfImage(),  # This will automatically handle loading images from file paths
    'caption': Value('string')
})

# Create the dataset
dataset = Dataset.from_dict(data, features=features)

# Create a DatasetDict if you have train/validation/test splits
dataset_dict = DatasetDict({'train': dataset})

# Define your Hugging Face repository details
hf_repo_name = 'HiFei4869/artbench_expressionism_200'

# Authenticate with Hugging Face (ensure you have your token set up)
hf_token = HfFolder.get_token()
if not hf_token:
    raise ValueError("Hugging Face token not found. Please authenticate using 'huggingface-cli login'.")

# Push the dataset to Hugging Face Hub
dataset_dict.push_to_hub(hf_repo_name)

print(f"Dataset successfully pushed to Hugging Face Hub: https://huggingface.co/datasets/{hf_repo_name}")
