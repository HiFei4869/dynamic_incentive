from diffusers import DiffusionPipeline
import torch
import argparse

torch.manual_seed(42)
model_path = "../generative-models/artbench_expressionism_200/pytorch_lora_weights.safetensors"
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipe.to("cuda")
pipe.load_lora_weights(model_path)

# prompt = "A portrait of an elderly woman with a white ruffled collar and dark clothing, set against a dark background."
# prompt = "A black and white sketch of a man sitting on a stool, with his back to the viewer."
# prompt = "A still life painting features a basket of fruit, including apples, pears, and oranges, on a table with a white cloth and a teapot."

# prompt = "A man in a black suit with a red flower on the lapel sits in a chair, looking at the viewer."
# prompt = "A painting by the artist Lali, signed '1912', features a group of nude figures in various poses against a blue background."
parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
    "--prompt",
    type=str,
    default="A painting by the artist Picasso, features old twin men in black clothes.",
    help="Prompt of the inference.",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="./metric_data/image_29.png",
    help="Path to store the inference output.",
)
args = parser.parse_args()
# prompt = "a painting of a woman with a bare chest and no shirt, an oil painting inspired by Raphael Soyer, reddit, figuration libre, half body portrait of juliana, portrait of a female model, upper body portrait"
image = pipe(args.prompt, num_inference_steps=100, guidance_scale=7.5).images[0]
image.save(args.output_path)



################## Multiple Image Inference ####################
# import argparse
# from datasets import load_dataset
# from diffusers import DiffusionPipeline
# import torch
# import os

# def get_captions_from_hf(repo_name):
#     # Load the dataset from Hugging Face
#     dataset = load_dataset(repo_name)
#     captions = dataset['train']['caption']
#     return captions

# def generate_images(captions, model_path, output_dir, num_images=10):
#     # Set manual seed for reproducibility
#     torch.manual_seed(42)

#     # Load the diffusion pipeline
#     pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
#     pipe.to("cuda")
#     pipe.load_lora_weights(model_path)

#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Generate images for the given captions
#     for i, prompt in enumerate(captions[:num_images]):
#         print(f"Generating image {i+1}/{num_images} for prompt: {prompt}")
#         image = pipe(prompt, num_inference_steps=100, guidance_scale=5).images[0]
#         output_path = os.path.join(output_dir, f"generated_image_{i+1}.png")
#         image.save(output_path)
#         print(f"Image saved to {output_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate images from captions.")
#     parser.add_argument(
#         "--hf_repo_name",
#         type=str,
#         default="HiFei4869/artbench_expressionism_200",
#         help="Hugging Face repository name to load captions from.",
#     )
#     parser.add_argument(
#         "--model_path",
#         type=str,
#         default="artbench_expressionism_200/pytorch_lora_weights.safetensors",
#         help="Path to the LoRA weights file.",
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="./inference_output/artbench_expressionism_200",
#         help="Directory to save the generated images.",
#     )
#     parser.add_argument(
#         "--num_images",
#         type=int,
#         default=200,
#         help="Number of images to generate.",
#     )
#     args = parser.parse_args()

#     # Get captions from Hugging Face repository
#     captions = get_captions_from_hf(args.hf_repo_name)
    
#     # Generate images using the retrieved captions
#     generate_images(captions, args.model_path, args.output_dir, args.num_images)
