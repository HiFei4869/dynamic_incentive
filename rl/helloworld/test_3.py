import subprocess
import os

dataset = "c"
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
    # dataholders = ["a", "b"] # Cartoon
    dataset_names = ["cartoon_" + item for item in dataholders]
    data_root = "cartoon"


# Dictionary to store the results
results = {}

# Base command template
base_command = [
    "python", "grad_sdxl.py",
    "--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
    "--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix",
    "--caption_column=caption",
    "--enable_xformers_memory_efficient_attention",
    "--resolution=512", "--center_crop", "--random_flip",
    "--train_batch_size=1",
    "--gradient_accumulation_steps=4", "--gradient_checkpointing",
    "--max_train_steps=10000",
    "--use_8bit_adam",
    "--learning_rate=1e-06", "--lr_scheduler=constant", "--lr_warmup_steps=0",
    "--mixed_precision=fp16",
    "--report_to=wandb",
    "--validation_prompt=a cute Sundar Pichai creature", "--validation_epochs=5",
    "--checkpointing_steps=5000",
    "--trak_output_dir=trak_output",
    "--e_seed=42",
    "--K=10",
    "--Z=32768"
]
trak_results = []
# Iterate over dataset names
for dataset_name in dataset_names:
    # Set the dataset name and output directory
    os.environ["DATASET_NAME"] = f"../../generative-models/{dataset_name}"
    os.environ["OUTPUT_DIR"] = f"../../generative-models/{data_root}/checkpoint-9500"
    
    # Construct the full command
    command = base_command + [
        "--train_data_dir=" + os.environ["DATASET_NAME"],
        "--output_dir=" + os.environ["OUTPUT_DIR"]
    ]
    
    # Run the command and capture the output
    result = subprocess.run(command, capture_output=True, text=True)
    # print(f"got one:{result}")

    
    # Assuming the generated value is printed as the last line of the output
    generated_value = result.stdout.strip().split('\n')[-1]
    
    print(generated_value)

    trak_results.append(float(generated_value))

# for dataset, value in results.items():
#     print(f"Dataset: {dataset}, Generated Value: {value}")

print(trak_results)

contribution = [x/sum(trak_results) for x in trak_results]
print(contribution)