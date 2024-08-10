import argparse
import logging
import math
import os
import re
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import datasets
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import shap
from typing import Dict, List, Tuple

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from sem_test import eval_sim
from regression import sv_diff, KernelRegression

#### preparation functions #####
check_min_version("0.30.0.dev0")

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")
    
DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}

os.environ['CHECKPOINT_PATH'] = "../generative-models/artbench_expressionism_200/checkpoint-9500"
os.environ['OUTPUT_DIR'] = "../generative-models/artbench_expressionism_200"
#os.environ['DATASET_NAME'] = "../generative-models/artbench_expressionism_200"
os.environ['DATASET_DIR'] = "../generative-models/artbench_11"
# os.environ['NUM_TRAIN_EPOCHS'] = "951"
os.environ['VAE_NAME'] = "madebyollin/sdxl-vae-fp16-fix"
global tokenizer1, tokenizer2
tokenizer1 = None
tokenizer2 = None


#### argparse ####
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    # Argument definitions with updated default values
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=os.getenv("VAE_NAME", None),
        help="Path to pretrained VAE model with better numerical stability.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, e.g. fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=os.getenv("DATASET_DIR", None),
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--shap_bg_dir",
        type=str,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--train_with_jsonl",
        action="store_true",
        help=(
            "train with customized jsonl file"
        ),
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getenv("OUTPUT_DIR", "sd-model-finetuned-lora"),
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="Whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=int(os.getenv("NUM_TRAIN_EPOCHS", 100)))
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=os.getenv("CHECKPOINT_PATH", None),
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--enable_npu_flash_attention", action="store_true", help="Whether or not to use npu flash attention."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--debug_loss",
        action="store_true",
        help="debug loss for each image, if filenames are available in the dataset",
    )

    # Parse the arguments
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Environment variable for local rank
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

#### Convert to dataframe and back ####
def dataloader_to_dataframe(dataloader):
    # Initialize lists to store the data
    pixel_values_list = []
    input_ids_one_list = []
    input_ids_two_list = []
    original_sizes_list = []
    crop_top_lefts_list = []
    devices_list = []

    # Iterate over the dataloader
    for batch in dataloader:
        # Store device information
        device = batch['pixel_values'].device

        # Move tensors to CPU and convert to lists
        pixel_values = batch['pixel_values'].cpu().numpy().tolist()
        input_ids_one = batch['input_ids_one'].cpu().numpy().tolist()
        input_ids_two = batch['input_ids_two'].cpu().numpy().tolist()
        original_sizes = batch['original_sizes']
        crop_top_lefts = batch['crop_top_lefts']

        # Append the data to the corresponding lists
        pixel_values_list.extend(pixel_values)
        input_ids_one_list.extend(input_ids_one)
        input_ids_two_list.extend(input_ids_two)
        original_sizes_list.extend(original_sizes)
        crop_top_lefts_list.extend(crop_top_lefts)
        devices_list.append('cuda:0')  # Store device as string

    # Create a DataFrame from the lists
    data = {
        'pixel_values': pixel_values_list,
        'input_ids_one': input_ids_one_list,
        'input_ids_two': input_ids_two_list,
        'original_sizes': original_sizes_list,
        'crop_top_lefts': crop_top_lefts_list,
        'device': devices_list
    }

    df = pd.DataFrame(data)
    return df

def dataloader_to_numpy(dataloader):
    numpy_dict = {}
    
    for batch in dataloader:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Convert tensor to numpy array
                numpy_array = value.cpu().numpy()  # Move to CPU and convert to NumPy array
                if key in numpy_dict:
                    numpy_dict[key] = np.concatenate((numpy_dict[key], numpy_array), axis=0)
                else:
                    numpy_dict[key] = numpy_array
            else:
                # Handle lists or tuples
                numpy_array = np.array(value)
                if key in numpy_dict:
                    numpy_dict[key] = np.concatenate((numpy_dict[key], numpy_array), axis=0)
                else:
                    numpy_dict[key] = numpy_array
    
    return numpy_dict


# New
def dataframe_to_dicts(df):
    data_dicts = []
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    for _, row in df.iterrows():
        device = torch.device(row['device'])  # Convert device string back to torch.device
        
        data_dict = {
            'pixel_values': torch.tensor(row['pixel_values'], device=device).unsqueeze(0),
            'input_ids_one': torch.tensor(row['input_ids_one'], device=device).unsqueeze(0),
            'input_ids_two': torch.tensor(row['input_ids_two'], device=device).unsqueeze(0),
            'original_sizes': [tuple(row['original_sizes'])],
            'crop_top_lefts': [tuple(row['crop_top_lefts'])]
        }
        data_dicts.append(data_dict)
    return data_dicts

def numpy_to_dict(df, device):
    # Convert the dataframe to a list of dictionaries
    data_dicts = df.to_dict(orient='records')
    
    for data_dict in data_dicts:
        # Process each key-value pair in the dictionary
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                # If the value is a numpy array, convert it to a torch tensor
                # and send it to the specified device
                value = torch.tensor(value).to(device)
                
                # Special handling for 'original_sizes' and 'crop_top_lefts'
                if key in ['original_sizes', 'crop_top_lefts']:
                    # Ensure the values are lists of tuples
                    value = [tuple(x) for x in value.tolist()]
            
            elif isinstance(value, list):
                # Convert lists to tensors if they contain numpy arrays
                value = [torch.tensor(x).to(device) if isinstance(x, np.ndarray) else x for x in value]
                # Special handling for 'original_sizes' and 'crop_top_lefts'
                if key in ['original_sizes', 'crop_top_lefts']:
                    # Ensure the values are lists of tuples
                    value = [tuple(x) for x in value]
                    
            data_dict[key] = value
    
    return data_dicts

#### Main ####
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.set_device(1)  # Set cuda:1 as the default device
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')


    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    unet.to(accelerator.device, dtype=weight_dtype)

    if args.pretrained_vae_model_name_or_path is None:
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            unet.enable_npu_flash_attention()
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # now we will add new LoRA weights to the attention layers
    # Set correct lora layers
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)

    # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
    if args.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder attn layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                elif isinstance(unwrap_model(model), type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif isinstance(unwrap_model(model), type(unwrap_model(text_encoder_two))):
                    text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(input_dir)
        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.train_text_encoder:
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [unet_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_, text_encoder_two_])
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        if args.train_text_encoder:
            models.extend([text_encoder_one, text_encoder_two])
        cast_training_params(models, dtype=torch.float32)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if args.train_text_encoder:
        params_to_optimize = (
            params_to_optimize
            + list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
            + list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))
        )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:

        # Downloading and loading a dataset from the hub.
        # dataset = load_dataset(
        #     args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, data_dir=args.train_data_dir
        # )
        dataset = load_dataset(args.dataset_name)
    else:
        data_files = {}
        data_files_bg = {}
        if args.train_data_dir is not None:
            if args.train_with_jsonl is False:
                data_files["train"] = os.path.join(args.train_data_dir, "**")
                dataset = load_dataset(
                    "imagefolder",
                    data_files=data_files,
                    cache_dir=args.cache_dir,
                )
            else:
                data_files["train"] = os.path.join(args.train_data_dir, "metadata.jsonl")
                dataset = load_dataset(
                    "json",
                    data_files=data_files,
                    cache_dir=args.cache_dir,
                )
        data_files_bg["train"] = os.path.join(args.shap_bg_dir, "**")

        dataset_bg = load_dataset(
                    "imagefolder",
                    data_files=data_files_bg,
                    cache_dir=args.cache_dir,
                )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.


    image_column = args.image_column
    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
        )


    caption_column = args.caption_column
    if caption_column not in column_names:
        raise ValueError(
            f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
        )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        tokens_one = tokenize_prompt(tokenizer_one, captions)
        tokens_two = tokenize_prompt(tokenizer_two, captions)
        return tokens_one, tokens_two

    # Preprocessing the datasets.
    train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    def numerical_sort(value):
        parts = re.split(r'(\d+)', value)
        return [int(part) if part.isdigit() else part for part in parts]

    def sort_dataset(dataset):
        sorted_indices = sorted(
            range(len(dataset)),
            key=lambda idx: numerical_sort(os.path.basename(dataset[idx][image_column].filename))
        )
        return dataset.select(sorted_indices)

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        # image aug
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        tokens_one, tokens_two = tokenize_captions(examples)
        examples["input_ids_one"] = tokens_one
        examples["input_ids_two"] = tokens_two
        if args.debug_loss:
            fnames = [os.path.basename(image.filename) for image in examples[image_column] if image.filename]
            if fnames:
                examples["filenames"] = fnames
        # example type: dict
        return examples

    # with accelerator.main_process_first():
    #     if args.max_train_samples is not None:
    #         dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))

    #     # train_dataset = dataset["train"].with_transform(preprocess_train, output_all_columns=True)
    #     # Sort dataset
    #     sorted_train_dataset = sort_dataset(dataset["train"])

    #     # Apply transformation
    #     train_dataset = sorted_train_dataset.with_transform(preprocess_train, output_all_columns=True)

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        
        # Calculate the split index
        train_size = len(dataset["train"])
        split_index = train_size // 2
        
        # Split the dataset
        if accelerator.local_process_index == 0:
            subset_indices = list(range(split_index))
        else:
            subset_indices = list(range(split_index, train_size))
        
        train_dataset = dataset["train"].select(subset_indices)

        # Set the training transforms
        # train_dataset = train_dataset.with_transform(preprocess_train, output_all_columns=True)

        sorted_train_dataset = sort_dataset(dataset["train"])

        train_dataset = sorted_train_dataset.with_transform(preprocess_train, output_all_columns=True)
        train_dataset_bg = dataset_bg["train"].with_transform(preprocess_train, output_all_columns=True)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]
        input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
        input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
        result = {
            "pixel_values": pixel_values,
            "input_ids_one": input_ids_one,
            "input_ids_two": input_ids_two,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
        }

        filenames = [example["filenames"] for example in examples if "filenames" in example]
        if filenames:
            result["filenames"] = filenames
            with accelerator.main_process_first():
                print(filenames)
        return result

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    train_dataloader1 = torch.utils.data.DataLoader(
        train_dataset_bg,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) # gradient_accumulation_steps=1
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))
    
    return accelerator, unet, vae, optimizer, noise_scheduler, text_encoder_one, text_encoder_two, lr_scheduler, num_update_steps_per_epoch, train_dataloader, train_dataloader1, params_to_optimize

#### Self defined ####
class Trainer:
    def __init__(self, args, accelerator, unet, vae, optimizer, noise_scheduler, text_encoder_one, text_encoder_two, lr_scheduler, num_update_steps_per_epoch, train_dataloader, params_to_optimize):
        self.args = args
        self.unet = unet
        self.text_encoder_one = text_encoder_one
        self.text_encoder_two = text_encoder_two
        self.accelerator = accelerator
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.weight_dtype = torch.float16
        self.global_step = 0
        self.progress_bar = None
        self.num_update_steps_per_epoch = num_update_steps_per_epoch
        self.first_epoch = 0
        self.train_dataloader = train_dataloader
    
    def prepare_for_train(self, args):
        # Assume resuming from checkpoints
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            # print("global_step:", global_step)
            # print("num_update_steps_per_epoch:", num_update_steps_per_epoch)
            first_epoch = global_step // self.num_update_steps_per_epoch


        progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )
        self.progress_bar = progress_bar
        self.global_step = global_step
        self.first_epoch = first_epoch
        print("--Finish prepare_for_train!--")
    
    def train(self, df):
        #print(df)
        data_dicts = dataframe_to_dicts(df)
        results = []
        #print("In trainer.train, df.shape", df.shape)
        print("In trainer.train, dict len", len(data_dicts))
        #print(df)
        #print("first_epoch:",self.first_epoch)
        #print("num_train_epochs:", self.args.num_train_epochs)
        for epoch in range(self.first_epoch, self.args.num_train_epochs):
            print(f'{self.first_epoch}, {self.args.num_train_epochs}')
            self.unet.train()
            if self.args.train_text_encoder:
                self.text_encoder_one.train()
                self.text_encoder_two.train()
            train_loss = 0.0
            for batch in data_dicts:
                with self.accelerator.accumulate(unet):
                    # Convert images to latent space
                    if self.args.pretrained_vae_model_name_or_path is not None:
                        pixel_values = batch["pixel_values"].to(dtype=self.weight_dtype)
                    else:
                        pixel_values = batch["pixel_values"]
                    model_input = self.vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * self.vae.config.scaling_factor
                    
                    if self.args.pretrained_vae_model_name_or_path is None:
                        model_input = model_input.to(self.weight_dtype)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    if self.args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += self.args.noise_offset * torch.randn(
                            (model_input.shape[0], model_input.shape[1], 1, 1), device=model_input.device
                        )

                    bsz = model_input.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                    timesteps = timesteps.long()

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)
                    # time ids
                    def compute_time_ids(original_size, crops_coords_top_left):
                        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                        target_size = (args.resolution, args.resolution)
                        add_time_ids = list(original_size + crops_coords_top_left + target_size)
                        add_time_ids = torch.tensor([add_time_ids])
                        add_time_ids = add_time_ids.to(accelerator.device, dtype=self.weight_dtype)
                        return add_time_ids

                    add_time_ids = torch.cat(
                        [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                    )

                    # Predict the noise residual
                    unet_added_conditions = {"time_ids": add_time_ids}
                    #print(batch["input_ids_one"].shape)
                    #print(batch["input_ids_two"].shape)
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[self.text_encoder_one, self.text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[batch["input_ids_one"], batch["input_ids_two"]],
                    )
                    unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
                    model_pred = self.unet(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds,
                        added_cond_kwargs=unet_added_conditions,
                        return_dict=False,
                    )[0]

                    # Get the target for loss depending on the prediction type
                    if self.args.prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        self.noise_scheduler.register_to_config(prediction_type=self.args.prediction_type)

                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    if self.args.snr_gamma is None:

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        ft_compute_sample_grad = loss.reshape(1, -1).mean()
                        #print(f'ft_compute_sample_grad:{ft_compute_sample_grad}')

                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(self.noise_scheduler, timesteps)
                        mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                            dim=1
                        )[0]
                        if self.noise_scheduler.config.prediction_type == "epsilon":
                            mse_loss_weights = mse_loss_weights / snr
                        elif self.noise_scheduler.config.prediction_type == "v_prediction":
                            mse_loss_weights = mse_loss_weights / (snr + 1)

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(self.args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / self.args.gradient_accumulation_steps
                    #print(f'avg_loss:{avg_loss}')
                    ft_compute_sample_grad = avg_loss
                    # Backpropagate
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # # Checks if the accelerator has performed an optimization step behind the scenes
                # if self.accelerator.sync_gradients:
                #     self.progress_bar.update(1)
                #     self.global_step += 1
                #     self.accelerator.log({"train_loss": train_loss}, step=self.global_step)
                #     train_loss = 0.0

                #     # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                #     if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
                #         if self.global_step % self.args.checkpointing_steps == 0:
                #             # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                #             if self.args.checkpoints_total_limit is not None:
                #                 checkpoints = os.listdir(self.args.output_dir)
                #                 checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                #                 checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                #                 # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                #                 if len(checkpoints) >= self.args.checkpoints_total_limit:
                #                     num_to_remove = len(checkpoints) - self.args.checkpoints_total_limit + 1
                #                     removing_checkpoints = checkpoints[0:num_to_remove]

                #                     logger.info(
                #                         f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                #                     )
                #                     logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                #                     for removing_checkpoint in removing_checkpoints:
                #                         removing_checkpoint = os.path.join(self.args.output_dir, removing_checkpoint)
                #                         shutil.rmtree(removing_checkpoint)

                #             save_path = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
                #             self.accelerator.save_state(save_path)
                #             logger.info(f"Saved state to {save_path}")


                logs = {"step_loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                self.progress_bar.set_postfix(**logs)

                if self.global_step >= self.args.max_train_steps:
                    break

                # results.append({
                #     'mean_squared_l2_norm': ft_compute_sample_grad.item()
                # })
                #print("inside train for batch:", ft_compute_sample_grad.item())
                results.append(ft_compute_sample_grad.item())
                #print(results)

            if self.accelerator.is_main_process:
                if self.args.validation_prompt is not None and epoch % self.args.validation_epochs == 0:
                    # create pipeline
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        self.args.pretrained_model_name_or_path,
                        vae=self.vae,
                        text_encoder=unwrap_model(self.text_encoder_one),
                        text_encoder_2=unwrap_model(self.text_encoder_two),
                        unet=unwrap_model(self.unet),
                        revision=self.args.revision,
                        variant=self.args.variant,
                        torch_dtype=self.weight_dtype,
                    )

                    images = log_validation(pipeline, args, self.accelerator, epoch)

                    del pipeline
                    torch.cuda.empty_cache()

        results_df = pd.DataFrame(results)
        print(f"inside train:{len(results_df)}")
        self.accelerator.wait_for_everyone()

        # if len(results) > 0:
        #     results_tensor = torch.tensor(results, device=self.accelerator.device)
        # else:
        #     # Create a tensor filled with zeros if results is empty
        #     results_tensor = torch.zeros(1, device=self.accelerator.device)

        # gathered_results = self.accelerator.gather(results_tensor)
        # # print(f'{self.accelerator.device}, {gathered_results}')

        # # results_df = pd.DataFrame(gathered_results.cpu().numpy(), columns=["mean_squared_l2_norm"])
        # gathered_results = gathered_results.view(-1, 1)

        # # Convert gathered results to a DataFrame
        # results_df = pd.DataFrame(gathered_results.cpu().numpy(), columns=["mean_squared_l2_norm"])
        self.accelerator.end_training()

        return results_df


def get_shapley_per_sample(array, k):
    
    # Calculate indices for (kn-1)-th rows
    indices = (np.arange(1, (array.shape[0] // k) + 1) * k - 1)
    indices_1 = indices
    indices = indices - 1

    selected_rows = array[indices]
    selected_rows_1 = array[indices_1]

    return np.sum(selected_rows), np.sum(selected_rows_1)


if __name__ == "__main__":
    args = parse_args()
    accelerator, unet, vae, optimizer, noise_scheduler, text_encoder_one, text_encoder_two, lr_scheduler, num_update_steps_per_epoch, train_dataloader, train_dataloader1, params_to_optimize = main(args)
    trainer = Trainer(args, accelerator, unet, vae, optimizer, noise_scheduler, text_encoder_one, text_encoder_two, lr_scheduler, num_update_steps_per_epoch, train_dataloader, params_to_optimize)
    df = dataloader_to_dataframe(train_dataloader)
    df_bg = dataloader_to_dataframe(train_dataloader1)

    # Heuristic KNN+ ##################################################################
    # n = len(df)
    # n = 11
    # k = 4
    # image_path = os.path.join(args.train_data_dir, "images/train", f"image_{k}.png")
    # directory_path = os.path.join(args.train_data_dir, "images/train")
    # simlist = eval_sim(directory_path, image_path)
    # print(simlist)

    # file_path_1 = 'shap_results_10_index.csv'  # Replace with the path to your first CSV file
    # file_path_2 = 'shap_results_11_index.csv'  # Replace with the path to your second CSV file
    # x_train = simlist[:10]
    # y_train = sv_diff(file_path_1, file_path_2)
    # kr = KernelRegression(bandwidth=1.0)
    # kr.fit(x_train, y_train)
    # predictions = kr.predict(x_train)
    # print(predictions)

    # df.to_csv('df_tem.csv', index=False)
    # df_sampled = df.sample(1)
    # simi_list = []
    # feature_cols = ['pixel_values', 'input_ids_one', 'input_ids_two']
    # #print(df_sampled.shape)
    # for _, row in df.iterrows():
    #     row = row.to_frame().T
    #     print(row.shape)
    #     print(row)
    #     simi = eval_simi(row, df_sampled, feature_cols, simi_type='ed')
    #     simi_list.append(simi)
    
    # print(simi_list)
    # Heuristic KNN+ ###################################################################

    # print("shape of df:", df.shape)

    trainer.prepare_for_train(args)
    #print(f"output of trainer.train: {trainer.train(df)}")

    explainer = shap.KernelExplainer(trainer.train, df_bg, keep_index=True, keep_index_ordered=True)

    # print(f"output of trainer.train: {trainer.train(df)} from {accelerator.device}")
    
    shap_values = explainer.shap_values(df)
    print(shap_values)
    accelerator.wait_for_everyone()
    gathered_results = accelerator.gather(torch.tensor(shap_values, device=accelerator.device))
    print(gathered_results)

    if accelerator.is_main_process:
        #print(gathered_results)

        numpy_results = gathered_results.cpu().numpy()

        shap_per_sample = get_shapley_per_sample(numpy_results, df.shape[0])
        shap_df = pd.DataFrame(shap_per_sample)
        #shap_df.to_csv("shap_results_test.csv")
        shap_raw = pd.DataFrame(numpy_results)
        shap_raw.to_csv("shap_group1_10000.csv")
        print(shap_per_sample)


    accelerator.end_training()

    
