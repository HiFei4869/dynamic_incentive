import os
import subprocess
from pathlib import Path

class ModelTrainer:
    def __init__(self, round, current_step):
        self.round = round
        self.current_step = current_step

    def train_model(self, group_list, n_checkpoint):
        # Construct TRAIN_DIR
        train_dir = '../../generative-models/group_' + '+'.join(f'{i}' for i in group_list)
        print(train_dir)        
        # Ensure OUTPUT_DIR exists
        output_dir = f"../rl_tem/{self.round}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if OUTPUT_DIR is empty and self.current_step is 0, copy the checkpoint
        if self.current_step == 0 and not os.listdir(output_dir):
            checkpoint_src = "../../generative-models/artbench_expressionism_200/checkpoint-9500"
            subprocess.run(f"cp -r {checkpoint_src} {output_dir}", shell=True)
        
        # Construct CHECKPOINT_PATH
        checkpoint_path = f'../rl_tem/{self.round}/checkpoint-{self.current_step * 500 + n_checkpoint}'

        # Environment variables
        os.environ["MODEL_NAME"] = "stabilityai/stable-diffusion-xl-base-1.0"
        os.environ["VAE_NAME"] = "madebyollin/sdxl-vae-fp16-fix"
        os.environ["TRAIN_DIR"] = train_dir
        os.environ["CHECKPOINT_PATH"] = checkpoint_path
        os.environ["OUTPUT_DIR"] = output_dir

        # Command to run the training script
        command = [
            "accelerate", "launch", "--num_processes", "2", "train_text_to_image_lora_sdxl.py",
            "--pretrained_model_name_or_path=$MODEL_NAME",
            "--pretrained_vae_model_name_or_path=$VAE_NAME",
            "--train_data_dir=$TRAIN_DIR", "--caption_column=caption",
            "--resolution=256", "--random_flip",
            "--train_batch_size=1", "--num_train_epochs=100",
            "--checkpointing_steps=100", "--learning_rate=1e-04",
            "--lr_scheduler=constant", "--lr_warmup_steps=100",
            "--mixed_precision=fp16", "--seed=42",
            f"--resume_from_checkpoint=$CHECKPOINT_PATH",
            "--output_dir=$OUTPUT_DIR",
            "--validation_prompt='a close up of a drawing of a man with glasses and a mustache, an ink drawing inspired by Stanisaw Tondos, reddit, shin hanga, man with glasses, portrait of sigmund freud, stanisaw'"
        ]

        # Run the command and hide the output
        with subprocess.Popen(' '.join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            out, err = proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"Training failed with error: {err.decode()}")

if __name__ == "__main__":
    trainer = ModelTrainer(round=1, current_step=0)
    trainer.train_model(group_list=[1, 3, 4], n_checkpoint=9500)

