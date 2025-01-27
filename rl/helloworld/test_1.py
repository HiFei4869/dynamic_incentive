import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from typing import List, Tuple
import subprocess
import os
import shutil
import re
import csv

def create_budget_env(**kwargs):
    return BudgetEnv(
                     steps_per_round=kwargs.get('steps_per_round', 5),
                     budget=kwargs.get('budget', 1000)
    )
gym.register(
    id='BudgetEnv-out',
    entry_point='__main__:create_budget_env',
    max_episode_steps=100,
)
ARY = np.ndarray

class BudgetEnv(gym.Env):
    def __init__(self, steps_per_round: int, budget: float):
        self.steps_per_round = steps_per_round
        self.budget = budget
        # self.get_image_quality = get_image_quality
        
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=0, high=budget, shape=(5,), dtype=np.float32)
        
        self.state = np.zeros(5, dtype=np.float32)
        self.current_step = 0
        self.current_budget = budget
        self.history_rewards: List[float] = []
        self.history_budgets: List[int] = []
        self.middle_round_zero = 0
        self.budget_fid = {}      # a dictionary recording the computed distribution plan


    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.current_budget = self.budget

        self.middle_round_zero = 0
        
        # initial state
        N_t = 2
        C_t = 1 * N_t
        D_t = 1 * N_t
        B_l_t = self.budget
        t_l = self.steps_per_round
        self.state = np.array([N_t, C_t, D_t, B_l_t, t_l], dtype=np.float32)
        return self.state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.current_step >= self.steps_per_round:
            raise ValueError("Exceeded the number of steps per round.")
        
        action_value = action * 200

        action_value = min(action_value, self.current_budget)
        
        # Update state
        N_t = action_value // 200
        C_t = 1 * N_t
        D_t = 1 * N_t
        B_l_t = self.current_budget - action_value
        t_l = self.steps_per_round - self.current_step - 1

        
        self.state = np.array([N_t, C_t, D_t, B_l_t, t_l], dtype=np.float32)
        
        # Calculate reward based on the current state
        self.history_budgets.append(action_value)
        reward = 0

        # if action_value > 0 and self.middle_round_zero == 0:
        #     print(f'step per round: {self.steps_per_round}')
        #     print(f'current step: {self.current_step}')
        # else:
        #     self.middle_round_zero = 1

        # if action_value > 0:
        #     self.middle_round_zero = 1

        reward = self.get_image_quality(self.history_rewards, self.history_budgets, self.state)

        self.current_budget -= action_value
        
        self.history_rewards.append(reward)

        done = False
        if t_l == 0:
            done = True
        else:
            self.current_step += 1
        
        return self.state, reward, done, False, {}

    def render(self, mode='human'):
        if mode == 'human':
            print("Round\tStep\tReward\tBudget Distribution")
            for i in range(len(self.history_rewards)):
                reward = self.history_rewards[i]
                budget_distribution = self.history_budgets[i]
                round_num = i // self.steps_per_round + 1
                step_num = i % self.steps_per_round + 1
                print(f"{round_num}\t{step_num}\t{reward:.2f}\t{budget_distribution}")

            for key, value in self.budget_fid.items():
                print(f"value:{value}")
        elif mode == 'ansi':
            print("Rendering in ANSI mode")
            # Implement any ASCII rendering logic if needed
        else:
            raise ValueError(f"Unsupported mode: {mode}")


    def process_input(self, input_list, round_num):

        non_zeros = [x for x in input_list if x != 0]
        zeros = [x for x in input_list if x == 0]
        input_list = non_zeros + zeros

        # Check if the last five elements match any entry in the dictionary
        for key, value in self.budget_fid.items():
            # print(f'dict: {value}')
            if value[:5] == input_list[:5]:
                return value[5]
        
        # If no match is found, add a new entry to the dictionary
        new_key = len(self.budget_fid)

        for i, budget_distribution in enumerate(self.history_budgets[round_num*5: round_num*5+5]):
            if budget_distribution == 0 and i != 4:
                break
            else:
                print(f'step:{i}; budget for this step:{budget_distribution}')
                # check = inner_train(round_num, budget_distribution) # training, get FID
                print("call inner_train!")
        new_fid = random.randint(9, 16)
        new_value = input_list[:5] + [new_fid]
        self.budget_fid[new_key] = new_value
        
        return new_fid

    def get_image_quality(self, history_rewards, history_budgets, state):
        _, C_t, D_t, _, t_l = state
        length = len(history_rewards)
        round_num = length // 5
        if t_l == 0:
            if length == 0:
                print("first round")
                return 0.01
            else:  
                print(f'round_num:{round_num}')
                # Process each budget_distribution
                
                # for i, budget_distribution in enumerate(history_budgets[round_num * 5: round_num * 5 + 5]):
                #     if budget_distribution == 0 and i != 4:
                #         break
                #     else:
                #         # check = inner_train(round_num, budget_distribution) # training, get FID
                #         print("call inner_train!")
                #         check = 1
                #         if check == 0:
                #             return -0.5

                # budget_distribution = history_budgets[length - 1]
                # if budget_distribution == 0:
                #     return -0.5  # punish distributing 0 at middle rounds
                # else:
                #     # check = inner_train(round_num, budget_distribution) # training, get FID
                #     print("call inner_train!")
                #     check = 1
                #     if check == 0:
                #         return -0.5

                fid_result = self.process_input(self.history_budgets[round_num * 5: round_num * 5 + 5], round_num)
                # if os.path.exists(round_dir):
                #     fid_result = fid(round_dir)
                # file.write(f'fid_result:{fid_result}\n')  # Write fid_result to the text file
                return 10/fid_result                      # normalize FID value
        else:
            budget_distribution = history_budgets[length - 1]
            if budget_distribution == 0:
                return -0.5
            else:
                return 0.1


def inner_train(round_num, budget):
    # dataholders = ["a","b","c","d","e","f","9","10"] # ArtBench
    # dataholders = ["i","j","k","l","m","n","o","p"] # Portrait
    dataholders = ["a","b","c","d","e","f","g","h"] # Cartoon
    mode = 0       # 0: inner RL; 1: inner random; 2: inner linear
    if mode == 0:  # contribution only

        budget_distribution = budget * np.array([0.67003, 0.11964162, 0.034562, 0.03226982, 0.02734387, 0.02987239, 0.03762612, 0.04865424]) # cartoon

    elif mode == 1:
        budget_distribution = budget * np.random.dirichlet(np.ones(8),size=1)[0]
    else:
        budget_distribution = budget * np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
    price = np.array([55, 75, 45, 65, 30, 40, 95, 160]) # Cartoon
    data_length = [50, 80, 40, 60, 25, 30, 100, 150] # Cartoon
    determine = np.where(budget_distribution - price >= 0)[0]
    train_dir = []
    num_data = 0
    for i in determine:
        dir = "cartoon_" + dataholders[i]
        train_dir.append(dir) # the training directory of the current step
        num_data += data_length[i]
    if num_data > 0:
        launch_training(round_num, train_dir, num_data)
        return 1
    else:
        return 0
        
def launch_training(round_num, train_dir, num_data):
    round_dir = f"../../generative-models/rl_logging/{round_num}"
    if not os.path.exists(round_dir):
        os.makedirs(round_dir)
        source_dir = "../../generative-models/artbench_expressionism_200/checkpoint-9500"
        target_dir = os.path.join(round_dir, "checkpoint-9500")
        shutil.copytree(source_dir, target_dir)
        checkpoint_path = f"{round_dir}/checkpoint-9500"
        last_checkpoint_number = 9500
    else:
        checkpoint_path, last_checkpoint_number = last_checkpoint(round_dir)
    
    print(f'last_checkpoint:{last_checkpoint_number}')
    print(f'num_data:{num_data}')
    print(f'start training round {round_num}')
    print(f'checkpoint_path:{checkpoint_path}')
    print(f'round_dir:{round_dir}')

    # train_dir = f"../../generative-models/{train_dir}"
    train_dir = ' '.join([f"../../generative-models/{group}" for group in train_dir])
    print(f'train_dir:{train_dir}')

    os.environ["MODEL_NAME"] = "stabilityai/stable-diffusion-xl-base-1.0"
    os.environ["VAE_NAME"] = "madebyollin/sdxl-vae-fp16-fix"
    
    os.environ["TRAIN_DIR"] = train_dir
    os.environ["CHECKPOINT_PATH"] = checkpoint_path
    os.environ["OUTPUT_DIR"] = round_dir
    os.environ["TRAIN_EPOCH"] = str(int((last_checkpoint_number + 500) * 4 / num_data + 1))
    print(os.environ["TRAIN_EPOCH"])

    # num_process = 2/4/6/8
    command = [
        "accelerate", "launch", "--num_processes", "2", "../../generative-models/train_text_to_image_lora_sdxl.py",
        "--pretrained_model_name_or_path", os.environ["MODEL_NAME"],
        "--pretrained_vae_model_name_or_path", os.environ["VAE_NAME"],
        "--train_data_dir", os.environ["TRAIN_DIR"],
        "--caption_column", "caption",
        "--resolution", "256",
        "--random_flip",
        "--train_batch_size", "1",
        "--num_train_epochs", os.environ["TRAIN_EPOCH"],
        "--checkpointing_steps", "100",
        "--learning_rate", "1e-04",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "100",
        "--mixed_precision", "fp16",
        "--seed", "42",
        "--resume_from_checkpoint", os.environ["CHECKPOINT_PATH"],
        "--output_dir", os.environ["OUTPUT_DIR"],
        "--validation_prompt", "painting of a house in a wooded area with trees and a mountain in the background, an art deco painting inspired by Alesso Baldovinetti, instagram, modernism, oil on canvas (1921), italian futurism style, cypresses and hills"
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

def last_checkpoint(round_dir):
    valid_numbers = {9500, 10000, 10500, 11000, 11500, 12000}
    
    # Compile a regex pattern
    pattern = re.compile(r"^checkpoint-(\d+)$")
    
    largest_checkpoint = None
    largest_number = -1
    
    for subfolder in os.listdir(round_dir):
        subfolder_path = os.path.join(round_dir, subfolder)
        if os.path.isdir(subfolder_path):
            match = pattern.match(subfolder)
            if match:
                checkpoint_number = int(match.group(1))
                if checkpoint_number in valid_numbers and checkpoint_number > largest_number:
                    largest_number = checkpoint_number
                    largest_checkpoint = subfolder_path # path to latest checkpoint
    
    return largest_checkpoint, largest_number

def fid(round_dir):
    ###### inference ######
    model_path, _ = last_checkpoint(round_dir)
    print(model_path)

    jsonl_file_path = '../../generative-models/shap_bg_imagefolder/metadata.jsonl'
    import json
    from diffusers import DiffusionPipeline
    import torch
    import subprocess
    import argparse

    # example: model_path = "./rl_track/3_1_0_0_0/checkpoint-10500"

    directory_path = os.path.dirname(model_path)
    model_identifier = os.path.basename(directory_path)

    output_dir = f"../../generative-models/rl_inference_output/{model_identifier}"
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)
    # pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    # pipe.to("cuda")
    # pipe.load_lora_weights(model_path)

    # output_dir = "./inference_output/3_1_0_0_0"
    # os.makedirs(output_dir, exist_ok=True)

    # Open the .jsonl file and read it line by line
    # with open(jsonl_file_path, 'r') as file:
    #     i = 0
    #     for line in file:
    #         # Parse the JSON line into a dictionary
    #         data = json.loads(line)
            
    #         # Access the desired column (e.g., 'age')
    #         prompt = data.get('caption')  # Use .get() to avoid KeyError if 'age' is missing
            
    #         #prompt = "A naruto with green eyes and red legs."
    #         image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    #         output_path = os.path.join(output_dir, f"generated_image_{i+1}.png")
    #         image.save(output_path)
    #         i += 1

    ###### fid ######
    command = [
    'python', '-m', 'pytorch_fid',
    '../../generative-models/shap_bg_imagefolder/images', output_dir,
    '--dims', '64',
    '--device', 'cuda:0'
    ]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)
    print("FID:\n", result.stdout)
    try:
        fid_value = float(result.stdout.strip().split()[-1])
    except (IndexError, ValueError):
        print("Error: Could not extract FID value from output.")

    return fid_value
    
    

import os
import sys
import gymnasium as gym
from erl_config import Config, get_gym_env_args
from erl_agent import AgentDQN
from erl_run import train_agent, valid_agent

def train_dqn_for_budget_env(gpu_id=0):
    agent_class = AgentDQN
    env_class = gym.make

    env_args = {
        'env_name': 'BudgetEnv-out',
        'state_dim': 5,
        'action_dim': 11,
        'steps_per_round': 5,
        'if_discrete': True,
        'budget': 1000
    }

    get_gym_env_args(env=gym.make('BudgetEnv-out'), if_print=True)  # Use correct env ID

    args = Config(agent_class, env_class, env_args)
    args.break_step = int(2e4)
    args.net_dims = [80, 40]
    args.gamma = 0.95
    args.gpu_id = gpu_id
    args.num_envs = 1
    args.clip_grad_norm = 0.5
    args.soft_update_tau = 5e-3
    args.state_value_tau = 5e-3
    args.explore_rate = 0.5

    env = train_agent(args)

    if input("| Press 'y' to load actor.pth and render:") == 'y':
        actor_name = sorted([s for s in os.listdir(args.cwd) if s.endswith('.pth')])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)
    

    print("Budget FID Dictionary:")
    for key, value in env.unwrapped.budget_fid.items():
        print(f"{key}: {value}")

    from datetime import datetime

    # Ensure the output directory exists
    output_dir = "./output_fid"
    os.makedirs(output_dir, exist_ok=True)

    # Generate the filename using the current date
    current_date = datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
    output_file = os.path.join(output_dir, f"{current_date}.txt")

    # Save the budget_fid dictionary to the file
    with open(output_file, "w") as file:
        for key, value in env.unwrapped.budget_fid.items():
            file.write(f"{key}: {value}\n")

    print(f"Budget FID dictionary saved to {output_file}")

if __name__ == "__main__":
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    train_dqn_for_budget_env(gpu_id=GPU_ID)

# if __name__ == "__main__":
#     round_dir = f"../../generative-models/rl_logging/0"
#     a = fid(round_dir)
#     print(f'FID output: {a}')