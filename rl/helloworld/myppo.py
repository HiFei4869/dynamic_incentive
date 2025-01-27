import os
import sys
import gymnasium as gym

from erl_config import Config, get_gym_env_args
from erl_agent import AgentPPO
from erl_run import train_agent, valid_agent
from erl_env import PendulumEnv

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
    return Custom_out(
                     steps_per_round=kwargs.get('steps_per_round', 5),
                     budget=kwargs.get('budget', 1000),
                     get_image_quality=kwargs.get('get_image_quality', get_image_quality))

gym.register(
    id='Custom_out',
    entry_point='__main__:create_budget_env',
    max_episode_steps=100,
)
ARY = np.ndarray

class Custom_out(gym.Env):
    def __init__(self, steps_per_round: int, budget: float, get_image_quality):
        self.steps_per_round = steps_per_round
        self.budget = budget
        self.get_image_quality = get_image_quality
        
        self.action_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=budget, shape=(5,), dtype=np.float32)
        
        self.state = np.zeros(5, dtype=np.float32)
        self.current_step = 0
        self.current_budget = budget
        self.history_rewards: List[float] = []
        self.history_budgets: List[int] = []


    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.current_budget = self.budget
        self.history_rewards = []
        self.history_budgets = []
        
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
        if action_value > 0:
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
        elif mode == 'ansi':
            print("Rendering in ANSI mode")
            # Implement any ASCII rendering logic if needed
        else:
            raise ValueError(f"Unsupported mode: {mode}")


def get_image_quality(history_rewards, history_budgets, state):
    _, C_t, D_t, _, t_l = state
    if t_l == 0:
        if len(history_rewards) == 0:
            return 0.01
        else:
            round_num = len(history_rewards) // 5
            print(round_num)
            with open("./budget_fid_log.txt", mode='a') as file:
                # Write round number
                file.write(f'round_num:{round_num}\n')

                # Collect all budget_distributions in a single line
                budget_distributions = [str(history_budgets[i]) for i in range(round_num * 5, round_num * 5 + 5)]
                file.write(" ".join(budget_distributions) + '\n')  # Join and write on a single line
                
                # Process each budget_distribution
                for budget_distribution in history_budgets[round_num * 5: round_num * 5 + 5]:
                    if budget_distribution == 0:
                        return -0.5  # punish distributing 0 at middle rounds
                    else:
                        check = inner_train(round_num, budget_distribution)
                        if check == 0:
                            return -0.5

                # Log fid result
                round_dir = f"../../generative-models/rl_logging/{round_num}"
                fid_result = 24
                if os.path.exists(round_dir):
                    fid_result = fid(round_dir)
                file.write(f'fid_result:{fid_result}\n')  # Write fid_result to the text file
                return 10/fid_result
    else:
        return 0.01


def inner_train(round_num, budget):
    # dataholders = ["a","b","c","d","e","f","9","10"] # ArtBench
    # dataholders = ["i","j","k","l","m","n","o","p"] # Portrait
    dataholders = ["a","b","c","d","e","f","g","h"] # Cartoon
    mode = 0       # 0: inner RL; 1: inner random; 2: inner linear
    if mode == 0:  # contribution only
        # budget_distribution = budget * np.array([0.41929144, 0.12463295, 0.07377037, 0.07244312, 0.06626723, 0.06589315, 0.08088747, 0.0968142])
        # budget_distribution = budget * np.array([1, 0, 0, 0, 0, 0, 0, 0])
        # budget_distribution = budget * np.array([0.12648772, 0.12831376, 0.11076288, 0.12974688, 0.12136942, 0.11740535, 0.1306551, 0.13525891])
        # [0.11461563, 0.13077076, 0.11106646, 0.12858988, 0.13041177, 0.12012152, 0.13342872, 0.13099532]
        # budget_distribution = budget * np.array([0.11741429, 0.13432065, 0.11099451, 0.12883137, 0.12621321, 0.11972708, 0.13195506, 0.13054389]) # portrait
        # budget_distribution = budget * np.array([0.12127227,0.13276975, 0.11028866, 0.1277392,  0.12527613, 0.11913971, 0.13001741, 0.13349685]) # p contribution
        budget_distribution = budget * np.array([0.67003, 0.11964162, 0.034562, 0.03226982, 0.02734387, 0.02987239, 0.03762612, 0.04865424]) # cartoon
        #budget_distribution = budget * np.array([0.6494294, 0.08202955, 0.04455208, 0.04405867, 0.03689169, 0.03740072, 0.04700756, 0.05863034]) # p contribution
    elif mode == 1:
        budget_distribution = budget * np.random.dirichlet(np.ones(8),size=1)[0]
    else:
        budget_distribution = budget * np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
    # price = np.array([65, 35, 60, 35, 55, 30, 50, 100]) # ArtBench
    # data_length = [100, 50, 100, 60, 100, 50, 100, 200] # ArtBench
    # price = np.array([50, 90, 60, 110, 50, 85, 90, 170]) # Portrait
    # data_length = [50, 100, 50, 100, 60, 90, 100, 200] # Portrait
    price = np.array([55, 75, 45, 65, 30, 40, 95, 160]) # Cartoon
    data_length = [50, 80, 40, 60, 25, 30, 100, 150] # Cartoon
    determine = np.where(budget_distribution - price >= 0)[0]
    train_dir = []
    num_data = 0
    for i in determine:
        # dir = "group_" + dataholders[i]
        # dir = "portrait_" + dataholders[i]
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
                    largest_checkpoint = subfolder_path
    
    return largest_checkpoint, largest_number

def fid(round_dir):
    ###### inference ######
    model_path = last_checkpoint(round_dir)

    jsonl_file_path = './shap_bg_imagefolder/metadata.jsonl'
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
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.load_lora_weights(model_path)

    # output_dir = "./inference_output/3_1_0_0_0"
    # os.makedirs(output_dir, exist_ok=True)

    # Open the .jsonl file and read it line by line
    with open(jsonl_file_path, 'r') as file:
        i = 0
        for line in file:
            # Parse the JSON line into a dictionary
            data = json.loads(line)
            
            # Access the desired column (e.g., 'age')
            prompt = data.get('caption')  # Use .get() to avoid KeyError if 'age' is missing
            
            #prompt = "A naruto with green eyes and red legs."
            image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
            output_path = os.path.join(output_dir, f"generated_image_{i+1}.png")
            image.save(output_path)
            i += 1

    ###### fid ######
    command = [
    'python', '-m', 'pytorch_fid',
    'shap_bg_imagefolder/images', output_dir,
    '--dims', '64'
    ]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the standard output and standard error
    print("FID:\n", result.stdout)

    return result.stdout

# def train_ppo_for_pendulum(gpu_id=0):
#     agent_class = AgentPPO  # DRL algorithm name
#     env_class = PendulumEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
#     env_args = {
#         'env_name': 'Pendulum',  # Apply torque on the free end to swing a pendulum into an upright position
#         # Reward: r = -(theta + 0.1 * theta_dt + 0.001 * torque)

#         'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
#         'action_dim': 1,  # the torque applied to free end of the pendulum
#         'if_discrete': False  # continuous action space, symbols → direction, value → force
#     }
#     get_gym_env_args(env=PendulumEnv(), if_print=True)  # return env_args

#     args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
#     args.break_step = int(2e5)  # break training if 'total_step > break_step'
#     args.net_dims = [64, 32]  # the middle layer dimension of MultiLayer Perceptron
#     args.gamma = 0.97  # discount factor of future rewards
#     args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small

#     args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
#     train_agent(args)
#     if input("| Press 'y' to load actor.pth and render:") == 'y':
#         actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
#         actor_path = f"{args.cwd}/{actor_name}"
#         valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)


# def train_ppo_for_lunar_lander(gpu_id=0):
#     agent_class = AgentPPO  # DRL algorithm name
#     env_class = gym.make
#     env_args = {
#         'env_name': 'LunarLanderContinuous-v2',  # A lander learns to land on a landing pad
#         # Reward: Lander moves to the landing pad and come rest +100; lander crashes -100.
#         # Reward: Lander moves to landing pad get positive reward, move away gets negative reward.
#         # Reward: Firing the main engine -0.3,  side engine -0.03 each frame.

#         'state_dim': 8,  # coordinates xy, linear velocities xy, angle, angular velocity, two booleans
#         'action_dim': 2,  # fire main engine or side engine.
#         'if_discrete': False  # continuous action space, symbols → direction, value → force
#     }
#     get_gym_env_args(env=gym.make('LunarLanderContinuous-v2'), if_print=True)  # return env_args

#     args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
#     args.break_step = int(4e5)  # break training if 'total_step > break_step'
#     args.net_dims = [64, 32]  # the middle layer dimension of MultiLayer Perceptron
#     args.repeat_times = 32  # repeatedly update network using ReplayBuffer to keep critic's loss small
#     args.lambda_entropy = 0.04  # the lambda of the policy entropy term in PPO
#     args.gamma = 0.98

#     args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
#     train_agent(args)
#     if input("| Press 'y' to load actor.pth and render:") == 'y':
#         actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
#         actor_path = f"{args.cwd}/{actor_name}"
#         valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)


def train_ppo_custom(gpu_id=0):
    agent_class = AgentPPO
    env_class = gym.make
    env_args = {
        'env_name': 'Custom_out',  # A lander learns to land on a landing pad
        # Reward: Lander moves to the landing pad and come rest +100; lander crashes -100.
        # Reward: Lander moves to landing pad get positive reward, move away gets negative reward.
        # Reward: Firing the main engine -0.3,  side engine -0.03 each frame.

        'state_dim': 8,  # coordinates xy, linear velocities xy, angle, angular velocity, two booleans
        'action_dim': 1,  # fire main engine or side engine.
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }
    get_gym_env_args(env=gym.make('Custom_out'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(4e5)  # break training if 'total_step > break_step'
    args.net_dims = [64, 32]  # the middle layer dimension of MultiLayer Perceptron
    args.repeat_times = 32  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.lambda_entropy = 0.04  # the lambda of the policy entropy term in PPO
    args.gamma = 0.98

    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    train_agent(args)
    if input("| Press 'y' to load actor.pth and render:") == 'y':
        actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)

if __name__ == "__main__":
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    train_ppo_custom(GPU_ID)
