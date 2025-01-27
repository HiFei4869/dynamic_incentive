import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from typing import List, Tuple
from find_shap import *
import subprocess
from pathlib import Path
import math
import time

def create_budget_env(**kwargs):
    return BudgetEnv(
                     steps_per_round=kwargs.get('steps_per_round', 5),
                     budget=kwargs.get('budget', 1600),
                     get_image_quality=kwargs.get('get_image_quality', get_image_quality))

gym.register(
    id='BudgetEnv-v0',
    entry_point='__main__:create_budget_env',
    max_episode_steps=100,
)
ARY = np.ndarray

class BudgetEnv(gym.Env):
    def __init__(self, steps_per_round: int, budget: float, get_image_quality):
        self.steps_per_round = steps_per_round
        self.budget = budget
        self.get_image_quality = get_image_quality
        
        self.action_space = spaces.Discrete(17)
        self.observation_space = spaces.Box(low=0, high=budget, shape=(5,), dtype=np.float32)
        
        self.state = np.zeros(5, dtype=np.float32)
        self.current_step = 0
        self.current_budget = budget
        self.history_rewards: List[float] = []
        self.history_budgets: List[int] = []
        self.chosen_client: List[int] = []
        self.episode = 0

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.current_budget = self.budget
        self.history_rewards = []
        self.history_budgets = []
        self.chosen_client = []
        self.history_client = []
        self.episode += 1
        
        # initial state
        N_t = 0
        C_t = 0
        D_t = 0
        B_l_t = self.budget
        t_l = self.steps_per_round
        self.state = np.array([N_t, C_t, D_t, B_l_t, t_l], dtype=np.float32)
        return self.state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.current_step >= self.steps_per_round:
            raise ValueError("Exceeded the number of steps per round.")
        
        action_value = action * 100
        action_value = min(action_value, self.current_budget)

        df = pd.read_csv('shap_form.csv', index_col=0)
        df_combine = pd.read_csv('shap_cl.csv', index_col=0)
        row_name = self.current_step*500 + 10000
        # Option 1: Shapley + Copyright loss
        _, self.chosen_client, shap_sum = find_closest_combination_in_row(df_combine, row_name, action_value)
        
        if self.chosen_client is None:
            N_t = 0
            copyright_sum = 0
        else:
            N_t = len(self.chosen_client)
            copyright_sum = sum_row_values('copyright_form.csv', self.chosen_client, row_name)


        # Option 2: Shapley only, linear
        # _, self.chosen_client, shap_sum = find_closest_combination_in_row(df, row_name, action_value/2)
        # if self.chosen_client is None:
        #     N_t = 0
        #     copyright_sum = 0
        # else:
        #     N_t = len(self.chosen_client)
        #     copyright_sum = 0
        # Option 3: Copyright only, linear

        # Option 4: Shapley only, random
        # self.chosen_client, shap_sum = pick_random_values(df, row_name, action_value/2)
        # if self.chosen_client is None:
        #     N_t = 0
        #     copyright_sum = 0
        #     shap_sum = 0
        # else:
        #     N_t = len(self.chosen_client)
        #     copyright_sum = 0
        # Option 5: Copyright only, random

        C_t = copyright_sum                                        # 0: don't consider
        D_t = shap_sum
        B_l_t = self.current_budget - action_value
        t_l = self.steps_per_round - self.current_step - 1
        
        self.state = np.array([N_t, C_t, D_t, B_l_t, t_l], dtype=np.float32)
        
        # Calculate reward based on the current state
        reward = 0
        if action_value > 0 and self.chosen_client is not None:
            reward = self.get_image_quality(self.state, self.chosen_client, self.episode)
        
        self.current_budget -= action_value
        
        self.history_rewards.append(reward)
        self.history_budgets.append(action_value)
        #print(f'step: {self.current_step}, {N_t}, {C_t}, {D_t}, action: {action_value}, reward: {reward}')
        done = False
        # if N_t == 0 or action < 1:
        #     done = True
        # else:
        #     if self.current_budget <= 100 or t_l == 0:
        #         if self.current_budget <= 100 and t_l > 0:
        #             for _ in range(0, t_l):
        #                 self.history_rewards.append(0)
        #                 self.history_budgets.append(0)
        #         done = True
        #         self.current_budget = self.budget
        #         self.current_step = 0
                
        #     else:
        #         self.current_step += 1


        if N_t == 0 or action < 1 or self.current_budget <= 100 or t_l<1:
            if self.current_budget <= 100 and t_l > 0:
                for _ in range(0, t_l+1):
                    self.history_rewards.append(0)
                    self.history_budgets.append(0)
            done = True
            self.current_budget = self.budget
            self.current_step = 0
            
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

# Function to calculate image quality based on state

# [1/(fid+1e-6)]*100
def is_folder_empty(folder_path):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
    
    return not any(os.scandir(folder_path))

def get_image_quality(state, group_list, episode):
    N_t, C_t, D_t, _, t_l = state
    # print(f'current_step:{5-t_l}')
    # time.sleep(3)
    # return 0

    group_list = np.array(group_list)
    # if len(group_list) == 4 and np.array_equal(group_list, [1, 2, 7, 8]):
    #     fid = 12.4389
    # elif len(group_list) == 4 and np.array_equal(group_list, [1, 2, 3, 4]):
    #     fid = 16.51
    # elif len(group_list) == 4 and np.array_equal(group_list, [3, 4, 5, 6]):
    #     fid = 8.3141
    # elif len(group_list) == 8 and np.array_equal(group_list, [1, 2, 3, 4, 5, 6, 7, 8]):
    #     fid = 15.1426
    # elif len(group_list) == 2 and np.array_equal(group_list, [1, 2]):
    #     fid = 14.7977
    # elif len(group_list) == 1 and np.array_equal(group_list, [1]):
    #     fid = 15.9034
    # else:
    #     if len(group_list)==0:
    #         fid = 100000
    #     else:
    #         if t_l > 3:
    #             fid = 14
    #         else:
    #             fid = 12
    if len(group_list) == 4:
        if np.array_equal(group_list, [1, 2, 7, 8]):
            fid = 12.4389
        elif np.array_equal(group_list, [1, 2, 3, 4]):
            fid = 16.51
        elif np.array_equal(group_list, [3, 4, 5, 6]):
            fid = 8.3141
        else:
            fid = 15  # Default case for len(group_list) == 4
    elif len(group_list) == 8 and np.array_equal(group_list, [1, 2, 3, 4, 5, 6, 7, 8]):
        fid = 15.1426
    elif len(group_list) == 2 and np.array_equal(group_list, [1, 2]):
        fid = 14.7977
    elif len(group_list) == 1 and np.array_equal(group_list, [1]):
        fid = 15.9034
    else:
        if len(group_list) == 0:
            fid = 100000
        else:
            if t_l > 3:
                fid = 14
            else:
                fid = 12
    # file_name = "output1.txt"
    # with open(file_name, "a") as file:
    #     line = ", ".join(str(item) for item in group_list)
    #     file.write(line + "\n")

    # N_t, C_t, D_t, _, t_l = state
    # n_checkpoint = 9500
    # round = int(5 - t_l + 1)
    # current_step = int(5 - t_l)
    # train_dir = '../../generative-models/group_' + '+'.join(f'{i}' for i in group_list)

    # output_dir = f"../rl_tem/{episode}" # episode
    # os.makedirs(output_dir, exist_ok=True)
    
    # # Check if OUTPUT_DIR is empty and current_step is 0, copy the checkpoint
    # if current_step == 1: #and not os.listdir(output_dir)
    #     checkpoint_src = "../../generative-models/artbench_expressionism_200/checkpoint-9500"
    #     subprocess.run(f"cp -r {checkpoint_src} {output_dir}", shell=True)

    # checkpoint_path = f'../rl_tem/{episode}/checkpoint-{(current_step - 1) * 500 + n_checkpoint}' # episode
    # print(f'episode:{episode}, round:{round}, current_step:{current_step}, checkpointpath:{checkpoint_path}, train_dir:{train_dir}')
    # # num_train_epochs = [(9500 + round * 500) * 2] / (N_t * 100)
    # # num_train_epochs = math.ceil(num_train_epochs)  # Round up to ensure sufficient training
    # if not os.path.exists(checkpoint_path):
    #     if is_folder_empty(output_dir):
    #         os.rmdir(output_dir)
    #     return 0
    # numerator = (9500 + round * 500) * 6
    # denominator = N_t * 100
    # num_train_epochs = numerator / denominator
    # num_train_epochs = math.ceil(num_train_epochs)  # Ensure it's a scalar
    # print(f'num_train_epochs: {num_train_epochs}')
    # # os.environ["MODEL_NAME"] = "stabilityai/stable-diffusion-xl-base-1.0"
    # # os.environ["VAE_NAME"] = "madebyollin/sdxl-vae-fp16-fix"
    # os.environ["TRAIN_DIR"] = train_dir
    # os.environ["CHECKPOINT_PATH"] = checkpoint_path
    # os.environ["OUTPUT_DIR"] = output_dir

    # command = [
    #     "accelerate", "launch", "--num_processes", "6", "train_text_to_image_lora_sdxl.py",
    #     f"--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
    #     f"--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix",
    #     f"--train_data_dir={train_dir}", "--caption_column=caption",
    #     "--resolution=256", "--random_flip",
    #     "--train_batch_size=1", f"--num_train_epochs={num_train_epochs}",
    #     "--checkpointing_steps=100", "--learning_rate=1e-04",
    #     "--lr_scheduler=constant", "--lr_warmup_steps=100",
    #     "--mixed_precision=fp16", "--seed=42",
    #     f"--resume_from_checkpoint={checkpoint_path}",
    #     f"--output_dir={output_dir}",
    #     "--validation_prompt='a close up of a drawing of a man with glasses and a mustache, an ink drawing inspired by Stanisaw Tondos, reddit, shin hanga, man with glasses, portrait of sigmund freud, stanisaw'"
    # ]

    # # # Run the training command and hide the output
    # # with subprocess.Popen(' '.join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
    # #     out, err = proc.communicate()

    # # if proc.returncode != 0:
    # #     raise RuntimeError(f"Training failed with error: {err.decode()}")
    # with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
    #     for line in proc.stdout:
    #         sys.stdout.write(line)
    #         sys.stdout.flush()

    #     err = proc.stderr.read()
    #     if proc.returncode != 0:
    #         raise RuntimeError(f"Training failed with error: {err}")

    # # inference script
    # inference_command = f"python3 inference_rl.py --model_path rl_tem/{episode}/checkpoint-{current_step * 500 + n_checkpoint}" # episode
    # print('Inferencing!')
    # # Run the inference command and collect the output
    # with subprocess.Popen(inference_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
    #     out, err = proc.communicate()
    
    # fid = float(out.decode())
    
    # # Print the episode and inference output
    # print(f"episode: {episode}")  # episode
    # print(f"FID Output:\n{fid}")

    #return N_t*0.2+(5-t_l)*0.1
    return (1/(fid+1e-6))*100


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
        'env_name': 'BudgetEnv-v0',
        'state_dim': 5,
        'action_dim': 17,
        'steps_per_round': 5,
        'if_discrete': True,
        'budget': 1600,
        'get_image_quality': get_image_quality
    }

    get_gym_env_args(env=gym.make('BudgetEnv-v0'), if_print=True)  # Use correct env ID

    args = Config(agent_class, env_class, env_args)
    args.break_step = int(7e4)
    args.eval_per_step = int(1e4)
    args.net_dims = [80, 40]
    args.gamma = 0.99
    args.gpu_id = gpu_id
    args.num_envs = 1
    args.clip_grad_norm = 0.5
    args.soft_update_tau = 5e-3
    args.state_value_tau = 5e-3
    args.explore_rate = 0.5

    train_agent(args)

    if input("| Press 'y' to load actor.pth and render:") == 'y':
        actor_name = sorted([s for s in os.listdir(args.cwd) if s.endswith('.pth')])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)

if __name__ == "__main__":
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    train_dqn_for_budget_env(gpu_id=GPU_ID)

# if __name__ == "__main__":
#     env = BudgetEnv(n_rounds=5, steps_per_round=5, budget=1000, get_image_quality=get_image_quality)
#     state, _ = env.reset()
#     print("Initial State:", state)

#     for _ in range(25):
#         action = env.action_space.sample()  # Sample an action from the discrete action space
#         state, reward, done, _, info = env.step(action)
#         env.render()

#         if done:
#             state, _ = env.reset()


