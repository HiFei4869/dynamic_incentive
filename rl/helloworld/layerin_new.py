import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from typing import List, Tuple
import itertools
import torch
import subprocess
import os
from copyright_fun import get_copyright

def create_budget_env(**kwargs):
    return BudgetEnv(
        steps_per_round=kwargs.get('steps_per_round', 5),
        budget=kwargs.get('budget', 1000),
        contribution=kwargs.get('contribution', [0.125] * 8),  # default: linear
        copyright=kwargs.get('copyright', [0.125] * 8)
    )



try:
    gym.spec('BudgetEnv-v0')
    print("Environment 'BudgetEnv-v0' is already registered.")
except gym.error.Error:
    gym.register(
        id='BudgetEnv-v0',
        entry_point='layerin_new:create_budget_env',  # Replace '__main__' with the module name
        max_episode_steps=100,
    )
ARY = np.ndarray

# def generate_valid_actions():
#     # Elements must be from the set {0, 0.1, 0.2, ..., 1.0} and sum to 1.
#     possible_values = [i * 0.1 for i in range(11)]  # [0.0, 0.1, ..., 1.0]
#     valid_actions = []

#     # Generate all combinations of length 8 from possible_values
#     for action in itertools.product(possible_values, repeat=8):
#         if np.isclose(sum(action), 1.0):  # Only keep those where the sum is exactly 1
#             valid_actions.append(np.array(action))

#     return np.array(valid_actions)
def generate_valid_actions(num_actions=10000):
    actions = []
    for _ in range(num_actions):
        action = np.random.dirichlet(np.ones(8))  # Generates a vector whose elements sum to 1
        action = np.round(action * 10) / 10.0  # Rounding to the nearest 0.1
        if np.isclose(sum(action), 1.0):  # Only keep those where the sum is exactly 1
            actions.append(action)
    
    return np.array(actions)

class BudgetEnv(gym.Env):
    def __init__(self, steps_per_round: int, budget: float, contribution: List[float], copyright: List[float]):
        self.steps_per_round = steps_per_round
        self.budget = budget
        
        # self.action_space = spaces.Discrete(11)
        # self.action_space = spaces.MultiDiscrete([11] * 8)  # Each value can be 0 to 10, representing 0.0 to 1.0
        # self.valid_actions = np.load('valid_combinations.npy')
        self.valid_actions = np.random.dirichlet(np.ones(8),size=1)[0]

        # Define the action space to be the index of valid actions
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=budget, shape=(8,), dtype=np.float32)
        
        self.state = np.zeros(8, dtype=np.float32)
        #self.reputation = np.array([0.56,0.12,0.49,0.137,0.36,0,0.319,1], dtype=np.float32)    # artbench contribution
        #self.reputation = np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125], dtype=np.float32)  # linear
        #self.copyright = np.array([0.45,0,0.45,0.09,0.45,0,0.27,1], dtype=np.float32)
        # self.copyright = np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125], dtype=np.float32)  # linear
        #self.copyright = np.array([0.167,0.0838,0.170,0.103,0.128,0.065,0.128,0.155], dtype=np.float32)  # artbench copyright
        #self.reputation = np.array([0.113, 0.202, 0.007, 0.126, 0.132, 0.186, 0, 0.168],dtype=np.float32) # portrait
        #self.copyright = np.array([0.065,0.129,0.063,0.127,0.078,0.117,0.152,0.269], dtype=np.float32) # portrait
        #self.copyright = np.array([0.089863,0.15378,0.068386,0.112402,0.038878,0.047288,0.169317,0.320086],dtype=np.float32) # cartoon
        # self.reputation = np.array([0.109801,0.181816,0.087079,0.132981,0.048913,0.062523,0.403145,0.596855],dtype=np.float32) # cartoon
        self.reputation = np.array(contribution, dtype=np.float32)
        self.copyright = np.array(copyright, dtype=np.float32)
        # get_contribution1()

        # self.reputation = np.array(contribution, dtype=np.float32)

        self.alpha = 0.5
        self.beta = 0.5
        self.current_step = 0
        self.current_budget = budget
        self.history_rewards: List[float] = []
        self.history_budgets = []
        # self.last_action = np.array([],dtype=np.float32)
        self.best_action = np.zeros(8, dtype=np.float32)
        self.best_reward = float('-inf')

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.current_budget = self.budget
        # self.history_rewards = []
        # self.history_budgets = []
        # self.last_action = np.array([],dtype=np.float32)

        self.state = np.zeros(8, dtype=np.float32)
        return self.state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # if self.current_step >= self.steps_per_round:
        #     raise ValueError("Exceeded the number of steps per round.")
        
        # action = action / 10.0
        # action = action / np.sum(action)
        # action_value = action * self.budget

        self.state = action.astype(np.float32)
        
        # self.state = np.array([N_t, C_t, D_t, B_l_t, t_l], dtype=np.float32)
             
        reward = np.sum((self.alpha * self.reputation - self.beta * self.copyright) * self.state)
        # reward = torch.tensor(reward, dtype=torch.float32, device='cuda')
        reward = float(reward)
        self.history_rewards.append(reward)
        self.history_budgets.append(self.state)
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_action = self.state.copy()
        
        # print(f'state: {self.last_action}')
        # print(f'reward: {reward}')
        self.current_step += 1
        done = False
        #print(f'step: {self.current_step}, {N_t}, {C_t}, {D_t}, action: {action_value}, reward: {reward}')
        
        return self.state, reward, done, False, {}

    def render(self, mode='human'):
        if mode == 'human':
            print("Reward\tBudget Distribution")
            for i in range(len(self.history_rewards)):
                reward = self.history_rewards[i]
                budget_distribution = self.history_budgets[i]
                print(f"{reward:.2f}\t{budget_distribution}")
        elif mode == 'ansi':
            print("Rendering in ANSI mode")
            # Implement any ASCII rendering logic if needed
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def get_last_action(self) -> np.ndarray:
        return self.last_action

def get_contribution1():
    print("call get_contribution()")
    return [0.125]*8
def get_contribution(dataset):
    print("call get_contribution()")
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

    # Iterate over dataset names
    trak_results = []
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
        
        trak_results.append(float(generated_value))

    # for dataset, value in results.items():
    #     print(f"Dataset: {dataset}, Generated Value: {value}")
    print(trak_results)
    contribution = [x/sum(trak_results) for x in trak_results]
    print(contribution)
    return contribution

import os
import sys
import gymnasium as gym
from erl_config import Config, get_gym_env_args
from erl_agent_new import AgentDQN
from erl_run_new import train_agent, valid_agent
# from erl_run import train_agent, valid_agent

def train_dqn_for_budget_env_in(gpu_id=0):
    agent_class = AgentDQN
    env_class = gym.make

    dataset = "c"
    contribution = get_contribution(dataset)
    copyright1 = get_copyright(dataset)
    env = gym.make('BudgetEnv-v0', contribution=contribution, copyright=copyright1)
    print(f"Created environment: {env}")

    env_args = {
        'env_name': 'BudgetEnv-v0',
        'state_dim': 8,
        'action_dim': 8,  # Number of valid actions
        'steps_per_round': 5,
        'if_discrete': True,
        'budget': 1000
    }

    # Ensure the environment is passed into get_gym_env_args
    get_gym_env_args(env=env, if_print=False)

    args = Config(agent_class, env_class, env_args)
    args.break_step = int(8e4)
    args.net_dims = [80, 40]
    args.gamma = 0.99
    args.gpu_id = gpu_id
    args.num_envs = 1
    args.clip_grad_norm = 0.5
    args.soft_update_tau = 5e-3
    args.state_value_tau = 5e-3
    args.explore_rate = 0.4

    # Train the agent using the created environment
    # train_agent(args)
    env = train_agent(args)


    # Uncomment this if you want to validate the agent
    # if input("| Press 'y' to load actor.pth and render:") == 'y':
    #     actor_name = sorted([s for s in os.listdir(args.cwd) if s.endswith('.pth')])[-1]
    #     actor_path = f"{args.cwd}/{actor_name}"
    #     valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)

    print("Last action taken:", env.unwrapped.best_action)
    last_action = env.unwrapped.best_action
    return last_action
    # length1 = len(env.unwrapped.history_rewards)
    # print(length1)

if __name__ == "__main__":
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    a = train_dqn_for_budget_env_in(gpu_id=GPU_ID)

# def train_dqn_for_budget_env(gpu_id=0):
#     agent_class = AgentDQN
#     env_class = gym.make
#     # env = gym.make('BudgetEnv-v0')
#     #print(f"Created environment: {env}")
#     env_args = {
#         'env_name': 'BudgetEnv-v0',
#         'state_dim': 8,
#         'action_dim': 8, #19448
#         'steps_per_round': 5,
#         'if_discrete': True,
#         'budget': 1000,
#         'get_image_quality': get_image_quality
#     }
#     '''
#     action: an ndarray, sum of each items add up to one
#     budget: action_out[n]*B
#     '''
#     get_gym_env_args(env=gym.make('BudgetEnv-v0'), if_print=True)  # Use correct env ID
#     # print("Testing the environment with a few steps.")
#     # for _ in range(5):
#     #     action = env.action_space.sample()
#     #     state, reward, done, _, info = env.step(action)
#     #     print(f"Step done: state={state}, reward={reward}, done={done}")

#     args = Config(agent_class, env_class, env_args)
#     args.break_step = int(1e4)
#     args.net_dims = [80, 40]
#     args.gamma = 0.95
#     args.gpu_id = gpu_id
#     args.num_envs = 1
#     args.clip_grad_norm = 0.5
#     args.soft_update_tau = 5e-3
#     args.state_value_tau = 5e-3
#     args.explore_rate = 0.5

#     train_agent(args)
#     print(env.get_last_action())
#     # if input("| Press 'y' to load actor.pth and render:") == 'y':
#     #     actor_name = sorted([s for s in os.listdir(args.cwd) if s.endswith('.pth')])[-1]
#     #     actor_path = f"{args.cwd}/{actor_name}"
#     #     valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)

# if __name__ == "__main__":
#     GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
#     train_dqn_for_budget_env(gpu_id=GPU_ID)

# if __name__ == "__main__":
#     env = BudgetEnv(steps_per_round=5, budget=1000, get_image_quality=get_image_quality)
#     state, _ = env.reset()
#     print("Initial State:", state)

#     for _ in range(25):
#         action = env.action_space.sample()  # Sample an action from the discrete action space
#         state, reward, done, _, info = env.step(action)
#         env.render()

#         if done:
#             state, _ = env.reset()


# import numpy as np
# import itertools

# # Define the set of possible values (0, 0.1, ..., 1.0)
# values = np.arange(0, 1.1, 0.1)

# # Generate all possible combinations of 8 elements from the set {0, 0.1, ..., 1.0}
# combinations = itertools.product(values, repeat=8)

# # Filter combinations whose sum is approximately 1 (to avoid floating-point precision issues)
# valid_combinations = [comb for comb in combinations if np.isclose(sum(comb), 1.0)]

# # Convert to a numpy array
# valid_combinations = np.array(valid_combinations)

# # Save to a .npy file
# np.save('valid_combinations.npy', valid_combinations)




