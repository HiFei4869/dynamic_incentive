import gym
from gym import spaces
import numpy as np

class BudgetEnv(gym.Env):
    def __init__(self, total_budget, num_rounds, quality_function):
        super(BudgetEnv, self).__init__()
        self.total_budget = total_budget
        self.num_rounds = num_rounds
        self.quality_function = quality_function  # Function to compute the quality of the generation
        self.current_round = 0
        self.current_budget = total_budget
        self.budget_distribution = np.zeros(num_rounds)
        
        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=self.total_budget, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=self.total_budget, shape=(num_rounds,), dtype=np.float32)

    def step(self, action):
        # Apply budget allocation action
        budget_allocation = action[0]
        self.budget_distribution[self.current_round] = budget_allocation
        self.current_budget -= budget_allocation
        self.current_round += 1

        # Check if the last round is reached
        done = self.current_round == self.num_rounds
        
        # Calculate reward
        if done:
            reward = self.quality_function(self.budget_distribution)
        else:
            reward = 0
        
        return self._get_state(), reward, done, {}

    def reset(self):
        self.current_round = 0
        self.current_budget = self.total_budget
        self.budget_distribution = np.zeros(self.num_rounds)
        return self._get_state()

    def render(self, mode='human'):
        print(f'Round: {self.current_round}, Budget Distribution: {self.budget_distribution}, Remaining Budget: {self.current_budget}')

    def _get_state(self):
        return self.budget_distribution

def quality_function(budget_distribution):
    # Placeholder for the actual quality function
    # Replace this with the actual computation of the quality of the generated image
    return np.random.random()

from elegantrl.train.run import train_agent
# from elegantrl.run import train_and_evaluate

from elegantrl.agents.AgentDQN import AgentDQN
from elegantrl.train.config import Config
from erl_config import Config, get_gym_env_args

# Initialize the environment
total_budget = 1000
num_rounds = 10
env = BudgetEnv(total_budget, num_rounds, quality_function)

# # Initialize the DQN agent
# agent = AgentDQN()
# config = Config(agent, env)
# config.target_step = 200  # Number of steps before target network update
# config.max_step = 5000  # Maximum number of steps for training
# config.if_remove = True  # Remove the previous log directory if it exists


# # Train the agent
# train_agent(config)



def train_budget_distribution(gpu_id=0):
    agent_class = AgentDQN  # DRL algorithm
    env_class = gym.make
    env_args = {
        'env_name': 'BudgetEnv',  # A pole is attached by an un-actuated joint to a cart.
        # Reward: keep the pole upright, a reward of `+1` for every step taken

        'state_dim': 5,  # (CartPosition, CartVelocity, PoleAngle, PoleAngleVelocity)
        'action_dim': 1,  # (Push cart to the left, Push cart to the right)
        'if_discrete': True,  # discrete action space
    }
    get_gym_env_args(env=gym.make('BudgetEnv'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e3)  # break training if 'total_step > break_step'
    args.net_dims = [64, 32]  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.95  # discount factor of future rewards

    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    train_agent(args)
    if input("| Press 'y' to load actor.pth and render:") == 'y':
        actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)

train_budget_distribution()
