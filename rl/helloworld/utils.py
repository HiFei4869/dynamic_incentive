# budget_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from typing import List, Tuple
import itertools
import torch

def create_budget_env(**kwargs):
    return BudgetEnv(
                     steps_per_round=kwargs.get('steps_per_round', 5),
                     budget=kwargs.get('budget', 1000),
                     get_image_quality=kwargs.get('get_image_quality', get_image_quality))

gym.register(
    id='BudgetEnv-v0',
    entry_point='budget_env:create_budget_env',
    max_episode_steps=100,
)

ARY = np.ndarray

def generate_valid_actions(num_actions=10000):
    actions = []
    for _ in range(num_actions):
        action = np.random.dirichlet(np.ones(8))  # Generates a vector whose elements sum to 1
        action = np.round(action * 10) / 10.0  # Rounding to the nearest 0.1
        if np.isclose(sum(action), 1.0):  # Only keep those where the sum is exactly 1
            actions.append(action)
    
    return np.array(actions)

class BudgetEnv(gym.Env):
    def __init__(self, steps_per_round: int, budget: float, get_image_quality):
        self.steps_per_round = steps_per_round
        self.budget = budget
        self.get_image_quality = get_image_quality
        
        self.valid_actions = np.random.dirichlet(np.ones(8),size=1)[0]

        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=budget, shape=(8,), dtype=np.float32)
        
        self.state = np.zeros(8, dtype=np.float32)
        self.copyright = np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125], dtype=np.float32)  # linear
        self.reputation = np.array([0.109801,0.181816,0.087079,0.132981,0.048913,0.062523,0.403145,0.596855],dtype=np.float32)

        self.alpha = 0.5
        self.beta = 0.5
        self.current_step = 0
        self.current_budget = budget
        self.history_rewards: List[float] = []
        self.history_budgets = []
        self.last_action = []

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.current_budget = self.budget
        self.last_action = []

        self.state = np.zeros(8, dtype=np.float32)
        return self.state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.state = action.astype(np.float32)
        
        reward = np.sum((self.alpha * self.reputation - self.beta * self.copyright) * self.state)
        reward = float(reward)
        self.history_rewards.append(reward)
        self.history_budgets.append(self.state)
        self.last_action = self.state.copy()
        self.current_step += 1
        done = False
        
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
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def get_last_action(self) -> np.ndarray:
        return self.last_action

def get_image_quality(state: np.ndarray) -> float:
    _, C_t, D_t, _, t_l = state
    output = [6.289, 7.921, 8.571, 8.245, 6.95]
    return output[int(5-t_l-1)] + C_t*0.1

def get_reward(state: np.ndarray) -> float:
    a = 0.5
    b = 0.5
    reputation = [0.1]*5      # draft
    copyrightloss = [0.2]*5   # draft
    reward = a*sum(reputation)-b*sum(copyrightloss)
    return reward