import numpy as np

# Load the valid combinations from the file
valid_combinations = np.load('valid_combinations.npy')

# Example input lists (replace with your actual data)
self_copyright = np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125], dtype=np.float32)
self_reputation = np.array([0.109801,0.181816,0.087079,0.132981,0.048913,0.062523,0.403145,0.596855],dtype=np.float32)

# Function to calculate the reward
def calculate_reward(output):
    return np.sum((0.5 * self_reputation - 0.5 * self_copyright) * output)

# Initialize variables to store the best output and its reward
best_output = None
best_reward = -np.inf

print(len(valid_combinations))
# Iterate over all valid combinations to find the one with the highest reward
for combination in valid_combinations:
    reward = calculate_reward(combination)
    if reward > best_reward:
        best_reward = reward
        best_output = combination

# Print the best output and its corresponding reward
print("Best Output:", best_output)
print("Best Reward:", best_reward)
print(len(valid_combinations))

# test_case = [0.1, 0.1, 0, 0, 0, 0.2, 0.3, 0.3]
# print(calculate_reward(test_case))