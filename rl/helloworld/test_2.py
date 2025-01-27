# from layerin_new import train_dqn_for_budget_env_in

# a = train_dqn_for_budget_env_in()
# print(f'In test_2: {a}')

# import os
# output_dir = "./output_fid"
# os.makedirs(output_dir, exist_ok=True)
# output_file = os.path.join(output_dir, "inner_rl.txt")

def train_dqn_for_budget_env_in():
    return [0.1]*8


import os
output_dir = "./output_fid"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"inner_rl.txt")

# Check if the file exists
if os.path.exists(output_file):
    # If the file exists, read the first line and convert it into a list
    with open(output_file, 'r') as file:
        first_line = file.readline().strip()
        distribution = list(map(float, first_line.split()))  # Convert string to list of floats

else:
    # If the file doesn't exist, call the function and get the result
    distribution = train_dqn_for_budget_env_in()

    # Write the result (list) to the file in a single line
    with open(output_file, 'w') as file:
        file.write(' '.join(map(str, distribution)) + '\n')
