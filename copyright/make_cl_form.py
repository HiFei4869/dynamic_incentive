import os
import pandas as pd
import re

# Define the range and set for n and r respectively
n_range = range(1, 10)
r_set = {10000, 10500, 11000, 11500, 12000}

# Initialize an empty DataFrame to store the results
result_df = pd.DataFrame(0, index=sorted(r_set), columns=n_range)

# Regular expression to match the file name format
pattern = re.compile(r"copyrightloss_(\d+)_(\d+)\.csv")

# Directory containing the CSV files
directory = '.'  # Current directory, change if needed

# Total sum for normalization
total_sum = 0
count = 0

# First pass: Calculate the sum of the appropriate column for each file
for filename in os.listdir(directory):
    match = pattern.match(filename)
    if match:
        n = int(match.group(1))
        r = int(match.group(2))

        if n in n_range and r in r_set:
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)

            # Determine the correct column name based on the value of n
            if n == 1:
                column_name = 'combined_loss'
            else:
                column_name = 'copyright_loss'

            if column_name in df.columns:
                # Sum the appropriate column
                loss_sum = df[column_name].sum()
            else:
                loss_sum = 0

            # Place the sum in the appropriate location in the result DataFrame
            result_df.at[r, n] = loss_sum
            total_sum += loss_sum
            count += 1

# Calculate the average sum
average_sum = total_sum / count if count > 0 else 0

# Calculate the normalization factor to make the average 100
normalization_factor = 100 / average_sum if average_sum != 0 else 1

# Apply normalization to each value in the DataFrame
result_df = result_df * normalization_factor

# Save the normalized result DataFrame to a new CSV file
output_file = 'copyright_form.csv'
result_df.to_csv(output_file)

print(f"Processing complete. The normalized results are saved in {output_file}.")


