import pandas as pd
from itertools import combinations

def closest_combination(values, target):
    closest_sum = float('-inf')
    closest_combo = None
    
    # Generate all possible combinations of different lengths
    for r in range(1, len(values) + 1):
        for combo in combinations(values, r):
            current_sum = sum(combo)
            # Check if the sum is less than the target and closer than the previous closest sum
            if current_sum < target and current_sum > closest_sum:
                closest_sum = current_sum
                closest_combo = combo
    
    return closest_combo, closest_sum

def find_closest_combination_in_row(df, row_name, target_value):
    # Select the specified row from the dataframe
    row = df.loc[row_name]
    
    # Get the values from the row (ignoring the column names)
    values = row.values
    
    # Find the closest combination of values
    closest_combo, closest_sum = closest_combination(values, target_value)
    
    # If a valid combination is found, return it along with the corresponding column names
    if closest_combo:
        columns_in_combo = [int(row.index[row == val][0]) for val in closest_combo]
        return closest_combo, columns_in_combo, closest_sum
    else:
        return None, None, None

# def pick_combination(df, row_name, target_value):
#     row = df.loc[row_name]
#     values = row.values
#     min_value = row.min()
    
#     # Find the corresponding column name
#     min_column = row.idxmin()
    
#     return int(min_column), min_value

import pandas as pd
import numpy as np

def pick_random_values(df, row_name, target_value):
    # Get the row by its name
    row = df.loc[row_name]
    
    # Calculate n as the floor of target_value divided by the maximum value in the row
    max_value = row.max()
    n = int(target_value // max_value)
    
    # Ensure n is at least 0 and does not exceed the number of columns
    n = min(max(n, 0), len(row))
    
    # Initialize lists for selected values and column indices
    selected_values = []
    column_indices = []

    if n > 0:
        # Randomly select n different values from the row
        sampled_row = row.sample(n=n)
        selected_values = sampled_row.values
        
        # Get the column names (as integers) for the selected values
        column_indices = sampled_row.index.to_series().astype(int).values

    return column_indices, sum(selected_values)



import argparse

def sum_row_values(csv_file, columns, row_name):
    """
    Sums the values in the specified columns for a given row in a CSV file.

    :param csv_file: Path to the CSV file.
    :param columns: List of column names (integers) to sum.
    :param row_name: The name of the row to sum values for.
    :return: The sum of the specified columns' values for the given row.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file, index_col=0)

    # Print the DataFrame and its index for debugging
    # print("DataFrame loaded:")
    # print(df)
    # print("DataFrame index:")
    # print(df.index)

    # Convert row_name to integer if it is a string representation of an integer
    try:
        row_name_int = int(row_name)
    except ValueError:
        row_name_int = row_name  # If conversion fails, keep row_name as is

    # Check if the row_name exists in the DataFrame index
    if row_name_int not in df.index:
        raise ValueError(f"Row '{row_name}' not found in the CSV file.")

    # Check if the columns exist in the DataFrame
    missing_columns = [str(col) for col in columns if str(col) not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the CSV file.")

    # Sum the values in the specified columns for the given row
    row_sum = df.loc[row_name_int, [str(col) for col in columns]].sum()

    return row_sum

# if __name__ == "__main__":
#     # Set up argparse to take user input
#     parser = argparse.ArgumentParser(description="Sum values in specified columns for a given row in a CSV file.")
#     parser.add_argument('csv_file', type=str, help="Path to the CSV file.")
#     parser.add_argument('columns', type=int, nargs='+', help="List of column names (integers) to sum.")
#     parser.add_argument('row_name', type=str, help="The name of the row to sum values for.")

#     args = parser.parse_args()

#     # Call the function with provided arguments
#     try:
#         result = sum_row_values(args.csv_file, args.columns, args.row_name)
#         print(f"The sum of the values in columns {args.columns} for row '{args.row_name}' is {result}.")
#     except ValueError as e:
#         print(f"Error: {e}")


# Example DataFrame
# data = {
#     'col1': [10, 15, 18, 22, 25],
#     'col2': [20, 25, 28, 32, 35],
#     'col3': [30, 35, 38, 42, 45],
#     'col4': [40, 45, 48, 52, 55]
# }
#df = pd.DataFrame(data, index=[10000, 10500, 11000, 11500, 12000])
df = pd.read_csv('shap_form.csv', index_col=0)
# Example usage
# row_name = 11000
# target_value = 120

# result = find_closest_combination_in_row(df, row_name, target_value)
# print(f"Closest combination: {result[0]}")
# print(f"Columns involved: {result[1]}")
# print(f"Sum of combination: {result[2]}")
