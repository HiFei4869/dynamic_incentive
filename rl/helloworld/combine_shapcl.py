import pandas as pd
import argparse

def add_csv_files(file1, file2, output_file):
    # Load the two CSV files
    df1 = pd.read_csv(file1, index_col=0)
    df2 = pd.read_csv(file2, index_col=0)

    # Ensure that both DataFrames have the same structure
    if df1.shape != df2.shape or not all(df1.columns == df2.columns) or not all(df1.index == df2.index):
        raise ValueError("The two CSV files do not have the same structure.")

    # Add the values in the same location from both DataFrames
    result_df = df1 + df2

    # Set all values in column '9' to zero
    if '9' in result_df.columns:
        result_df['9'] = 0

    # Save the resulting DataFrame to a new CSV file
    result_df.to_csv(output_file)

    print(f"Files {file1} and {file2} have been added up. The result is saved in {output_file}. Column '9' has been set to zero.")

if __name__ == "__main__":
    # Set up argparse to take user input for CSV file paths
    parser = argparse.ArgumentParser(description="Add values from two CSV files, set column '9' to zero, and save the result.")
    parser.add_argument('file1', type=str, help="Path to the first CSV file.")
    parser.add_argument('file2', type=str, help="Path to the second CSV file.")
    parser.add_argument('output_file', type=str, help="Path to save the output CSV file.")

    args = parser.parse_args()

    # Call the function to add the CSV files
    add_csv_files(args.file1, args.file2, args.output_file)


