import argparse
import pandas as pd

def preprocess_csv(per_loss_file, sem_loss_file, output_file):
    # Load the CSV files
    per_loss_df = pd.read_csv(per_loss_file)
    sem_loss_df = pd.read_csv(sem_loss_file)
    
    # Check if the column names are correct
    if 'per_loss' not in per_loss_df.columns or 'sem_loss' not in sem_loss_df.columns:
        raise ValueError("Column names are not as expected. Ensure that 'per_loss' and 'sem_loss' are present.")

    # Multiply sem_loss by 300
    sem_loss_df['sem_loss'] = sem_loss_df['sem_loss'] * 500

    # Add per_loss and sem_loss
    combined_df = per_loss_df['per_loss'] + sem_loss_df['sem_loss']
    
    # Save the result to a new CSV file
    combined_df.to_csv(output_file, index=False, header=['combined_loss'])

if __name__ == "__main__":
    # Set up argparse to take user input for CSV file paths
    parser = argparse.ArgumentParser(description="Preprocess two CSV files and save the combined result.")
    parser.add_argument('per_loss_file', type=str, help="Path to the per_loss CSV file.")
    parser.add_argument('sem_loss_file', type=str, help="Path to the sem_loss CSV file.")
    parser.add_argument('output_file', type=str, help="Path to save the output CSV file.")

    args = parser.parse_args()

    # Call the preprocess function with the provided arguments
    preprocess_csv(args.per_loss_file, args.sem_loss_file, args.output_file)

