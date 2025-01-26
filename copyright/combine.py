import csv
import pandas as pd

def load_csv(file_path):
    return pd.read_csv(file_path)

def calculate_scores(perceptual_loss_csv, semantic_loss_csv, output_csv, top_n):
    # Load the CSV files
    perceptual_losses = load_csv(perceptual_loss_csv)
    clip_mse_losses = load_csv(semantic_loss_csv)

    # Ensure the CSV files have the same length
    if len(perceptual_losses) != len(clip_mse_losses):
        raise ValueError("The input CSV files must have the same length")

    # Calculate the scores
    scores = perceptual_losses['Perceptual Loss'] + 600*clip_mse_losses['Semantic Loss']

    # Create a DataFrame for the results
    results = pd.DataFrame({
        'Image Index': perceptual_losses['Image Index'],
        'Score': scores
    })
    top_scores = results.nsmallest(top_n, 'Score')
    print(top_scores)
    # Save the results to a new CSV file
    results.to_csv(output_csv, index=False)

if __name__ == "__main__":
    perceptual_loss_csv = 'per_loss_cartoon.csv'  # Path to perceptual losses CSV
    semantic_loss_csv = 'sem_loss_cartoon.csv'      # Path to semantic losses CSV
    output_csv = 'copyright_loss_cartoon.csv'               # Path to output CSV
    top_n = 5
    calculate_scores(perceptual_loss_csv, semantic_loss_csv, output_csv, top_n)
    print(f'Scores successfully calculated and saved to {output_csv}')
