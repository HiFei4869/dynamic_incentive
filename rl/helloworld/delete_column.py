import pandas as pd

def delete_column_from_csv(input_csv, output_csv, column_name_to_delete):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # Check if the column name to delete exists in the DataFrame
    if column_name_to_delete in df.columns:
        # Drop the column
        df = df.drop(columns=[column_name_to_delete])
        print(f"Column '{column_name_to_delete}' deleted.")
    else:
        print(f"Column '{column_name_to_delete}' does not exist in the CSV file.")
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv, index=False, header=False)
    print(f"Modified CSV saved as '{output_csv}'.")

# Usage
input_csv = 'shap_form.csv'  # Replace with your input CSV file path
output_csv = 'shap_form_d.csv'  # Replace with your desired output CSV file path
column_name_to_delete = '9'  # Column name to delete

delete_column_from_csv(input_csv, output_csv, column_name_to_delete)
