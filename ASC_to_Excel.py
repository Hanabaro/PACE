import os
import pandas as pd

def generate_unique_filename(output_excel):
    base, ext = os.path.splitext(output_excel)
    counter = 1
    new_filename = output_excel
    while os.path.exists(new_filename):
        new_filename = f"{base}({counter}){ext}"
        counter += 1
    return new_filename

def convert_asc_to_excel(file_path, output_excel):
    # Load the entire .asc file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Convert the lines to a DataFrame
    data = pd.DataFrame(lines, columns=['Data'])

    # Generate a unique file name
    output_excel = generate_unique_filename(output_excel)

    # Save the DataFrame to an Excel file
    data.to_excel(output_excel, index=False)
    print(f"Converted file saved to {output_excel}")

# Example usage
file_path = '/Users/hanabaro/Library/CloudStorage/OneDrive-MacquarieUniversity/00_Projects/2024_PACE/Dataset/DATASET/HJIMP11_1.asc'  # Enter the path to your data file here
output_excel = '/Users/hanabaro/Downloads/data/test.xlsx'  # Name of the Excel file to save results
convert_asc_to_excel(file_path, output_excel)
