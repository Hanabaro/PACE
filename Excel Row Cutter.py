import os
import pandas as pd

def generate_unique_filename(base_output, index):
    base, ext = os.path.splitext(base_output)
    return f"{base}_part{index}{ext}"

def generate_unique_filename_if_exists(filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}({counter}){ext}"
        counter += 1
    return new_filename

def split_excel_file(file_path, output_excel, row_count):
    # Load the entire Excel file
    df = pd.read_excel(file_path)

    # Get the header (first row)
    header = df.columns.tolist()  # The header row (column names)

    # Calculate the number of files needed
    num_files = (len(df) - 1) // row_count + 1

    for i in range(num_files):
        # Calculate start and end indices for this chunk
        start = i * row_count  # Start from the correct index
        end = start + row_count

        # Extract the chunk including the header
        chunk = df.iloc[start:end]

        # Generate a unique filename with a part number
        output_filename = generate_unique_filename(output_excel, i + 1)

        # Ensure the filename does not overwrite an existing file
        output_filename = generate_unique_filename_if_exists(output_filename)

        # Save the chunk to a new Excel file, including the header
        chunk.to_excel(output_filename, index=False, header=header)
        print(f"Saved {output_filename}")

# Example usage
file_path = '/Users/hanabaro/Downloads/data/original.xlsx'  # Path to your input Excel file
output_excel = '/Users/hanabaro/Downloads/data/split.xlsx'  # Base name for output files
row_count = 150000  # Number of rows per split file

split_excel_file(file_path, output_excel, row_count)
