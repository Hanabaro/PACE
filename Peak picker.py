import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

def detect_and_save_voltage_peaks(file_path, output_excel):
    # Load the entire file data
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the start of the actual data
    data_start_index = None
    for i, line in enumerate(lines):
        if line.strip() == '[[DATA]]':
            data_start_index = i + 1
            break

    # Use all data lines after the data start index
    data_lines = lines[data_start_index:]

    # Convert data to DataFrame
    data_str = ''.join(data_lines)
    data = pd.read_csv(pd.io.common.StringIO(data_str), delim_whitespace=True)

    # Select the 'Voltage' column and filter for numeric values
    voltage_data = pd.to_numeric(data['Voltage'], errors='coerce').dropna()

    # Estimate baseline and normalize the signal
    baseline = uniform_filter1d(voltage_data, size=100)
    normalized_voltage = voltage_data - baseline

    # Detect peaks
    peaks, _ = find_peaks(normalized_voltage, distance=50)
    peak_voltages = voltage_data.iloc[peaks]

    # Save the peaks to a DataFrame
    peak_df = pd.DataFrame({'Peak Voltage': peak_voltages})

    # Save the result to an Excel file
    peak_df.to_excel(output_excel, index=False)
    print(f"Detected peaks saved to {output_excel}")

# Example usage
file_path = 'your_file_path_here.asc'  # Enter the path to your data file here
output_excel = 'detected_peaks.xlsx'  # Name of the Excel file to save results
detect_and_save_voltage_peaks(file_path, output_excel)
