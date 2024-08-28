import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def detect_and_save_voltage_peaks_balanced(file_path, output_excel, degree=2, z_threshold=3.0):
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
    voltage_data = pd.to_numeric(data['Voltage'], errors='coerce').dropna().reset_index(drop=True)

    # Create a trendline using a simpler polynomial regression (degree 2)
    X = np.arange(len(voltage_data)).reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, voltage_data)
    trendline = model.predict(X_poly)

    # Calculate the residuals (difference from the trendline)
    residuals = voltage_data - trendline

    # Detect outliers based on a wider Z-score threshold
    z_scores = (residuals - residuals.mean()) / residuals.std()
    outliers = np.abs(z_scores) > z_threshold

    # Filter out the outliers
    filtered_voltage_data = voltage_data[~outliers]

    # Estimate baseline and normalize the filtered signal
    baseline = uniform_filter1d(filtered_voltage_data, size=100)
    normalized_voltage = filtered_voltage_data - baseline

    # Detect peaks
    peaks, _ = find_peaks(normalized_voltage, distance=50)
    peak_voltages = filtered_voltage_data.iloc[peaks]

    # Add row indices (1-based index)
    peak_df = pd.DataFrame({
        'Row Index': peaks + 1,  # +1 to match typical row numbering
        'Peak Voltage': peak_voltages.values
    })

    # Save the filtered peaks to an Excel file
    peak_df.to_excel(output_excel, index=False)
    print(f"Detected and filtered peaks saved to {output_excel}")

# Example usage
file_path = '/mnt/data/HJIMP.asc'  # Enter the path to your data file here
output_excel = '/mnt/data/detected_peaks_balanced.xlsx'  # Name of the Excel file to save results
detect_and_save_voltage_peaks_balanced(file_path, output_excel)
