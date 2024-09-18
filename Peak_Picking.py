# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:30:00 2024

@author: yousf

"""

import pandas as pd
from scipy.signal import find_peaks

# Load the data from Excel file
file_path = 'C:/Users/yousf/OneDrive/Desktop/Day 1.xlsx'  # Adjust this to your file path
df = pd.read_excel(file_path)  # Adjust sheet_name if necessary

# Assuming the data has a column called 'Values' where the peak data is stored
data = df['Voltage (V)']  

# Find peak indices using SciPy's find_peaks function
peaks, _ = find_peaks(data)

# Get the peak values
peak_values = data[peaks]

# Filter the peak values: greater than 1.06 and less than 1.13
filtered_peaks = peak_values[(peak_values > 0) & (peak_values < 2)]

# Output the filtered peaks
print(filtered_peaks)
