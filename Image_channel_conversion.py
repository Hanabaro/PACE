# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:30:54 2024

@author: yousf
"""

import cv2
import os

def convert_images_to_2d_grayscale(source_folder, destination_folder):
    """
    Convert all images in the source_folder to 2D grayscale format and save them to the destination_folder.
    """
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Loop over all files in the source folder
    for filename in os.listdir(source_folder):
        # Construct full file path
        file_path = os.path.join(source_folder, filename)
        
        # Read the image
        img = cv2.imread(file_path)
        
        if img is None:
            print(f"Error: Couldn't read the image {filename}")
            continue
        
        # Convert the image to grayscale (1 channel, 2D)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Save the 2D grayscale image to the destination folder
        save_path = os.path.join(destination_folder, filename)
        cv2.imwrite(save_path, gray_img)
        
        print(f"Converted and saved {filename} as a 2D grayscale image.")

# Example usage
source_folder = 'C:/Users/yousf/OneDrive/Desktop/solar_model/images/'  # Replace with the path to your source folder
destination_folder = 'C:/Users/yousf/OneDrive/Desktop/solar_model/images/'  # Replace with the path to your destination folder
convert_images_to_2d_grayscale(source_folder, destination_folder)
