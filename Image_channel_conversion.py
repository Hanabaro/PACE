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

########################################To check image Channel######################################
import cv2

def check_image_channels(image_path):
    """
    Check the number of channels in an image and return the result.
    """
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Image could not be loaded. Check the path.")
        return
    
    # Check the number of channels
    if len(image.shape) == 2:
        print(f"The image is grayscale with 1 channel.")
    elif len(image.shape) == 3:
        print(f"The image is a color image with {image.shape[2]} channels.")
    else:
        print("Error: Unsupported image format.")

# Example usage
image_path = 'C:/Users/yousf/OneDrive/Desktop/solar_model/images/_20240902_151621.jpg' # Replace with your image path
check_image_channels(image_path)

