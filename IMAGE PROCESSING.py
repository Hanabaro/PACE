import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image.
image = cv2.imread('/Users/hanabaro/Library/CloudStorage/OneDrive-MacquarieUniversity/00_Projects/2024_PACE/Dataset/GOODCELLS/1.jpg')

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Define the structuring element
kernel = np.ones((5, 5), np.uint8)

# Perform erosion
eroded_image = cv2.erode(thresholded_image, kernel, iterations=1)

# Perform dilation
dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

# Perform Canny edge detection
edges = cv2.Canny(gray_image, 100, 200)

# Perform connected component analysis
num_labels, labels_im = cv2.connectedComponents(dilated_image)

# Convert labels image to color for visualization
labels_im_color = cv2.cvtColor(labels_im.astype(np.uint8) * 50, cv2.COLOR_GRAY2BGR)

# Display images using matplotlib
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(2, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(2, 3, 3)
plt.imshow(thresholded_image, cmap='gray')
plt.title('Thresholded Image')

plt.subplot(2, 3, 4)
plt.imshow(eroded_image, cmap='gray')
plt.title('Eroded Image')

plt.subplot(2, 3, 5)
plt.imshow(dilated_image, cmap='gray')
plt.title('Dilated Image')

plt.subplot(2, 3, 6)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')

plt.tight_layout()
plt.show()
