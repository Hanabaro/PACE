import numpy as np
import pandas as pd
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

# Assume images are stored in a directory
image_dir = 'C:/Users/yousf/OneDrive/Desktop/Dataset'
image_size = (224, 224)  # Adjust based on your images

# Load and preprocess images
def load_images(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, filename))
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
    return np.array(images)

image_data = load_images(image_dir)

# Normalization
image_data = image_data / 255.0
# Example: Extracting mean pixel intensity to represent a time series
mean_intensity = np.mean(image_data, axis=(1, 2, 3))

# STL Decomposition
from statsmodels.tsa.seasonal import STL

# Sample rate; adjust as needed
period = 30
stl = STL(mean_intensity, period=period)
result = stl.fit()
trend = result.trend
seasonal = result.seasonal
residual = result.resid
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Assuming mean_intensity is the time series data extracted from images
# For demonstration, create a sample time series if you don't have actual data
np.random.seed(0)
n_periods = 365  # Example: one year of daily data
mean_intensity = 10 + np.sin(2 * np.pi * np.arange(n_periods) / 30) + np.random.normal(0, 0.5, n_periods)

# STL Decomposition
period = 30  # Assuming a monthly seasonality for example
stl = STL(mean_intensity, period=period)
result = stl.fit()

# Extract components
trend = result.trend
seasonal = result.seasonal
residual = result.resid

# Plotting
plt.figure(figsize=(10, 8))

# Original Time Series
plt.subplot(4, 1, 1)
plt.plot(mean_intensity, label='Original Time Series')
plt.legend(loc='upper left')
plt.title('STL Decomposition of Time Series')

# Trend
plt.subplot(4, 1, 2)
plt.plot(trend, label='Trend', color='orange')
plt.legend(loc='upper left')

# Seasonal
plt.subplot(4, 1, 3)
plt.plot(seasonal, label='Seasonal', color='green')
plt.legend(loc='upper left')

# Residual
plt.subplot(4, 1, 4)
plt.plot(residual, label='Residual', color='red')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Sample image data for demonstration (replace with your actual image data)
# Here, assuming `image_data` is a numpy array of shape (n_samples, height, width)
n_samples = 100  # Example number of images
height, width = 28, 28  # Example dimensions of each image
np.random.seed(0)
image_data = np.random.rand(n_samples, height, width)  # Replace with actual image data

# Flatten image data for PCA
flat_image_data = image_data.reshape(image_data.shape[0], -1)

# Perform PCA
n_components = 50  # Adjust the number of components as needed
pca = PCA(n_components=n_components)
reduced_data = pca.fit_transform(flat_image_data)

# Plotting explained variance
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of Principal Components')
plt.grid(True)
plt.show()

# Plotting data projected onto the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='blue', alpha=0.5)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Projection of Data onto First Two Principal Components')
plt.grid(True)
plt.show()
