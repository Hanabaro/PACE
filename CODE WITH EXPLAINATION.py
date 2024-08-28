# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:43:45 2024

@author: yousf
@supervised: Jincheol Kim 
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to images directory
image_dir = 'path/to/images'

# Load pre-trained VGG16 model for feature extraction (without the top classification layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Data generator for augmentation
data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to extract features using VGG16
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    features = base_model.predict(image)
    return features.flatten()

# Iterate through images and extract features
features_list = []
for image_file in os.listdir(image_dir):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        features = extract_features(os.path.join(image_dir, image_file))
        features_list.append(features)

# Convert to numpy array for further processing
features_array = np.array(features_list)
import pandas as pd
from statsmodels.tsa.seasonal import STL

# Simulated time-series data
time_series_data = pd.Series(features_array[:, 0])  # Use the first feature as an example

# STL decomposition
stl = STL(time_series_data, seasonal=13)
result = stl.fit()

trend = result.trend
seasonal = result.seasonal
residual = result.resid

# Plotting the components (optional)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.subplot(311)
plt.plot(trend)
plt.title('Trend')
plt.subplot(312)
plt.plot(seasonal)
plt.title('Seasonal')
plt.subplot(313)
plt.plot(residual)
plt.title('Residual')
plt.tight_layout()
plt.show()
from sklearn.decomposition import PCA

# Assume 'features_array' is the matrix of features extracted from images
pca = PCA(n_components=10)
pca_features = pca.fit_transform(features_array)

# Print explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Simulated target variable (e.g., some performance metric)
y = np.random.rand(features_array.shape[0])  # Replace with actual target values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(pca_features, y, test_size=0.2, random_state=42)

# Gradient Boosting Machine
gbm = GradientBoostingRegressor()
gbm.fit(X_train, y_train)

# Support Vector Regression
svr = SVR()
svr.fit(X_train, y_train)

# Implementing LSTM (using TensorFlow/Keras)
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(pca_features.shape[1], 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape data for LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train LSTM
model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Predicting and evaluating
gbm_pred = gbm.predict(X_test)
svr_pred = svr.predict(X_test)
lstm_pred = model.predict(X_test_lstm)

# Combine predictions (ensemble approach)
final_pred = (gbm_pred + svr_pred + lstm_pred.flatten()) / 3
from tensorflow.keras.layers import Dropout

# Redefine LSTM model with dropout for Monte Carlo
mc_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(pca_features.shape[1], 1)),
    tf.keras.layers.Dropout(0.2, training=True),  # Force dropout during prediction
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dropout(0.2, training=True),
    tf.keras.layers.Dense(1)
])

mc_model.compile(optimizer='adam', loss='mean_squared_error')
mc_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Monte Carlo sampling
mc_predictions = np.array([mc_model.predict(X_test_lstm) for _ in range(100)])
mc_mean = mc_predictions.mean(axis=0)
mc_std = mc_predictions.std(axis=0)

# Display mean and uncertainty
print("MC Mean:", mc_mean.flatten())
print("MC Std Dev (Uncertainty):", mc_std.flatten())
import shap

# Use the GBM model for SHAP analysis
explainer = shap.Explainer(gbm, X_train)
shap_values = explainer(X_test)

# Plot SHAP summary plot
shap.summary_plot(shap_values, X_test)
from sklearn.model_selection import cross_val_score

# k-fold cross-validation
cv_scores = cross_val_score(gbm, pca_features, y, cv=5)
print("CV Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Further validation on separate hold-out test set
# Assuming test_set_features and test_set_targets are defined
# test_score = gbm.score(test_set_features, test_set_targets)
# print("Test Score:", test_score)
# Function to retrain the model with new data
def retrain_model(new_data_features, new_data_targets):
    # Add new data to existing dataset
    updated_features = np.concatenate((pca_features, new_data_features), axis=0)
    updated_targets = np.concatenate((y, new_data_targets), axis=0)
    
    # Retrain GBM
    gbm.fit(updated_features, updated_targets)
    
    # Retrain LSTM
    updated_features_lstm = updated_features.reshape((updated_features.shape[0], updated_features.shape[1], 1))
    model.fit(updated_features_lstm, updated_targets, epochs=50, batch_size=32, validation_split=0.2)

# Example usage with new data (replace with actual new data)
# new_data_features, new_data_targets = ...  # Load new data
# retrain_model(new_data_features, new_data_targets)
