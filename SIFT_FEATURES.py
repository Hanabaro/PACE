# Necessary imports
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.color import rgb2gray
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load CSV file with image paths and Voc values
image_dir = 'C:/Users/yousf/OneDrive/Desktop/solar_model/images/'  # Folder with solar panel images
csv_file = 'C:/Users/yousf/OneDrive/Desktop/solar_model/Voc_Values.csv'  # CSV with image paths and voltage values
df = pd.read_csv(csv_file)

# Inspect the DataFrame
print(df.head())

# Example DataFrame:
#        image_path      open_voltage
# 0    img1.jpg              22.5
# 1    img2.jpg              23.0
# 2    img3.jpg              21.8

# Function to extract and visualize SIFT features
def extract_sift_features(image):
    # Convert the image to grayscale for SIFT
    gray_image = rgb2gray(image)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Convert the grayscale image to uint8 (SIFT requires this)
    gray_image_uint8 = (gray_image * 255).astype(np.uint8)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_image_uint8, None)

    # Handle the case where no keypoints are found
    if descriptors is None:
        descriptors = np.zeros((1, 128))

    # Visualize keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(gray_image_uint8, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Create a histogram of the SIFT descriptors
    (hist, _) = np.histogram(descriptors.ravel(), bins=128, range=(0, 256))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    # Visualize the grayscale image, the image with SIFT keypoints, and the histogram
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    ax[0].imshow(gray_image, cmap='gray')
    ax[0].set_title('Grayscale Image')
    ax[0].axis('off')
    
    ax[1].imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    ax[1].set_title('SIFT Keypoints')
    ax[1].axis('off')
    
    ax[2].bar(np.arange(len(hist)), hist, width=0.3)
    ax[2].set_title('SIFT Histogram')
    
    plt.show()

    # Print the SIFT descriptor histogram
    print("SIFT Feature Histogram Values:", hist)
    
    return hist

# Solar Image Data Generator with SIFT feature extraction
class SolarImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size, image_dir, target_size=(224, 224), shuffle=True):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.image_dir = image_dir
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.dataframe) // self.batch_size

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.dataframe.iloc[batch_indices]
        images = np.array([self.load_image(img_path) for img_path in batch_df['image_path']])
        sift_features = np.array([self.load_sift(img_path) for img_path in batch_df['image_path']])
        labels = batch_df['Voltage'].values
        return [images, sift_features], labels

    def on_epoch_end(self):
        self.indices = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def load_image(self, img_path):
        img = load_img(self.image_dir + img_path, target_size=self.target_size)
        img = img_to_array(img)
        img /= 255.0  # Normalize pixel values to [0, 1]
        return img

    def load_sift(self, img_path):
        img = load_img(self.image_dir + img_path, target_size=self.target_size)
        img = img_to_array(img)
        img /= 255.0  # Normalize the image
        return extract_sift_features(img)

# Load and visualize SIFT features for a sample image
img_path = 'C:/Users/yousf/OneDrive/Desktop/solar_model/images/_20240902_151621.jpg'
img = load_img(img_path, target_size=(224, 224))
img = img_to_array(img) / 255.0  # Normalize the image
sift_features = extract_sift_features(img)

# Model Creation
resnet_input = Input(shape=(224, 224, 3), name='resnet_input')
sift_input = Input(shape=(128,), name='sift_input')  # SIFT produces 128-dimensional descriptors

# ResNet50 as the base model for image features
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=resnet_input)
base_model.trainable = False

# Extract ResNet50 features
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)

# Combine ResNet50 features with SIFT features
combined = Concatenate()([x, sift_input])

# Add fully connected layers
combined = Dense(128, activation='relu')(combined)
combined = Dropout(0.2)(combined)

# Output layer (for regression task)
output = Dense(1, activation='linear')(combined)

# Build the final model
model = tf.keras.Model(inputs=[resnet_input, sift_input], outputs=output)

# Compile the model
optimizer = Adam(learning_rate=1e-4, clipvalue=1.0)  # Clip gradients to a maximum value
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Create a new model that outputs the ResNet50 features
resnet_feature_extractor = Model(inputs=model.input, outputs=base_model.output)

# Print ResNet50 Features for a Sample Image
resnet_features = resnet_feature_extractor.predict([np.expand_dims(img, axis=0), np.expand_dims(sift_features, axis=0)])
print("ResNet50 Feature Values:", resnet_features.flatten())

# Train-test split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create data generators
batch_size = 16
train_generator = SolarImageDataGenerator(train_df, batch_size=batch_size, image_dir=image_dir)
val_generator = SolarImageDataGenerator(val_df, batch_size=batch_size, image_dir=image_dir)

# Train the model
history = model.fit(train_generator, validation_data=val_generator, epochs=10, verbose=1)

# Evaluate the model on validation data
val_loss, val_mae = model.evaluate(val_generator)
print(f'Validation MAE: {val_mae}')
