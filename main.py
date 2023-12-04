import tensorflow as tf  # Import TensorFlow library
import numpy as np  # Import NumPy library
import matplotlib.pyplot as plt  # Import Matplotlib library

# Load the MNIST dataset from TensorFlow
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Expand the dimensions of the images for the convolutional layers
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

# Function to create a model
def create_model():
  # Define the model architecture
  model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),  # Input layer for the images
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),  # First convolutional layer
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),  # Second convolutional layer
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer
    tf.keras.layers.Dropout(0.25),  # Dropout layer to prevent overfitting
    tf.keras.layers.Flatten(),  # Flatten layer to convert 3D data to 1D
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # Output layer for 10 classes
  ])
  # Compile the model
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model  # Return the compiled model

# Create an ImageDataGenerator object with specified parameters for data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
  rotation_range=30,  # Rotate images by up to 30 degrees
  width_shift_range=0.25,  # Shift images horizontally by up to 25%
  height_shift_range=0.25,  # Shift images vertically by up to 25%
  shear_range=0.25,  # Apply shear transformations by up to 25%
  zoom_range=0.2  # Zoom in on images by up to 20%
)

# Create generators for the training and testing datasets
train_generator = datagen.flow(train_images, train_labels)
test_generator = datagen.flow(test_images, test_labels)

print("Model fitting...")
# Create and train the model using the training data generator
improved_model = create_model()
improved_model.fit(train_generator, epochs=10, validation_data=test_generator)

print("Model converting...")
# Convert the trained model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(improved_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

print("Model saving...")
# Save the converted model to a .tflite file
f = open('mnistt.tflite', "wb")
f.write(tflite_quantized_model)
f.close()