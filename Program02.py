import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Load the MNIST dataset
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Task 1: Plot loss and accuracy for training and validation data
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Plot loss and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

# Task 2: Plot an image from the test data and perform inference
plt.figure()
plt.imshow(X_test[0], cmap='gray')
plt.title(f"Actual Label: {y_test[0]}")

# Perform inference
test_image = X_test[0].reshape(1, 28, 28)  # Reshape for model input
prediction = model.predict(test_image)
predicted_label = np.argmax(prediction)
print(f"Predicted Label: {predicted_label}")

# Task 3: Change the number of hidden layers and activation function
model2 = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='tanh'),  # Change activation to 'tanh'
    layers.Dense(64, activation='tanh'),   # Change activation to 'tanh'
    layers.Dense(10, activation='softmax')
])

model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history2 = model2.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Task 4: Run without scaling the images
# No scaling is applied, so we can reuse the original X_train and X_test

model3 = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history3 = model3.fit(X_train, y_train, epochs=10, validation_split=0.2)

# You can also plot loss and accuracy for history2 and history3 as in Task 1.