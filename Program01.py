import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Task 1: Add more Dense layers to the existing code and check accuracy changes
# Load the Breast Cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Sequential model
model = keras.Sequential()

# Add more Dense layers to the model
model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))

# Add the output layer with a sigmoid activation for binary classification
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model on the test data
accuracy = model.evaluate(X_test, y_test)[1]
print(f"Accuracy with additional Dense layers: {accuracy}")

# Task 2: Change the data source to Breast Cancer dataset
# Data is already loaded above.

# Task 3: Normalize the data before feeding it to the model
sc = StandardScaler()
X_train_normalized = sc.fit_transform(X_train)
X_test_normalized = sc.transform(X_test)

# Create a new model for normalized data
model_normalized = keras.Sequential()
model_normalized.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model_normalized.add(layers.Dense(32, activation='relu'))
model_normalized.add(layers.Dense(16, activation='relu'))
model_normalized.add(layers.Dense(1, activation='sigmoid'))

model_normalized.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_normalized.fit(X_train_normalized, y_train, epochs=10, batch_size=32, verbose=1)

accuracy_normalized = model_normalized.evaluate(X_test_normalized, y_test)[1]
print(f"Accuracy with data normalization: {accuracy_normalized}")
