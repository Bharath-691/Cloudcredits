import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense


print("Loading MNIST Dataset...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Dataset Loaded Successfully")
print("Training Shape:", x_train.shape)
print("Testing Shape:", x_test.shape)

# Normalize Data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Data Preprocessing Completed")

# Build CNN Model
model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(28, 28, 1)
    )
)

model.add(MaxPooling2D((2, 2)))

model.add(
    Conv2D(
        64,
        (3, 3),
        activation='relu'
    )
)

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(
    Dense(
        64,
        activation='relu'
    )
)

model.add(
    Dense(
        10,
        activation='softmax'
    )
)

print("\nModel Summary")
model.summary()

# Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining Started...\n")

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)

print("\nTraining Completed")

# Save Model
model.save("../Model/digit_model.h5")

print("\nModel Saved Successfully")
print("Location: ../Model/digit_model.h5")