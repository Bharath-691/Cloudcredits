import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load Trained Model
model = load_model("../Model/digit_model.h5")

# Image Path
image_path = input("Enter image path: ")

# Open Image
img = Image.open(image_path).convert('L')

# Resize to MNIST size
img = img.resize((28, 28))

# Convert to NumPy Array
img = np.array(img)

# Invert colors (white digit on black background)
img = 255 - img

# Normalize
img = img / 255.0

# Reshape for CNN
img = img.reshape(1, 28, 28, 1)

# Prediction
prediction = model.predict(img)

predicted_digit = np.argmax(prediction)

print("\nPredicted Digit =", predicted_digit)