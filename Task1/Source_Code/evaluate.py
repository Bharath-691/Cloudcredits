import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Load Saved Model
print("Loading Model...")

model = load_model("../Model/digit_model.h5")

print("Model Loaded Successfully")

# Load Dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Preprocess Data
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

print("Test Data Loaded")

# Make Predictions
predictions = model.predict(x_test)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

# Accuracy
accuracy = accuracy_score(
    y_test,
    predicted_classes
)

print("\nModel Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    predicted_classes
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# Classification Report
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predicted_classes
    )
)

# Wrong Predictions
wrong = np.where(
    predicted_classes != y_test
)[0]

plt.figure(figsize=(12, 8))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        x_test[wrong[i]].reshape(28, 28),
        cmap='gray'
    )

    plt.title(
        f"A:{y_test[wrong[i]]} P:{predicted_classes[wrong[i]]}"
    )

    plt.axis('off')

plt.tight_layout()
plt.show()

print("\nEvaluation Completed Successfully")