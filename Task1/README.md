# Handwritten Digit Recognition using CNN

## Overview

This project implements a Handwritten Digit Recognition System using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The model recognizes handwritten digits from 0–9 with high accuracy.

---

## Project Structure

```
Task1/
│
├── Model/
│   └── digit_model.h5
│
├── Notebook/
│   └── digit_recognition.ipynb
│
├── Source_Code/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── requirements.txt
│
├── Results/
│   ├── ClassDistribution.png
│   ├── ModelAccuracy.png
│   ├── ModelLoss.png
│   ├── ConfusionMatrix.png
│   ├── WrongPredictions.png
│   └── recognition.png
│
└── README.md
```

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Workflow

1. Load MNIST dataset
2. Preprocess and normalize images
3. Train CNN model
4. Evaluate model performance
5. Save trained model
6. Predict handwritten digits from custom images

---

## Results

Generated outputs include:

- Class Distribution Analysis
- Training Accuracy Graph
- Training Loss Graph
- Confusion Matrix
- Wrong Prediction Analysis
- Sample Digit Recognition

---

## Model

Trained model file:

```
digit_model.h5
```

---

## Author

**Bharathchandran BR**