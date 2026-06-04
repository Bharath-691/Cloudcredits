# Handwritten Digit Recognition using CNN

## Overview

This project implements a Handwritten Digit Recognition System using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The model recognizes handwritten digits from 0–9 with high accuracy.

## Model Performance

- Dataset: MNIST Handwritten Digits
- Model: Convolutional Neural Network (CNN)
- Test Accuracy: 98.89%

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

## Workflow

1. Load and preprocess the MNIST dataset
2. Perform Exploratory Data Analysis (EDA)
3. Build and train a CNN model
4. Evaluate model performance
5. Save the trained model
6. Predict handwritten digits from input images

## Results

Generated outputs include:

- Class Distribution Analysis
- Training Accuracy Graph
- Training Loss Graph
- Confusion Matrix
- Wrong Prediction Analysis
- Sample Digit Recognition

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Author

**Bharathchandran BR**