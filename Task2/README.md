# 🎬 Movie Review Sentiment Analyzer

## Overview

This project performs sentiment analysis on movie reviews using Machine Learning. It classifies a review as **Positive** or **Negative** through an interactive Streamlit web application.

## Dataset

- IMDb Movie Reviews Dataset
- 50,000 movie reviews
- Balanced positive and negative samples
Source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Technologies Used

- Python
- Pandas
- Scikit-Learn
- Streamlit
- Matplotlib
- Seaborn

## Model

- TF-IDF Vectorization
- Multinomial Naive Bayes Classifier

## Results

- **Accuracy:** 84.57%
- Evaluated using Accuracy, Precision, Recall, and F1 Score

## Features

- Real-time sentiment prediction
- Interactive Streamlit interface
- Positive/Negative review classification
- Machine Learning powered text analysis

## Screenshots

### Streamlit Application

![Application UI](screenshots/app_ui.png)

### Positive Review Prediction

![Positive Review](screenshots/positive_review.png)

### Negative Review Prediction

![Negative Review](screenshots/negative_review.png)

### Sentiment Distribution

![Sentiment Distribution](screenshots/sentiment_distribution.png)

### Confusion Matrix

![Confusion Matrix](screenshots/confusion_matrix.png)

## Project Structure

```text
Task2/
├── app.py
├── data/
├── model/
├── notebook/
├── screenshots/
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Sample Prediction

### Input

```text
This movie was absolutely fantastic and amazing.
```

### Output

```text
😊 Positive Review
```

## Author

**Bharath Chandran BR**

Machine Learning & Artificial Intelligence Internship Project