import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Title
st.title("🎬 Movie Review Sentiment Analyzer")

# Input box
review = st.text_area("Enter a Movie Review")

# Prediction button
if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)

        if prediction[0] == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")