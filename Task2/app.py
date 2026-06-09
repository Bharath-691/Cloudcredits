import streamlit as st
import pickle

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 3.5rem;
    font-weight: bold;
    color: white;
    margin-bottom: 0px;
}

.subtitle {
    text-align: center;
    color: #B0B0B0;
    font-size: 1.1rem;
    margin-bottom: 30px;
}

.stTextArea textarea {
    border-radius: 15px !important;
    border: 2px solid #4F8BF9 !important;
    font-size: 16px !important;
}

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    background-color: #4F8BF9;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background-color: #2E6EF7;
    color: white;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
try:
    model = pickle.load(open("model/sentiment_model.pkl", "rb"))
    vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
except:
    st.error("❌ Model files not found.")
    st.stop()

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.title("📊 Project Dashboard")

    st.metric(
        label="Model Accuracy",
        value="84.57%"
    )

    st.markdown("---")

    st.subheader("🛠 Technologies")

    st.markdown("""
    - Python
    - Streamlit
    - Scikit-Learn
    - TF-IDF Vectorizer
    - Naive Bayes
    """)

    st.markdown("---")

    st.subheader("📚 Dataset")

    st.write("IMDb Movie Reviews Dataset")

    st.markdown("---")

    st.info("🤖 Machine Learning Powered Sentiment Analysis")

# =========================
# HEADER
# =========================
st.markdown(
    '<p class="title">🎬 Movie Review Sentiment Analyzer</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI-Powered Sentiment Detection using Machine Learning</p>',
    unsafe_allow_html=True
)

# =========================
# INPUT SECTION
# =========================
review = st.text_area(
    "✍️ Enter Movie Review",
    placeholder="Example: This movie was amazing, the acting was brilliant and the storyline was engaging...",
    height=180
)

# =========================
# PREDICTION
# =========================
if st.button("🚀 Analyze Sentiment"):

    if review.strip() == "":
        st.warning("⚠️ Please enter a review first.")

    else:

        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)

        st.markdown("---")

        if prediction[0] == 1:

            st.success("😊 Positive Review")

            st.info("""
This review expresses positive emotions, appreciation,
satisfaction, or favorable opinions.
""")

            st.balloons()

        else:

            st.error("😞 Negative Review")

            st.info("""
This review expresses negative emotions, criticism,
dissatisfaction, or unfavorable opinions.
""")

# =========================
# SAMPLE REVIEWS
# =========================
st.markdown("---")

st.subheader("🎭 Try These Sample Reviews")

col1, col2 = st.columns(2)

with col1:
    st.success(
        "This movie was fantastic. The acting and story were amazing."
    )

with col2:
    st.error(
        "Worst movie ever. Completely boring and a waste of time."
    )

# =========================
# FOOTER
# =========================
st.markdown("---")

st.markdown(
    '<p class="footer">Built using Streamlit • Scikit-Learn • IMDb Dataset</p>',
    unsafe_allow_html=True
)