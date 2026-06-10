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
# ADVANCED CUSTOM CSS
# =========================
st.markdown("""
<style>
/* Main App Styling */
.stApp {
    background-color: #0F172A; /* Slate 900 */
    font-family: 'Inter', system-ui, sans-serif;
}

/* Header Sections */
.title-container {
    text-align: center;
    padding: 2rem 0 0.5rem 0;
}

.title-text {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38BDF8, #818CF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
    letter-spacing: -0.05em;
}

.subtitle-text {
    color: #94A3B8; /* Slate 400 */
    font-size: 1.1rem;
    font-weight: 400;
    margin-top: 5px;
    margin-bottom: 2rem;
}

/* Form Inputs & Textarea Labels */
label p {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #E2E8F0 !important;
}

.stTextArea textarea {
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    font-size: 16px !important;
    transition: all 0.3s ease;
}

.stTextArea textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
}

/* Modernized Primary Button */
div.stButton > button {
    width: 100% !important;
    height: 52px;
    border-radius: 12px;
    background: linear-gradient(135deg, #4F46E5, #6366F1);
    color: white !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    transition: all 0.2s ease-in-out;
}

div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
    background: linear-gradient(135deg, #4338CA, #4F46E5);
}

div.stButton > button:active {
    transform: translateY(1px);
}

/* Custom Sample Review Cards */
.sample-card {
    background-color: #1E293B;
    border-left: 5px solid #64748B;
    padding: 1.25rem;
    border-radius: 8px;
    color: #E2E8F0;
    font-size: 0.95rem;
    line-height: 1.5;
    min-height: 90px;
    display: flex;
    align-items: center;
}
.pos-card { border-left-color: #10B981; background-color: rgba(16, 185, 129, 0.08); }
.neg-card { border-left-color: #EF4444; background-color: rgba(239, 68, 68, 0.08); }

/* Sidebar styling overrides */
section[data-testid="stSidebar"] {
    background-color: #1E293B !important;
}

.tech-badge {
    display: inline-block;
    background-color: #334155;
    color: #94A3B8;
    padding: 4px 10px;
    border-radius: 6px;
    margin: 3px 2px;
    font-size: 0.85rem;
    font-weight: 500;
}

.footer-text {
    text-align: center;
    color: #64748B;
    font-size: 0.85rem;
    margin-top: 3rem;
    border-top: 1px solid #334155;
    padding-top: 1.5rem;
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
    st.error("❌ Model files not found. Ensure paths for your pkl files are correct.")
    st.stop()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("<h2 style='color:#F8FAFC; font-weight:700; margin-bottom:0;'>📊 Project Dashboard</h2>", unsafe_allow_html=True)
    st.write("")
    
    st.metric(
        label="Model Accuracy",
        value="84.57%"
    )
    
    st.markdown("<hr style='margin:1rem 0; border-color:#334155;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#E2E8F0;'>🛠 Technologies</h4>", unsafe_allow_html=True)
    
    # Render modern visual badges instead of standard bullet points
    st.markdown("""
    <span class="tech-badge">Python</span>
    <span class="tech-badge">Streamlit</span>
    <span class="tech-badge">Scikit-Learn</span>
    <span class="tech-badge">TF-IDF Vectorizer</span>
    <span class="tech-badge">Naive Bayes</span>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin:1rem 0; border-color:#334155;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#E2E8F0;'>📚 Dataset</h4>", unsafe_allow_html=True)
    st.info("IMDb Movie Reviews Dataset")
    
    st.markdown("<hr style='margin:1rem 0; border-color:#334155;'>", unsafe_allow_html=True)
    st.success("🤖 Core Inference Engine Active")

# =========================
# HEADER
# =========================
st.markdown("""
<div class="title-container">
    <p class="title-text">🎬 Movie Review Sentiment Analyzer</p>
    <p class="subtitle-text">Instant AI-powered sentiment breakdown using Natural Language Processing</p>
</div>
""", unsafe_allow_html=True)

# =========================
# INPUT SECTION
# =========================
review = st.text_area(
    "✍️ Share Your Review:",
    placeholder="Type or paste a movie review here... e.g., 'The cinematography was striking, but the plot dragged in the second act.'",
    height=150
)

st.write("") # Layout spacing spacer

# =========================
# PREDICTION LOGIC & DISPLAY
# =========================
if st.button("🚀 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter some review text before running the analysis.")
    else:
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)
        
        st.markdown("<h3 style='color:#F8FAFC; font-size:1.3rem; margin-top:1.5rem;'>📊 Analysis Result</h3>", unsafe_allow_html=True)
        
        if prediction[0] == 1:
            st.success("😊 **Positive Sentiment Detected**")
            st.markdown("""
            <div style='background-color:rgba(16, 185, 129, 0.1); border-left:4px solid #10B981; padding:12px; border-radius:6px; color:#E2E8F0;'>
                <strong>Evaluation:</strong> This review indicates clear appreciation, highlighting positive emotional tone, approval, or content satisfaction.
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.error("😞 **Negative Sentiment Detected**")
            st.markdown("""
            <div style='background-color:rgba(239, 68, 68, 0.1); border-left:4px solid #EF4444; padding:12px; border-radius:6px; color:#E2E8F0;'>
                <strong>Evaluation:</strong> This review surfaces critical language, structural complaints, underlying dissatisfaction, or an unfavorable overview.
            </div>
            """, unsafe_allow_html=True)

# =========================
# SAMPLE REVIEWS (STUNNING TILES)
# =========================
st.markdown("<br><h3 style='color:#F8FAFC; font-size:1.3rem; margin-bottom:1rem;'>🎭 Test Driving Templates</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="sample-card pos-card">
        🟢 "This movie was fantastic. The casting choices and narrative direction were absolute masterpieces."
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="sample-card neg-card">
        🔴 "Worst cinematic experience this year. Completely boring pacing and an absolute waste of time."
    </div>
    """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer-text">
    Built with ❤️ using Streamlit • Scikit-Learn Pipeline • IMDb Evaluation Set
</div>
""", unsafe_allow_html=True)