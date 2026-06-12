import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# ---------------------------
# Load Model
# ---------------------------

model = load_model("Model/digit_model.h5")
def display_prediction_results(prediction):
    """
    Helper function to process and display the predicted digit, 
    confidence meter, and the top 3 class probabilities.
    """
    digit = np.argmax(prediction)
    confidence_decimal = float(np.max(prediction))  # Safe float between 0.0 and 1.0
    confidence_pct = confidence_decimal * 100

    # 1. Main Prediction Visual Card
    st.markdown(
        f"""
        <div class="pred-card">
            <div class="pred-digit">{digit}</div>
            <div class="pred-text">Predicted Digit</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Confidence Metrics
    st.progress(confidence_decimal)
    st.write(f"**Confidence:** {confidence_pct:.2f}%")

    # 3. Top 3 Probabilities Breakdown
    st.subheader("Top 3 Predictions")
    
    # Sort indices by probability in ascending order, take the last 3, and reverse them
    top3_indices = np.argsort(prediction[0])[-3:][::-1]

    for idx in top3_indices:
        prob = prediction[0][idx] * 100
        # Use a streamlit progress bar or clean text format
        st.write(f"**Digit {idx}:** {prob:.2f}%")
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️"
)
with st.sidebar:

    st.header("🧠 Model Information")

    st.markdown("""
    **Dataset:** MNIST

    **Training Images:** 60,000

    **Testing Images:** 10,000

    **Architecture:** CNN

    **Accuracy:** 98.89%
    """)

    st.markdown("---")

    st.markdown("""
    ### 🚀 Features

    ✅ Upload Image

    ✅ Camera Capture

    ✅ Draw Digit

    ✅ CNN Prediction

    ✅ Confidence Score

    ✅ Top-3 Predictions
    """)
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#050816,#0f172a,#111827);
}

.block-container{
    max-width:1100px;
    padding-top:2rem;
}
.glass-card{
background:rgba(255,255,255,0.08);
backdrop-filter:blur(12px);
padding:20px;
border-radius:20px;
border:1px solid rgba(255,255,255,0.1);
text-align:center;
margin-bottom:15px;
}

.pred-card{
    background: linear-gradient(135deg,#00c853,#00a152);
    padding:25px;
    border-radius:20px;
    text-align:center;
}

.pred-digit{
    font-size:60px;
    font-weight:bold;
    color:white;
}

.pred-text{
    color:white;
    font-size:20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Preprocessing Function
# ---------------------------
def preprocess_digit(img):
    """
    MNIST-style preprocessing for uploaded images,
    camera captures, and drawn digits.
    """

    try:

        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Reduce noise
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        # Thin overly thick strokes
        #kernel = np.ones((2, 2), np.uint8)
        #thresh = cv2.erode(
            #thresh,
            #kernel,
            #iterations=1
        #)

        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None, None

        # Largest contour
        contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(contour)

        digit = thresh[y:y+h, x:x+w]

        # Preserve aspect ratio
        if w > h:
            new_w = 20
            new_h = max(1, int(h * 20 / w))
        else:
            new_h = 20
            new_w = max(1, int(w * 20 / h))

        digit = cv2.resize(
            digit,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        # Create 28x28 MNIST canvas
        canvas = np.zeros((28, 28), dtype=np.uint8)

        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2

        canvas[
            y_offset:y_offset + new_h,
            x_offset:x_offset + new_w
        ] = digit

        # Slight dilation after resize
        kernel = np.ones((2, 2), np.uint8)
        canvas = cv2.dilate(
            canvas,
            kernel,
            iterations=1
        )

        preview = canvas.copy()

        processed = canvas.astype("float32") / 255.0

        processed = processed.reshape(
            1,
            28,
            28,
            1
        )

        return processed, preview

    except Exception as e:
        print(f"Preprocessing Error: {e}")
        return None, None

# ---------------------------
# UI
# ---------------------------

st.markdown("""
<h1 style='text-align:center;font-size:60px;'>
✍️ Handwritten Digit Recognition
</h1>

<p style='text-align:center;font-size:20px;color:#B8BCC8;'>
CNN-Based Recognition using the MNIST Dataset
</p>
""", unsafe_allow_html=True)

col1,col2,col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="glass-card">
    <h4>🎯 Accuracy</h4>
    <h2>98.89%</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
    <h4>🔢 Classes</h4>
    <h2>10</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="glass-card">
    <h4>🧠 Model</h4>
    <h2>CNN</h2>
    </div>
    """, unsafe_allow_html=True)
option = st.radio(
    "Choose Input Method",
    ["Upload Image", "Camera Capture", "Draw Digit"]
)

# ==================================================
# IMAGE UPLOAD
# ==================================================

if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a digit image",
        type=["png", "jpg", "jpeg"]
    )

    # 🚨 EVERYTHING BELOW MUST BE INDENTED INSIDE THIS IF-STATEMENT 🚨
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = np.array(image)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if np.mean(img) > 127:
            img = 255 - img

        processed, preview = preprocess_digit(img)

        if processed is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            with col2:
                st.image(preview, caption="Processed Digit", use_container_width=True)

            # Predict and display using our function
            prediction = model.predict(processed, verbose=0)
            display_prediction_results(prediction)
        else:
            st.warning("No digit detected.")
# ==================================================
# CAMERA CAPTURE
# ==================================================
elif option == "Camera Capture":

    st.subheader("📷 Capture a Handwritten Digit")
    camera_image = st.camera_input("Take a picture of a handwritten digit")

    # Everything below this line MUST be indented inside this if-statement!
    if camera_image is not None:
        image = Image.open(camera_image)
        img = np.array(image)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if np.mean(img) > 127:
            img = 255 - img

        processed, preview = preprocess_digit(img)

        if processed is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Captured Image", use_container_width=True)
            with col2:
                st.image(preview, caption="Processed Digit", use_container_width=True)

            # Predict and show results
            prediction = model.predict(processed, verbose=0)
            display_prediction_results(prediction)
        else:
            st.warning("No digit detected.")

# ==================================================
# ==================================================
# DRAW DIGIT
# ==================================================
else:
    st.write("Draw a digit below:")

    # FIX: Replaced the literal '...' with your actual original parameters
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=5,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button("Predict Digit"):
        if canvas_result.image_data is not None:
            img = canvas_result.image_data.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

            processed, preview = preprocess_digit(img)

            if processed is not None:
                st.image(preview, caption="Processed Digit", width=150)

                # Predict and call the top 3 function
                prediction = model.predict(processed, verbose=0)
                display_prediction_results(prediction)
            else:
                st.warning("No digit detected on the canvas.")