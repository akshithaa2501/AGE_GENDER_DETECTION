import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from keras.models import load_model

# Page Config
st.set_page_config(
    page_title="Age Gender Hair Detection",
    page_icon="👤",
    layout="centered"
)

# CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #1E3A8A;
    margin-bottom: 2rem;
}

.result-text {
    font-size: 1.2rem;
    font-weight: 500;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}

.image-container {
    padding: 10px;
    border-radius: 10px;
    background-color: rgba(237,242,247,0.5);
}
</style>
""", unsafe_allow_html=True)


# Load Model
@st.cache_resource
def load_age_gender_model():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(
        BASE_DIR,
        "Age_Sex_Detection.h5"
    )

    model = load_model(
        model_path,
        compile=False
    )

    return model


# Preprocess
def preprocess_image(uploaded_image):

    if uploaded_image.mode != "RGB":
        uploaded_image = uploaded_image.convert("RGB")

    image = uploaded_image.resize((48, 48))

    image_array = np.array(image) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# Predict Age Gender
def predict_age_gender(model, image_array):

    predictions = model.predict(image_array)

    age = int(
        np.round(predictions[1][0][0])
    )

    gender_prob = float(
        predictions[0][0][0]
    )

    gender = (
        "Female"
        if gender_prob > 0.5
        else "Male"
    )

    confidence = (
        gender_prob
        if gender == "Female"
        else 1 - gender_prob
    )

    return age, gender, confidence


# Hair Detection
def detect_hair_length(image):

    img = np.array(image)

    h, w = img.shape[:2]

    left_region = img[
        int(h * 0.15):int(h * 0.85),
        0:int(w * 0.20)
    ]

    right_region = img[
        int(h * 0.15):int(h * 0.85),
        int(w * 0.80):w
    ]

    left_gray = cv2.cvtColor(
        left_region,
        cv2.COLOR_RGB2GRAY
    )

    right_gray = cv2.cvtColor(
        right_region,
        cv2.COLOR_RGB2GRAY
    )

    left_dark = np.sum(left_gray < 100)

    right_dark = np.sum(right_gray < 100)

    total_pixels = (
        left_gray.size +
        right_gray.size
    )

    hair_ratio = (
        left_dark +
        right_dark
    ) / total_pixels

    if hair_ratio > 0.25:
        return "Long Hair"
    else:
        return "Short Hair"


# Main
def main():

    st.markdown(
        '<div class="main-header">Age Gender Hair Detection</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Loading Model..."):

        model = load_age_gender_model()

    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button("Detect"):

            for i, uploaded_file in enumerate(uploaded_files):

                image = Image.open(uploaded_file)

                st.image(
                    image,
                    caption=uploaded_file.name,
                    width=300
                )

                processed_image = preprocess_image(
                    image
                )

                age, gender, confidence = predict_age_gender(
                    model,
                    processed_image
                )

                hair = detect_hair_length(
                    image
                )

                original_gender = gender

                # Internship Logic
                if 20 <= age <= 30:

                    if hair == "Long Hair":

                        final_gender = "Female"

                    else:

                        final_gender = "Male"

                    rule_status = "Hair Rule Applied"

                else:

                    final_gender = original_gender

                    rule_status = "Original Gender Used"

                st.markdown(
                    f"""
                    <div class="result-text"
                    style="background-color:rgba(37,99,235,0.1);">
                    Age: {age}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="result-text"
                    style="background-color:rgba(34,197,94,0.15);">
                    Hair Length: {hair}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="result-text"
                    style="background-color:rgba(168,85,247,0.15);">
                    Original Gender: {original_gender}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="result-text"
                    style="background-color:rgba(59,130,246,0.15);">
                    Final Gender: {final_gender}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="result-text"
                    style="background-color:rgba(250,204,21,0.15);">
                    Rule Status: {rule_status}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write(
                    f"Confidence: {confidence:.2%}"
                )

                st.divider()


if __name__ == "__main__":
    main()