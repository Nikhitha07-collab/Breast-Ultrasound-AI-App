import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Breast Ultrasound AI", layout="wide")

st.title("Breast Ultrasound AI Diagnostic Support")

st.write("Testing TensorFlow model loading...")

@st.cache_resource
def load_models():

    seg_model = load_model("unet_breast_tumor_model.h5")

    cls_model = load_model("breast_ultrasound_classifier.h5")

    return seg_model, cls_model

try:

    seg_model, cls_model = load_models()

    st.success("TensorFlow models loaded successfully!")

except Exception as error:

    st.error("Model loading failed.")

    st.write(error)