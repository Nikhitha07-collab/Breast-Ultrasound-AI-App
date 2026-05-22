import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pydicom
from tensorflow.keras import layers, models


st.set_page_config(page_title="Breast Ultrasound AI", layout="wide")

st.title("Breast Ultrasound AI Diagnostic Support")
st.write("Upload PNG, JPG, JPEG, or DICOM ultrasound image.")


def build_unet():
    inputs = layers.Input((128, 128, 1))

    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(c3)

    u1 = layers.UpSampling2D((2, 2))(c3)
    u1 = layers.Concatenate()([u1, c2])
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(u1)
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(c4)

    u2 = layers.UpSampling2D((2, 2))(c4)
    u2 = layers.Concatenate()([u2, c1])
    c5 = layers.Conv2D(16, 3, activation="relu", padding="same")(u2)
    c5 = layers.Conv2D(16, 3, activation="relu", padding="same")(c5)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c5)

    return models.Model(inputs, outputs)


def build_classifier():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 1)),

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(3, activation="softmax")
    ])

    return model


@st.cache_resource
def load_models():
    seg_model = build_unet()
    cls_model = build_classifier()

    seg_model.load_weights("unet_breast_tumor.weights.h5")
    cls_model.load_weights("breast_classifier.weights.h5")

    return seg_model, cls_model


seg_model, cls_model = load_models()

class_names = ["Normal", "Benign", "Malignant"]


def read_image(uploaded_file):
    if uploaded_file.name.lower().endswith(".dcm"):
        dicom = pydicom.dcmread(uploaded_file)
        image = dicom.pixel_array.astype(np.float32)

        image = image - image.min()

        if image.max() > 0:
            image = image / image.max()

        image = (image * 255).astype(np.uint8)

    else:
        image = Image.open(uploaded_file).convert("L")
        image = np.array(image)

    return image


uploaded_file = st.file_uploader(
    "Upload ultrasound image",
    type=["png", "jpg", "jpeg", "dcm"]
)

if uploaded_file is not None:
    image = read_image(uploaded_file)

    image_small = cv2.resize(image, (128, 128)) / 255.0
    model_input = image_small.reshape(1, 128, 128, 1)

    raw_mask = seg_model.predict(model_input)[0, :, :, 0]
    mask = (raw_mask > 0.2).astype(np.uint8)

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    clean_mask = np.zeros_like(mask)

    if labels_count > 1:
        largest_region = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean_mask[labels == largest_region] = 1

    class_scores = cls_model.predict(model_input)[0]
    class_id = np.argmax(class_scores)

    predicted_class = class_names[class_id]
    confidence = class_scores[class_id]

    boxed_image = cv2.cvtColor(
        (image_small * 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )

    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    lesion_location = "No clear lesion detected."

    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            boxed_image,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

        lesion_location = (
            f"Lesion boxed at x={x}, y={y}, width={w}, height={h}."
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            image_small,
            caption="Uploaded Ultrasound Image",
            use_column_width=True
        )

    with col2:
        st.image(
            clean_mask * 255,
            caption="Predicted Lesion Mask",
            use_column_width=True
        )

    with col3:
        st.image(
            cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB),
            caption="Boxed Lesion",
            use_column_width=True
        )

    lesion_pixels = np.sum(clean_mask)
    total_pixels = clean_mask.shape[0] * clean_mask.shape[1]
    lesion_coverage = (lesion_pixels / total_pixels) * 100

    st.subheader("AI Classification Result")
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")

    st.write("**Probability Breakdown:**")
    st.write(f"Normal: {class_scores[0]:.2f}")
    st.write(f"Benign: {class_scores[1]:.2f}")
    st.write(f"Malignant: {class_scores[2]:.2f}")

    st.subheader("AI Diagnostic Note")

    note = f"""
Diagnostic Method:
Breast ultrasound image analysis.

AI Classification:
{predicted_class}

Classification Confidence:
{confidence:.2f}

Target Lesion Localization:
{lesion_location}

Predicted Lesion Coverage:
{lesion_coverage:.2f}% of the resized image.

Recommendation:
This is an AI-generated support result only. A radiologist or physician must review before clinical use.
"""

    st.text(note)