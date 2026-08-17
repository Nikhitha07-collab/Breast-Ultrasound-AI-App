import io
import os
import tempfile
import cv2

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from tensorflow.keras import layers, models

import pydicom
from pydicom.errors import InvalidDicomError


# ============================================================
# PYDICOM COMPATIBILITY

# Supports both older and newer pydicom versions
# ============================================================

try:
    # pydicom 3.x
    from pydicom.pixels import apply_modality_lut, apply_voi_lut

except (ImportError, ModuleNotFoundError):
    # pydicom 2.x
    from pydicom.pixel_data_handlers.util import (
        apply_modality_lut,
        apply_voi_lut,
    )

# ============================================================
# U-NET SEGMENTATION MODEL
# ============================================================

def build_unet():
    """
    Build the existing U-Net segmentation architecture.

    Input:
        128 x 128 x 1 grayscale image

    Output:
        128 x 128 lesion probability mask
    """

    inputs = layers.Input((128, 128, 1))

    # Encoder block 1
    c1 = layers.Conv2D(
        16,
        3,
        activation="relu",
        padding="same"
    )(inputs)

    c1 = layers.Conv2D(
        16,
        3,
        activation="relu",
        padding="same"
    )(c1)

    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Encoder block 2
    c2 = layers.Conv2D(
        32,
        3,
        activation="relu",
        padding="same"
    )(p1)

    c2 = layers.Conv2D(
        32,
        3,
        activation="relu",
        padding="same"
    )(c2)

    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(
        64,
        3,
        activation="relu",
        padding="same"
    )(p2)

    c3 = layers.Conv2D(
        64,
        3,
        activation="relu",
        padding="same"
    )(c3)

    # Decoder block 1
    u1 = layers.UpSampling2D((2, 2))(c3)
    u1 = layers.Concatenate()([u1, c2])

    c4 = layers.Conv2D(
        32,
        3,
        activation="relu",
        padding="same"
    )(u1)

    c4 = layers.Conv2D(
        32,
        3,
        activation="relu",
        padding="same"
    )(c4)

    # Decoder block 2
    u2 = layers.UpSampling2D((2, 2))(c4)
    u2 = layers.Concatenate()([u2, c1])

    c5 = layers.Conv2D(
        16,
        3,
        activation="relu",
        padding="same"
    )(u2)

    c5 = layers.Conv2D(
        16,
        3,
        activation="relu",
        padding="same"
    )(c5)

    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c5)

    return models.Model(inputs, outputs)


# ============================================================
# CNN CLASSIFICATION MODEL
# ============================================================

def build_classifier():
    """
    Build the existing CNN classifier.

    Output classes:
        Normal
        Benign
        Malignant
    """

    model = models.Sequential([
        layers.Input(shape=(128, 128, 1)),

        layers.Conv2D(
            32,
            3,
            activation="relu",
            padding="same"
        ),

        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            64,
            3,
            activation="relu",
            padding="same"
        ),

        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            128,
            3,
            activation="relu",
            padding="same"
        ),

        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.4),

        layers.Dense(
            3,
            activation="softmax"
        )
    ])

    return model


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():
    """
    Load the existing trained weights.

    These are the same models already used in your project.
    """

    segmentation_model = build_unet()
    classification_model = build_classifier()

    segmentation_model.load_weights(
        "unet_breast_tumor.weights.h5"
    )

    classification_model.load_weights(
        "breast_classifier.weights.h5"
    )

    return segmentation_model, classification_model


try:
    seg_model, cls_model = load_models()

except Exception as error:
    st.error(
        f"Unable to load the AI models: {error}"
    )
    st.stop()


# IMPORTANT:
# This order must match the original model training order.
class_names = [
    "Normal",
    "Benign",
    "Malignant"
]


# ============================================================
# GENERAL IMAGE NORMALIZATION
# ============================================================

def normalize_to_uint8(image_array):
    """
    Normalize numerical image data to the 0-255 range.
    """

    image_array = np.asarray(
        image_array,
        dtype=np.float32
    )

    finite_values = image_array[
        np.isfinite(image_array)
    ]

    if finite_values.size == 0:
        raise ValueError(
            "Image contains no valid pixel values."
        )

    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))

    if maximum <= minimum:
        return np.zeros(
            image_array.shape,
            dtype=np.uint8
        )

    normalized = (
        image_array - minimum
    ) / (
        maximum - minimum
    )

    normalized = np.clip(
        normalized * 255.0,
        0,
        255
    )

    return normalized.astype(
        np.uint8
    )


# ============================================================
# DICOM FRAME HANDLING
# ============================================================

def select_first_frame(pixel_array):
    """
    Handle:

    - single-frame grayscale
    - multi-frame grayscale
    - RGB
    - multi-frame RGB

    For now, the first frame is used.
    """

    array = np.asarray(
        pixel_array
    )

    if array.ndim == 2:
        return array

    if array.ndim == 3:

        # RGB/RGBA image
        if array.shape[-1] in (3, 4):
            return array

        # multi-frame grayscale
        return array[0]

    if array.ndim == 4:
        return array[0]

    raise ValueError(
        f"Unsupported DICOM image shape: {array.shape}"
    )


# ============================================================
# DICOM READING
# ============================================================

def read_dicom_image(uploaded_file):
    """
    Read a DICOM file and return:

    dataset
    grayscale image
    modality
    """

    uploaded_file.seek(0)

    dataset = pydicom.dcmread(
        uploaded_file,
        force=False
    )

    if "PixelData" not in dataset:
        raise ValueError(
            "This DICOM file does not contain PixelData."
        )

    try:
        pixel_array = dataset.pixel_array

    except Exception as error:
        raise RuntimeError(
            "Unable to decode the DICOM pixel data. "
            "The file may use an unsupported compressed transfer syntax."
        ) from error

    pixel_array = select_first_frame(
        pixel_array
    )

    # Apply modality LUT when available.
    try:
        pixel_array = apply_modality_lut(
            pixel_array,
            dataset
        )

    except Exception:
        pass

    # Apply VOI LUT/windowing when available.
    try:
        pixel_array = apply_voi_lut(
            pixel_array,
            dataset
        )

    except Exception:
        pass

    image = normalize_to_uint8(
        pixel_array
    )

    photometric = str(
        getattr(
            dataset,
            "PhotometricInterpretation",
            ""
        )
    ).upper()

    # MONOCHROME1 needs inversion.
    if photometric == "MONOCHROME1":
        image = 255 - image

    # Convert RGB/RGBA DICOM to grayscale.
    if image.ndim == 3:

        if image.shape[-1] == 4:
            image = image[:, :, :3]

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

    if image.ndim != 2:
        raise ValueError(
            f"Unsupported processed DICOM shape: {image.shape}"
        )

    modality = str(
        getattr(
            dataset,
            "Modality",
            "UNKNOWN"
        )
    ).upper()

    return dataset, image, modality


# ============================================================
# SAFE DICOM METADATA
# ============================================================

def safe_value(
    dataset,
    attribute,
    default="Not available"
):
    """
    Safely read optional DICOM metadata.
    """

    value = getattr(
        dataset,
        attribute,
        None
    )

    if value is None:
        return default

    return str(value)


def get_safe_dicom_metadata(dataset):
    """
    Return selected imaging metadata.

    Patient-identifying metadata is intentionally excluded.
    """

    return {
        "Modality":
            safe_value(
                dataset,
                "Modality"
            ),

        "Manufacturer":
            safe_value(
                dataset,
                "Manufacturer"
            ),

        "Study Date":
            safe_value(
                dataset,
                "StudyDate"
            ),

        "Series Description":
            safe_value(
                dataset,
                "SeriesDescription"
            ),

        "Body Part Examined":
            safe_value(
                dataset,
                "BodyPartExamined"
            ),

        "Rows":
            safe_value(
                dataset,
                "Rows"
            ),

        "Columns":
            safe_value(
                dataset,
                "Columns"
            ),

        "Number of Frames":
            safe_value(
                dataset,
                "NumberOfFrames",
                "1"
            ),

        "Photometric Interpretation":
            safe_value(
                dataset,
                "PhotometricInterpretation"
            ),

        "Pixel Spacing":
            safe_value(
                dataset,
                "PixelSpacing"
            ),

        "Slice Thickness":
            safe_value(
                dataset,
                "SliceThickness"
            ),

        "Window Center":
            safe_value(
                dataset,
                "WindowCenter"
            ),

        "Window Width":
            safe_value(
                dataset,
                "WindowWidth"
            ),

        "SOP Class UID":
            safe_value(
                dataset,
                "SOPClassUID"
            ),

        "Study Instance UID":
            safe_value(
                dataset,
                "StudyInstanceUID"
            ),

        "Series Instance UID":
            safe_value(
                dataset,
                "SeriesInstanceUID"
            )
    }


# ============================================================
# STANDARD IMAGE READING
# ============================================================

def read_standard_image(
    uploaded_file
):
    """
    Read PNG/JPG/JPEG and convert to grayscale.
    """

    image = Image.open(
        uploaded_file
    ).convert("L")

    return np.asarray(
        image,
        dtype=np.uint8
    )


# ============================================================
# MODEL PREPROCESSING
# ============================================================

def prepare_model_input(
    image
):
    """
    Convert the image into the model input format:

    1 x 128 x 128 x 1
    """

    image_small = cv2.resize(
        image,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )

    image_normalized = (
        image_small.astype(
            np.float32
        )
        / 255.0
    )

    model_input = (
        image_normalized.reshape(
            1,
            128,
            128,
            1
        )
    )

    return (
        image_small,
        image_normalized,
        model_input
    )


# ============================================================
# SEGMENTATION
# ============================================================

def predict_clean_mask(
    model_input
):
    """
    Run the U-Net segmentation model.
    """

    raw_mask = seg_model.predict(
        model_input,
        verbose=0
    )[0, :, :, 0]

    # Keep original project threshold.
    binary_mask = (
        raw_mask > 0.2
    ).astype(
        np.uint8
    )

    labels_count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
    )

    clean_mask = np.zeros_like(
        binary_mask
    )

    if labels_count > 1:

        largest_region = (
            1
            + np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA
                ]
            )
        )

        clean_mask[
            labels == largest_region
        ] = 1

    return (
        raw_mask,
        clean_mask
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def predict_class(
    model_input
):
    """
    Run the CNN classifier.
    """

    class_scores = (
        cls_model.predict(
            model_input,
            verbose=0
        )[0]
    )

    class_id = int(
        np.argmax(
            class_scores
        )
    )

    predicted_class = (
        class_names[
            class_id
        ]
    )

    confidence = float(
        class_scores[
            class_id
        ]
    )

    return (
        class_scores,
        predicted_class,
        confidence
    )


# ============================================================
# BOUNDING BOX
# ============================================================

def create_boxed_image(
    image_small,
    clean_mask
):
    """
    Draw a red bounding box around the largest predicted lesion.
    """

    boxed_image = cv2.cvtColor(
        image_small,
        cv2.COLOR_GRAY2BGR
    )

    contours, _ = (
        cv2.findContours(
            clean_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
    )

    lesion_location = (
        "No clear lesion detected."
    )

    if contours:

        contour = max(
            contours,
            key=cv2.contourArea
        )

        if cv2.contourArea(
            contour
        ) > 0:

            x, y, width, height = (
                cv2.boundingRect(
                    contour
                )
            )

            cv2.rectangle(
                boxed_image,
                (x, y),
                (
                    x + width,
                    y + height
                ),
                (0, 0, 255),
                2
            )

            lesion_location = (
                f"x={x}, y={y}, "
                f"width={width}, "
                f"height={height}"
            )

    return (
        boxed_image,
        lesion_location
    )


# ============================================================
# LESION COVERAGE
# ============================================================

def calculate_lesion_coverage(
    clean_mask
):
    """
    Calculate lesion area percentage.
    """

    lesion_pixels = int(
        np.sum(
            clean_mask
        )
    )

    total_pixels = int(
        clean_mask.shape[0]
        * clean_mask.shape[1]
    )

    if total_pixels == 0:
        return 0.0

    return (
        lesion_pixels
        / total_pixels
    ) * 100.0


# ============================================================
# SIDEBAR / FILE UPLOAD
# ============================================================

with st.sidebar:

    st.header(
        "Breast Ultrasound Upload"
    )

    uploaded_file = st.file_uploader(
        "Choose a breast ultrasound image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "dcm"
        ]
    )

    st.divider()

    st.write(
        "**Supported formats**"
    )

    st.write(
        "Ultrasound DICOM (.dcm)"
    )

    st.write(
        "PNG / JPG / JPEG"
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is None:

    st.subheader(
        "Upload a breast ultrasound image to begin"
    )

    st.write(
        "This application is designed specifically for breast ultrasound images. "
        "Upload a PNG, JPG, JPEG, or ultrasound DICOM image to run lesion "
        "segmentation, localization, and classification."
    )

else:

    try:

        file_name = (
            uploaded_file.name.lower()
        )

        dataset = None
        modality = None

        # ====================================================
        # DICOM
        # ====================================================

        if file_name.endswith(
            ".dcm"
        ):

            (
                dataset,
                image,
                modality
            ) = read_dicom_image(
                uploaded_file
            )

            if modality != "US":
                raise ValueError(
                    "This application accepts ultrasound images only. "
                    f"The uploaded DICOM reports modality '{modality}'. "
                    "Please use the separate Medical Imaging Platform for CT or MRI."
                )

            st.success(
                "Detected Modality: US"
            )

            metadata = (
                get_safe_dicom_metadata(
                    dataset
                )
            )

            with st.expander(
                "DICOM Metadata",
                expanded=False
            ):

                for (
                    key,
                    value
                ) in metadata.items():

                    st.write(
                        f"**{key}:** {value}"
                    )

        # ====================================================
        # PNG/JPG/JPEG
        # ====================================================

        else:

            image = (
                read_standard_image(
                    uploaded_file
                )
            )

            # Standard images in this project are assumed
            # to be breast ultrasound images.
            modality = "US"

            st.success(
                "Detected Modality: US"
            )


        # ====================================================
        # VIEWER
        # ====================================================

        st.subheader(
            "Breast Ultrasound Viewer"
        )

        st.image(
            image,
            caption=(
                f"Original Image — "
                f"Modality: {modality}"
            ),
            width="stretch",
            clamp=True
        )


        # ====================================================
        # ULTRASOUND AI
        # ====================================================

        if modality == "US":

            (
                image_small,
                image_normalized,
                model_input
            ) = prepare_model_input(
                image
            )

            (
                raw_mask,
                clean_mask
            ) = predict_clean_mask(
                model_input
            )

            (
                class_scores,
                predicted_class,
                confidence
            ) = predict_class(
                model_input
            )

            (
                boxed_image,
                lesion_location
            ) = create_boxed_image(
                image_small,
                clean_mask
            )

            lesion_coverage = (
                calculate_lesion_coverage(
                    clean_mask
                )
            )

            st.subheader(
                "Breast Ultrasound Analysis"
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:

                st.image(
                    image_small,
                    caption=(
                        "Processed Ultrasound"
                    ),
                    width="stretch",
                    clamp=True
                )

            with col2:

                st.image(
                    clean_mask * 255,
                    caption=(
                        "Predicted Lesion Mask"
                    ),
                    width="stretch",
                    clamp=True
                )

            with col3:

                st.image(
                    cv2.cvtColor(
                        boxed_image,
                        cv2.COLOR_BGR2RGB
                    ),
                    caption=(
                        "Lesion Localization"
                    ),
                    width="stretch"
                )


            # =================================================
            # RESULTS
            # =================================================

            st.subheader(
                "Classification Result"
            )

            result1, result2, result3 = (
                st.columns(3)
            )

            with result1:

                st.metric(
                    "Prediction",
                    predicted_class
                )

            with result2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2%}"
                )

            with result3:

                st.metric(
                    "Lesion Coverage",
                    f"{lesion_coverage:.2f}%"
                )


            st.write(
                "### Probability Breakdown"
            )

            for (
                index,
                class_name
            ) in enumerate(
                class_names
            ):

                probability = float(
                    class_scores[
                        index
                    ]
                )

                st.write(
                    f"**{class_name}:** "
                    f"{probability:.2%}"
                )

                st.progress(
                    min(
                        max(
                            probability,
                            0.0
                        ),
                        1.0
                    )
                )


        else:
            st.error(
                "This Breast Ultrasound application supports ultrasound images only."
            )


    # ========================================================
    # FRIENDLY ERROR HANDLING
    # ========================================================

    except InvalidDicomError:

        st.error(
            "The uploaded file is not a valid DICOM file."
        )


    except UnidentifiedImageError:

        st.error(
            "The uploaded PNG/JPG/JPEG file could not be read."
        )


    except RuntimeError as error:

        st.error(
            str(error)
        )

        st.info(
            "If this is a compressed DICOM, an additional "
            "DICOM pixel decoder may be required."
        )


    except Exception as error:

        st.error(
            f"Unable to process the uploaded image: {error}"
        )

        with st.expander(
            "Technical Details"
        ):

            st.exception(
                error
            )