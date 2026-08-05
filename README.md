# Breast Ultrasound AI Diagnostic Support System

An educational medical imaging application that uses deep learning to analyze breast ultrasound images. The system performs lesion segmentation, lesion localization, and classification into Normal, Benign, or Malignant categories.

## Live Application

Add your Streamlit application link here.

## Portfolio

https://nikhita-portfolio-liard.vercel.app

## Project Overview

This project demonstrates an end-to-end healthcare AI workflow for breast ultrasound image analysis.

Users can upload a PNG, JPG, JPEG, or DICOM ultrasound image. The application preprocesses the image, generates a lesion segmentation mask, identifies the suspected lesion region, performs classification, and displays the prediction with confidence scores.

The project is intended for educational and research purposes.

## Features

- PNG, JPG, and JPEG image upload
- DICOM image upload
- DICOM pixel-data extraction
- DICOM metadata display
- MONOCHROME1 and MONOCHROME2 handling
- Multi-frame DICOM support using the first frame
- Breast lesion segmentation using U-Net
- Image classification using a CNN
- Normal, Benign, and Malignant predictions
- Lesion-mask visualization
- Bounding-box lesion localization
- Confidence score
- Probability breakdown
- Lesion-coverage calculation
- Interactive Streamlit interface

## AI Workflow

1. Upload a breast ultrasound image
2. Read and validate the image
3. Extract DICOM pixels when applicable
4. Convert the image to grayscale
5. Resize the image to 128 × 128
6. Normalize pixel values
7. Run the U-Net segmentation model
8. Generate the predicted lesion mask
9. Keep the largest connected lesion region
10. Draw a bounding box around the predicted lesion
11. Run the CNN classification model
12. Display prediction, confidence, and probabilities

## Models

### U-Net Segmentation Model

The U-Net model performs pixel-level lesion segmentation. It identifies which image pixels are likely to belong to a lesion.

The architecture contains:

- Encoder blocks
- Convolution layers
- Max-pooling layers
- Bottleneck
- Decoder blocks
- Upsampling
- Skip connections
- Sigmoid output layer

### CNN Classification Model

The CNN classifies the ultrasound image into:

- Normal
- Benign
- Malignant

The architecture contains:

- Convolution layers
- ReLU activation
- Max-pooling
- Flatten layer
- Dense layer
- Dropout
- Softmax output

## DICOM Support

The application uses `pydicom` to:

- Read DICOM datasets
- Extract pixel data
- Detect imaging metadata
- Apply modality transformations
- Apply VOI LUT or window information
- Correct MONOCHROME1 display
- Process grayscale and RGB images
- Handle multi-frame files using the first frame

Direct patient-identifying metadata is not displayed.

## Technologies Used

- Python
- Streamlit
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pillow
- pydicom
- Git
- GitHub

## Project Files

```text
Breast-Ultrasound-AI-App/
│
├── app.py
├── app_ultrasound_backup.py
├── requirements.txt
├── runtime.txt
├── README.md
├── breast_classifier.weights.h5
├── breast_ultrasound_classifier.h5
├── breast_ultrasound_classifier.keras
├── unet_breast_tumor.weights.h5
├── unet_breast_tumor_model.h5
├── unet_breast_tumor_model.keras
├── convert_models.py
├── export_weights.py
└── nikhita-portfolio/