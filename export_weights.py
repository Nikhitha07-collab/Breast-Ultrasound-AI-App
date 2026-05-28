from tensorflow.keras.models import load_model

seg_model = load_model("unet_breast_tumor_model.h5", compile=False)
cls_model = load_model("breast_ultrasound_classifier.h5", compile=False)

seg_model.save_weights("unet_breast_tumor.weights.h5")
cls_model.save_weights("breast_classifier.weights.h5")

print("Weights saved successfully.")