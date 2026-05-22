from keras.models import load_model

seg_model = load_model("unet_breast_tumor_model.h5", compile=False)
cls_model = load_model("breast_ultrasound_classifier.h5", compile=False)

seg_model.save("unet_breast_tumor_model.keras")
cls_model.save("breast_ultrasound_classifier.keras")

print("Models converted successfully!")