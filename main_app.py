import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt

# 모델과 클래스 로딩
MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f]

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 이미지 전처리 함수
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# 예측 함수
def classify_image(image):
    input_tensor = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = np.argmax(output)
    confidence = float(np.max(output))
    return class_names[predicted_index], confidence, output

# Streamlit UI
st.title("Image Scanning Toolkit")
st.write("Upload your images and get instant classification results using an AI-based machine learning model.")

uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    class_counts = {name: 0 for name in class_names}
    confidence_scores = {name: [] for name in class_names}

    for file in uploaded_files:
        image = Image.open(file)
        label, confidence, raw = classify_image(image)
        st.image(image, caption=f"예측: {label} ({confidence*100:.1f}%)", use_container_width=True)
        class_counts[label] += 1
        confidence_scores[label].append(confidence)

    # 클래스별 이미지 개수 시각화
    st.subheader("Image Count by Class")
    fig, ax = plt.subplots()
    ax.bar(class_counts.keys(), class_counts.values())
    ax.set_ylabel("Class")
    ax.set_xlabel("Number")
    st.pyplot(fig)


    st.subheader("Average Confidence")
    avg_conf = {k: (np.mean(v)*100 if v else 0) for k, v in confidence_scores.items()}
    for k, v in avg_conf.items():
        st.write(f"{k}: {v:.1f}%")
