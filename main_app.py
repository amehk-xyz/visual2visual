import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
import tensorflow as tf

# 모델 로드
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# 라벨 로드
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# 전처리 함수
def preprocess_image(image):
    image = ImageOps.fit(
        image,
        (input_shape[2], input_shape[1]),
        method=Image.Resampling.LANCZOS
    )
    image = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# 예측 함수
def predict(image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.squeeze(output_data)

# UI
st.title("Visual to Visual Toolkit")
st.write("Upload your images and get instant classification results using an AI-based machine learning model.")

uploaded_files = st.file_uploader("Drag and drop files here", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("Results")

    # 예측 결과 저장용
    class_counts = {label: 0 for label in class_names}
    confidences = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        prediction = predict(image)
        predicted_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        predicted_label = class_names[predicted_idx]
        class_counts[predicted_label] += 1
        confidences.append(confidence)

        st.image(image, width=300)
        st.write(f"**{uploaded_file.name}** → `{predicted_label}` ({confidence*100:.2f}%)")

    # 📊 클래스별 이미지 개수
    st.subheader("📊 Image Count by Class")
    fig, ax = plt.subplots()
    ax.bar(class_counts.keys(), class_counts.values())
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # 📈 평균 신뢰도
    st.subheader("📈 Average Confidence")
    avg_confidence = np.mean(confidences)
    st.write(f"{avg_confidence*100:.2f}%")


