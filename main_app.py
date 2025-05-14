import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
import tensorflow as tf
tflite = tf.lite

# 모델 로드
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# 라벨 로드
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# 이미지 전처리 함수
def preprocess_image(image):
    image = image.resize((input_shape[2], input_shape[1]))
    image = ImageOps.fit(image, (input_shape[2], input_shape[1]), Image.ANTIALIAS)
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

uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("Results")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        prediction = predict(image)
        predicted_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        st.write(f"**{uploaded_file.name}** → `{class_names[predicted_idx]}` ({confidence*100:.2f}%)")

