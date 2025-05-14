import os
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. 모델 로드
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# 2. 클래스 이름 로드 (labels.txt에서 자동)
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f]

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. 이미지 전처리 (손상 이미지 무시)
def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)
        img = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"[무시됨] 손상된 이미지: {image_path} → {e}")
        return None

# 4. 중복 방지된 파일명 생성
def generate_unique_filename(folder, base_name, extension):
    full_path = os.path.join(folder, f"{base_name}{extension}")
    counter = 2
    while os.path.exists(full_path):
        full_path = os.path.join(folder, f"{base_name} ({counter}){extension}")
        counter += 1
    return full_path

# 5. 예측 + 이동
def classify_and_move(image_path):
    img_tensor = preprocess_image(image_path)
    if img_tensor is None:
        return

    interpreter.set_tensor(input_details[0]['index'], img_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    max_prob = np.max(output)
    predicted_index = np.argmax(output)
    predicted_label = class_names[predicted_index]
    confidence = f"{max_prob * 100:.1f}%"

    dest_folder = os.path.join("sorted", predicted_label)
    os.makedirs(dest_folder, exist_ok=True)

    _, ext = os.path.splitext(image_path)
    base_name = f"{predicted_label} {confidence}"
    dest_path = generate_unique_filename(dest_folder, base_name, ext)

    shutil.move(image_path, dest_path)
    print(f"{image_path} → {os.path.basename(dest_path)}")

# 6. 하위 폴더까지 전체 탐색
input_dir = "input_images"
for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(root, filename)
            classify_and_move(filepath)


