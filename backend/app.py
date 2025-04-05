from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

app = Flask(__name__)
CORS(app)

# 모델 준비
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()

# 이미지 분석 함수들
def analyze_colors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    avg_color = np.mean(image, axis=(0, 1))
    avg_brightness = np.mean(hsv[:, :, 2])
    avg_saturation = np.mean(hsv[:, :, 1])
    return {
        "avg_color": avg_color.tolist(),
        "avg_brightness": avg_brightness,
        "avg_saturation": avg_saturation
    }

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)
    return len(keypoints)

def classify_animal(image):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = resnet_model(input_tensor)
    class_id = output.argmax(dim=1).item()
    return f"Predicted class ID: {class_id}"

def blip_caption(image):
    inputs = blip_processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

@app.route("/analyze", methods=["POST"])
def analyze_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "이미지가 업로드되지 않았습니다."}), 400

    image = Image.open(file.stream).convert("RGB")
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # 분석 시작
    result = {}
    result["colors"] = analyze_colors(rgb_image)
    result["features"] = extract_features(rgb_image)
    result["classification"] = classify_animal(rgb_image)
    result["caption"] = blip_caption(image)
    result["emotion"] = emotion_classifier(result["caption"])

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
