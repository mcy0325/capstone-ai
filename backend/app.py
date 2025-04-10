from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from keybert import KeyBERT
import sys

# 경로 등록
sys.path.append('./yolov5')
sys.path.append('./U-2-Net')

app = Flask(__name__)
CORS(app)

kw_model = KeyBERT()

# ---------------- 분석 함수들 ----------------
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

def edge_detection(image, save_path="static/edge_result.jpg"):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, edges)
    return save_path

def detect_objects_with_yolo(image_np):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    results = model(image_np)
    detections = results.pandas().xyxy[0]
    output = []
    for _, row in detections.iterrows():
        output.append({
            "label": row["name"],
            "confidence": float(row["confidence"]),
            "bbox": [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])]
        })
    return output

def simulate_brain_activation(description_features):
    mapping = {
        "color": "V4 – 색상 처리",
        "edge": "V1 – 윤곽 처리",
        "shape": "V3 – 형태 인식",
        "object": "FFA – 대상을 시각적으로 인식",
        "motion": "MT/V5 – 움직임 처리"
    }
    activated = set()
    for word in description_features:
        for key in mapping:
            if key in word.lower():
                activated.add(mapping[key])
    if not activated:
        activated.add("V1 – 기본 시각 처리")
    return list(activated)

def segment_image(image_pil, save_path="static/segmentation_result.png"):
    from u2net_test import normPRED
    from model import U2NET

    model_path = './U-2-Net/saved_models/u2net/u2net_portrait.pth'
    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image_pil).unsqueeze(0)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    with torch.no_grad():
        d1, _, _, _, _, _, _ = net(image_tensor)
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)
        pred = pred.squeeze().cpu().numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pred = (pred * 255).astype(np.uint8)
    mask = Image.fromarray(pred).convert("L")
    mask.save(save_path)

    return save_path

# ---------------- 라우터 ----------------
@app.route("/analyze", methods=["POST"])
def analyze_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "이미지가 업로드되지 않았습니다."}), 400

    image = Image.open(file.stream).convert("RGB")
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    keywords = kw_model.extract_keywords(caption, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    description_features = [kw for kw, _ in keywords]

    result = {
        "features": extract_features(rgb_image),
        "colors": analyze_colors(rgb_image),
        "detected_parts": detect_objects_with_yolo(rgb_image),
        "description_features": description_features,
        "brain_regions": simulate_brain_activation(description_features),
        "edge_image_url": edge_detection(rgb_image),
        "segmentation_mask_url": segment_image(image)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
