import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)
client = OpenAI(api_key=api_key)

# 1. 로컬 이미지 불러오기
def load_image(image_path):
    if not image_path:
        return None  # 빈 문자열이면 바로 None 반환
    image = cv2.imread(image_path)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# 2. 색상, 명도, 채도 분석
def analyze_colors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    avg_color = np.mean(image, axis=(0, 1))
    avg_brightness = np.mean(hsv[:, :, 2])
    avg_saturation = np.mean(hsv[:, :, 1])
    return {
        "avg_color": avg_color,
        "avg_brightness": avg_brightness,
        "avg_saturation": avg_saturation
    }

# 3. 특징점 추출 (SIFT)
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return len(keypoints)

# 4. 이미지 분류 (ResNet50)
def classify_animal(image):
    model = models.resnet50(pretrained=True)
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    predicted_class = output.argmax(dim=1).item()
    return f"Predicted class ID: {predicted_class}"

# 5. GPT-4 Vision으로 이미지 설명/느낌 분석
def gpt4_image_analysis(image_url):
    response = client.chat.completions.create(
        model="gpt-4o",  # 또는 "gpt-4-vision-preview"
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "이 이미지를 보고 어떤 대상이 보이며, 전체적인 분위기나 감정을 묘사해주세요."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        max_tokens=300
    )
    return response.choices[0].message.content

# 6. 전체 실행 함수
def process_image(local_path, url):
    result = {}

    if local_path:
        image = load_image(local_path)
        if image is not None:
            result["색상/명도/채도 분석"] = analyze_colors(image)
            result["특징점 개수"] = extract_features(image)
            result["ResNet 분류 결과"] = classify_animal(image)
        else:
            result["로컬 이미지 오류"] = "이미지를 불러올 수 없습니다."

    if url:
        result["GPT-4 Vision 분석 결과"] = gpt4_image_analysis(url)

    return result


# 예시 실행
if __name__ == "__main__":
    local_image_path = ""  # 로컬 이미지 경로
    uploaded_image_url = "https://upload.wikimedia.org/wikipedia/commons/5/56/Tiger.50.jpg"  # 인터넷 URL

    result = process_image(local_image_path, uploaded_image_url)
    for key, value in result.items():
        print(f"\n {key}:\n{value}")
