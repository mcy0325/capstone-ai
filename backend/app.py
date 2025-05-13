from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from keybert import KeyBERT
from skimage.feature import local_binary_pattern, hog
from skimage.measure import shannon_entropy
import sys

sys.path.append('./yolov5')
sys.path.append('./U-2-Net')

app = Flask(__name__)
CORS(app)
kw_model = KeyBERT()

# 🔥 Feature functions
def blur_detection(image_gray):
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()

def colorfulness(image):
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    std_root = np.sqrt((np.std(rg) ** 2) + (np.std(yb) ** 2))
    mean_root = np.sqrt((np.mean(rg) ** 2) + (np.mean(yb) ** 2))
    return std_root + (0.3 * mean_root)

def aspect_ratio(image):
    h, w = image.shape[:2]
    return w / h

def edge_density(edges):
    return int(np.sum(edges > 0))

def dominant_color(image, k=3):
    data = image.reshape((-1, 3))
    data = np.float32(data)
    _, labels, centers = cv2.kmeans(data, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS)
    dominant = centers[np.argmax(np.bincount(labels.flatten()))]
    return dominant.tolist()

def dominant_color_masked(image, mask_path, k=3):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask_bin = mask_resized > 128
    masked_pixels = image[mask_bin]

    if len(masked_pixels) == 0:
        return [[0, 0, 0]]  # fallback: 검정색 1개 반환

    data = np.float32(masked_pixels)
    _, labels, centers = cv2.kmeans(data, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS)

    counts = np.bincount(labels.flatten())
    sorted_idx = np.argsort(-counts)
    sorted_centers = centers[sorted_idx]
    return sorted_centers.tolist()



def bright_region_ratio(image_gray, threshold=200):
    bright_pixels = np.sum(image_gray > threshold)
    total_pixels = image_gray.size
    return bright_pixels / total_pixels

def symmetry(image_gray):
    flipped = cv2.flip(image_gray, 1)
    diff = cv2.absdiff(image_gray, flipped)
    return np.mean(diff)

def image_entropy(image_gray):
    return shannon_entropy(image_gray)

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
    return edges, save_path

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

def analyze_texture(image_gray):
    lbp = local_binary_pattern(image_gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)
    return hist.tolist()

def analyze_contrast(image_gray):
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    variance = laplacian.var()
    return float(variance)

def analyze_hog(image_gray):
    features, _ = hog(image_gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return features[:20].tolist()

def optical_flow(img1, img2, save_path="static/optical_flow_result.jpg"):
    prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if prev_gray.shape != next_gray.shape:
        next_gray = cv2.resize(next_gray, (prev_gray.shape[1], prev_gray.shape[0]))
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_magnitude = np.mean(mag)
    avg_angle = np.mean(ang)
    hsv = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, bgr_flow)
    return {
        "average_magnitude": float(avg_magnitude),
        "average_angle": float(avg_angle),
        "flow_image_path": save_path
    }

def motion_interpretation(flow_data):
    if flow_data["average_magnitude"] > 2.0:
        return "빠른 움직임"
    elif flow_data["average_magnitude"] > 0.5:
        return "느린 움직임"
    else:
        return "정지 상태"

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
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image_pil).unsqueeze(0)

    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(input_tensor)
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)
        pred = pred.squeeze().cpu().numpy()
        pred = (pred * 255).astype(np.uint8)
        result_img = Image.fromarray(pred)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result_img.save(save_path)

    return save_path

@app.route("/analyze", methods=["POST"])
def analyze_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "이미지가 업로드되지 않았습니다."}), 400

    image = Image.open(file.stream).convert("RGB")
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    keywords = kw_model.extract_keywords(caption, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    description_features = [kw for kw, _ in keywords]

    detections = detect_objects_with_yolo(rgb_image)
    object_count = len(detections)

    edges, edge_img_path = edge_detection(rgb_image)
    seg_mask_path = segment_image(image)
    dominant_obj_colors = dominant_color_masked(rgb_image, seg_mask_path, k=3)

    result = {
        "features": extract_features(rgb_image),
        "colors": analyze_colors(rgb_image),
        "colorfulness": colorfulness(rgb_image),
        "dominant_color": dominant_color(rgb_image),
        "dominant_color_object_list": dominant_obj_colors,
        "aspect_ratio": aspect_ratio(rgb_image),
        "blur": blur_detection(gray_image),
        "entropy": image_entropy(gray_image),
        "symmetry": symmetry(gray_image),
        "bright_region_ratio": bright_region_ratio(gray_image),
        "brain_regions": simulate_brain_activation(description_features),
        "edge_density": edge_density(edges),
        "detected_parts": detections,
        "object_count": object_count,
        "description_features": description_features,
        "edge_image_url": edge_img_path,
        "segmentation_mask_url": seg_mask_path,
        "texture_histogram": analyze_texture(gray_image),
        "contrast": analyze_contrast(gray_image),
        "hog_summary": analyze_hog(gray_image),
        "caption": caption
    }
    return jsonify(result)

@app.route("/opticalflow", methods=["POST"])
def analyze_optical_flow():
    file1 = request.files.get("image1")
    file2 = request.files.get("image2")
    if not file1 or not file2:
        return jsonify({"error": "두 장의 이미지가 필요합니다."}), 400

    img1 = cv2.cvtColor(np.array(Image.open(file1.stream).convert("RGB")), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(Image.open(file2.stream).convert("RGB")), cv2.COLOR_RGB2BGR)

    flow_result = optical_flow(img1, img2)
    motion_label = motion_interpretation(flow_result)

    result = {
        "optical_flow": flow_result,
        "motion_label": motion_label
    }
    return jsonify(result)

@app.route("/videoflow", methods=["POST"])
def analyze_video():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "비디오 파일이 필요합니다."}), 400

    video_path = "./temp_video.mp4"
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    total_flow = []

    while cap.isOpened():
        ret, next_frame = cap.read()
        if not ret:
            break
        flow = optical_flow(prev_frame, next_frame)
        total_flow.append(flow)
        prev_frame = next_frame

    cap.release()
    os.remove(video_path)

    avg_magnitude = np.mean([f["average_magnitude"] for f in total_flow])
    avg_angle = np.mean([f["average_angle"] for f in total_flow])

    result = {
        "average_magnitude": float(avg_magnitude),
        "average_angle": float(avg_angle),
        "motion_label": motion_interpretation({"average_magnitude": avg_magnitude})
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
