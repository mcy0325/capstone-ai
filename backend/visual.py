import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from keybert import KeyBERT
from skimage.feature import local_binary_pattern, hog
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
from transformers import BlipProcessor, BlipForConditionalGeneration
import sys
import requests
from dotenv import load_dotenv

sys.path.append('./yolov5')
sys.path.append('./U-2-Net')

load_dotenv()

kw_model = KeyBERT()

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

def glcm_contrast(image_gray):
    glcm = graycomatrix(image_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return float(graycoprops(glcm, 'contrast')[0, 0])

def mask_area_ratio(mask_path, image_shape):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_resized = cv2.resize(mask, (image_shape[1], image_shape[0]))
    mask_bin = mask_resized > 128
    return float(np.sum(mask_bin)) / mask_bin.size

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

def generate_claude_prompt_with_object_behavior(flow_data, yolo_results_frame1, yolo_results_frame2, image_analysis):
    movement_descriptions = []

    for obj1 in yolo_results_frame1:
        for obj2 in yolo_results_frame2:
            if obj1["label"] == obj2["label"]:
                x1 = (obj1["bbox"][0] + obj1["bbox"][2]) / 2
                y1 = (obj1["bbox"][1] + obj1["bbox"][3]) / 2
                x2 = (obj2["bbox"][0] + obj2["bbox"][2]) / 2
                y2 = (obj2["bbox"][1] + obj2["bbox"][3]) / 2
                dx = x2 - x1
                dy = y2 - y1

                direction = []
                if dy > 20:
                    direction.append("아래로")
                elif dy < -20:
                    direction.append("위로")
                if dx > 20:
                    direction.append("오른쪽으로")
                elif dx < -20:
                    direction.append("왼쪽으로")
                direction_str = " ".join(direction) if direction else "거의 이동 없음"

                distance = (dx ** 2 + dy ** 2) ** 0.5

                movement_descriptions.append(
                    f"- 어떤 객체가 '{direction_str}' 약 {distance:.1f}px 이동했습니다."
                )

    if not movement_descriptions:
        movement_descriptions.append("- 탐지된 객체의 유의미한 이동이 관찰되지 않았습니다.")

    flow_summary = f"- 평균 속도: {flow_data.get('average_magnitude', 0):.2f}\n" \
                   f"- 평균 방향: {flow_data.get('average_angle', 0):.2f} 라디안"

    visual_summary = f"- 색상 변화: {image_analysis.get('dominant_color_change', '정보 없음')}\n" \
                     f"- 밝은 영역 변화: {image_analysis.get('bright_region_change', '정보 없음')}\n" \
                     f"- 배경 변화 판단: {image_analysis.get('background_stability', '정보 없음')}"

    prompt = f"""
[객체 이동 분석]
{chr(10).join(movement_descriptions)}

[추가 시각 정보]
{visual_summary}

[Optical Flow 요약 정보]
{flow_summary}

위 정보를 바탕으로,
각 움직임을 보인 객체가 어떤 동작을 수행 중일 수 있는지 **정체를 유추하지 않고**, 
순수하게 움직임과 방향성에 기반해 가능한 일반적 행동 유형을 설명해주세요.

예: 아래로 빠르게 움직였다면 '달리는 중', 천천히 왼쪽으로 움직였다면 '걷는 중'일 수 있습니다.

반드시 **'개', '사람' 등 객체 이름은 사용하지 말고**, 동작 유형 중심으로 서술해주세요.
"""
    return prompt.strip()


# Claude 호출 함수

def query_claude(prompt):
    api_key = os.getenv("CLAUDE_API_KEY")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "claude-opus-4-20250514",
        "max_tokens": 1024,
        "temperature": 0.5,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
        return response.json()["content"][0]["text"]
    except Exception as e:
        return f"Claude 응답 실패: {str(e)}"

def generate_caption_and_keywords(image_pil):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(images=image_pil, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    keywords = kw_model.extract_keywords(caption, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    description_features = [kw for kw, _ in keywords]
    return caption, description_features


# YOLO 모델 전역 로딩
print("Loading YOLOv5 model...")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects_with_yolo(image_np):
    results = yolo_model(image_np)
    detections = results.pandas().xyxy[0]
    output = []
    for _, row in detections.iterrows():
        output.append({
            "label": row["name"],
            "confidence": float(row["confidence"]),
            "bbox": [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])]
        })
    return output

def analyze_visual(video_path, segments=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # frame_interval = int(fps)

    # frame_idx = 0
    # saved_idx = 0
    # frames = []
    frame_results = []
    flow_results = []
    frames = []

    if segments:
        for idx, seg in enumerate(segments):
            start_frame = int(seg["start_sec"] * fps)
            end_frame = int(seg["end_sec"] * fps)
            
            if end_frame <= start_frame + 2:
                continue

            # 중간 지점 2개 추출
            mid1 = start_frame + (end_frame - start_frame) // 3
            mid2 = start_frame + 2 * (end_frame - start_frame) // 3

            # 첫 번째 프레임
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid1)
            ret1, frame1 = cap.read()
            # 두 번째 프레임
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid2)
            ret2, frame2 = cap.read()
            
            if not (ret1 and ret2):
                continue

        for i, frame in enumerate([frame1, frame2]):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            edges, edge_img_path = edge_detection(rgb)
            seg_mask_path = segment_image(pil)
            dom_colors = dominant_color_masked(rgb, seg_mask_path)
            caption, description_features = generate_caption_and_keywords(pil)
            detections = detect_objects_with_yolo(rgb)

            result = {
                "frame_index": f"{idx}_{i}",
                "features": extract_features(rgb),
                "colors": analyze_colors(rgb),
                "colorfulness": colorfulness(rgb),
                "dominant_color": dominant_color(rgb),
                "dominant_color_object_list": dom_colors,
                "aspect_ratio": aspect_ratio(rgb),
                "blur": blur_detection(gray),
                "entropy": image_entropy(gray),
                "symmetry": symmetry(gray),
                "bright_region_ratio": bright_region_ratio(gray),
                "edge_density": edge_density(edges),
                "texture_histogram": analyze_texture(gray),
                "contrast": analyze_contrast(gray),
                "hog_summary": analyze_hog(gray),
                "caption": caption,
                "glcm_contrast": glcm_contrast(gray),
                "mask_area_ratio": mask_area_ratio(seg_mask_path, rgb.shape),
                "description_features": description_features,
                "detected_parts": detections,
                "object_count": len(detections),
                "brain_regions": simulate_brain_activation(description_features),
                "edge_image_url": edge_img_path,
                "segmentation_mask_url": seg_mask_path,
            }

            frame_results.append(result)
            frames.append(frame)
            
        # Optical flow between the two frames
        flow = optical_flow(frame1, frame2)
        flow["between_frames"] = [f"{idx}_0", f"{idx}_1"]
        flow["motion_label"] = motion_interpretation(flow)
        flow_results.append(flow)

    cap.release()
    
    return {
        "frame_analysis": frame_results,
        "optical_flow": flow_results,
    }, frames
    
