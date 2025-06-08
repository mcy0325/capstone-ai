import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter
import os
from dotenv import load_dotenv
from anthropic import Anthropic
import tensorflow as tf
import tensorflow_hub as hub
import scipy.signal

load_dotenv()
anthropic = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

# Load YAMNet model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load YAMNet class names
def class_names_from_csv(class_map_csv_text):
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        import csv
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

yamnet_class_map_path = yamnet_model.class_map_path().numpy()
yamnet_class_names = class_names_from_csv(yamnet_class_map_path)

def classify_with_yamnet(y_segment, sr):
    if y_segment.ndim > 1:
        y_segment = np.mean(y_segment, axis=1)
    y_segment = y_segment.astype(np.float32)
    if sr != 16000:
        desired_length = int(round(float(len(y_segment)) / sr * 16000))
        y_segment = scipy.signal.resample(y_segment, desired_length)
        sr = 16000
    waveform = y_segment / np.max(np.abs(y_segment))
    scores, embeddings, spectrogram = yamnet_model(waveform)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    top_indices = np.argsort(mean_scores)[::-1][:5]
    top_predictions = [(yamnet_class_names[i], float(mean_scores[i])) for i in top_indices]
    return top_predictions

def claude_api_call(prompt):
    message = anthropic.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=1000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

def extract_features(y, sr, start=None, end=None, hop_length=512):
    segment = y if start is None or end is None else y[start:end]
    return {
        "zcr": np.mean(librosa.feature.zero_crossing_rate(y=segment, hop_length=hop_length)[0]),
        "centroid": np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr, hop_length=hop_length)[0]),
        "bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr, hop_length=hop_length)[0]),
        "rolloff": np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr, hop_length=hop_length)[0]),
        "mfcc": np.mean(librosa.feature.mfcc(y=segment, sr=sr, hop_length=hop_length, n_mfcc=13), axis=1).round(2).tolist()
    }

def extract_background_features(y, sr, exclude_segments, hop_length=512):
    mask = np.ones(len(y), dtype=bool)
    for start_sec, end_sec in exclude_segments:
        start = int(start_sec * sr)
        end = int(end_sec * sr)
        mask[start:end] = False
    return extract_features(y[mask], sr, hop_length=hop_length)

def build_comparative_prompt(background, focus, start_sec, end_sec, top_predictions):
    top_text = "\n".join([f"- {label}: {score:.2%}" for label, score in top_predictions])
    return f"""
전체 오디오에서 특정 구간({start_sec:.2f}s ~ {end_sec:.2f}s)을 제외한 배경 소리의 특징은 다음과 같습니다:

- 평균 MFCC: {background["mfcc"]}
- ZCR: {round(background["zcr"], 5)}
- Spectral Centroid: {round(background["centroid"], 2)} Hz
- Spectral Bandwidth: {round(background["bandwidth"], 2)} Hz
- Spectral Roll-off: {round(background["rolloff"], 2)} Hz

반면, 특정 구간({start_sec:.2f}s ~ {end_sec:.2f}s)은 다음과 같은 특징을 가집니다:

- 평균 MFCC: {focus["mfcc"]}
- ZCR: {round(focus["zcr"], 5)}
- Spectral Centroid: {round(focus["centroid"], 2)} Hz
- Spectral Bandwidth: {round(focus["bandwidth"], 2)} Hz
- Spectral Roll-off: {round(focus["rolloff"], 2)} Hz

YAMNet 모델이 분류한 상위 5개 소리 후보는 다음과 같습니다:
{top_text}

이 소리는 동물 소리 중 하나입니다. 어떤 소리가 실제에 가장 가까울까요? 이 소리에 대해 알 수 있는 정보는 무엇이며, 어떤 감정이나 인지가 유발될 수 있을까요?
"""

def save_segment_graphs(y, sr, start, end, index, output_dir="segment_graphs"):
    segment = y[start:end]
    hop_length = 512
    n_fft = 2048

    mfcc = librosa.feature.mfcc(y=segment, sr=sr, hop_length=hop_length, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y=segment, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr, hop_length=hop_length)[0]

    frames = range(len(zcr))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    axs = axs.flatten()

    librosa.display.waveshow(segment, sr=sr, ax=axs[0])
    axs[0].set_title('Waveform')

    img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=hop_length, ax=axs[1])
    axs[1].set_title('MFCC')
    if img:
        fig.colorbar(img, ax=axs[1])

    axs[2].plot(times, zcr)
    axs[2].set_title('Zero Crossing Rate')

    axs[3].plot(times, centroid)
    axs[3].set_title('Spectral Centroid')

    axs[4].plot(times, bandwidth)
    axs[4].set_title('Spectral Bandwidth')

    axs[5].plot(times, rolloff)
    axs[5].set_title('Spectral Roll-off')

    plt.tight_layout()
    filename = f"{output_dir}/segment_{index+1}_{round(start/sr, 2)}s_{round(end/sr, 2)}s.png"
    plt.savefig(filename)
    plt.close()

def merge_similar_segments(segments, sr, max_gap=0.4, centroid_thresh=300, zcr_thresh=0.05):
    merged = []
    current = segments[0]

    for next_seg in segments[1:]:
        time_gap = (next_seg["start"] - current["end"]) / sr
        centroid_diff = abs(next_seg["centroid"] - current["centroid"])
        zcr_diff = abs(next_seg["zcr"] - current["zcr"])

        if time_gap <= max_gap and centroid_diff <= centroid_thresh and zcr_diff <= zcr_thresh:
            current["end"] = next_seg["end"]
        else:
            merged.append((current["start"], current["end"]))
            current = next_seg

    merged.append((current["start"], current["end"]))
    return merged

def extract_features_for_segment(y, sr, start, end, hop_length=512):
    segment = y[start:end]
    return {
        "start": start,
        "end": end,
        "centroid": np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr, hop_length=hop_length)[0]),
        "zcr": np.mean(librosa.feature.zero_crossing_rate(y=segment, hop_length=hop_length)[0]),
        "bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr, hop_length=hop_length)[0]),
    }

def analyze_audio(file_path):
    if not file_path or not os.path.exists(file_path):
        print("유효한 파일 경로가 아닙니다.")
        return

    global sr
    y, sr = librosa.load(file_path, sr=None)
    hop_length = 512
    n_fft = 2048
    min_duration = 0.3

    energy = np.array([
        np.sum(np.abs(y[i:i+n_fft])**2)
        for i in range(0, len(y), hop_length)
    ])
    mask = energy > np.percentile(energy, 75)

    frames = np.where(mask)[0]
    raw_intervals = []
    for k, g in groupby(enumerate(frames), lambda x: x[0]-x[1]):
        group = list(map(itemgetter(1), g))
        start_f, end_f = group[0], group[-1]
        start = start_f * hop_length
        end = min((end_f + 1) * hop_length, len(y))
        duration = (end - start) / sr
        if duration >= min_duration:
            raw_intervals.append((start, end))

    segments_with_features = [
        extract_features_for_segment(y, sr, start, end)
        for start, end in raw_intervals
    ]

    if not segments_with_features:
        print("의미 있는 구간이 탐지되지 않았습니다.")
        return

    merged_segments = merge_similar_segments(segments_with_features, sr)

    print("\n<병합된 소리 구간>")
    results = []
    for i, (start, end) in enumerate(merged_segments):
        save_segment_graphs(y, sr, start, end, i)
        print(f"구간 {i+1}: [{round(start/sr, 2)}s ~ {round(end/sr, 2)}s]")

        start_sec = round(start / sr, 2)
        end_sec = round(end / sr, 2)

        focus_features = extract_features(y, sr, start, end)
        background_features = extract_background_features(y, sr, exclude_segments=[(start_sec, end_sec)])
        top_predictions = classify_with_yamnet(y[start:end], sr)

        prompt = build_comparative_prompt(background_features, focus_features, start_sec, end_sec, top_predictions)
        response = claude_api_call(prompt)

        results.append({
            "segment": i + 1,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "focus_features": focus_features,
            "background_features": background_features,
            "yamnet_top_predictions": top_predictions,
            "claude_prompt": prompt,
            "claude_response": response
        })

    print("\n- 그래프 저장 완료 (segment_graphs/ 폴더 확인)")
    
    segment_ranges = [
        {"start_sec": round(start / sr + 3.0, 2), "end_sec": round(end / sr + 3.0, 2)}
        for start, end in merged_segments
    ]
    
    return {
        "auditory_results": results,
        "segments": segment_ranges  
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python script.py <오디오_파일_경로>")
    else:
        analyze_audio(sys.argv[1])