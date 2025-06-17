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
    return [(yamnet_class_names[i], float(mean_scores[i])) for i in top_indices]


def claude_api_call(prompt):
    message = anthropic.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=1000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def load_audio(file_path):
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError("유효한 파일 경로가 아닙니다.")
    return librosa.load(file_path, sr=None)


def detect_raw_intervals(y, sr, hop_length=512, n_fft=2048, min_duration=0.3):
    energy = np.array([
        np.sum(np.abs(y[i:i + n_fft]) ** 2)
        for i in range(0, len(y), hop_length)
    ])
    threshold = np.percentile(energy, 75)
    mask = energy > threshold
    frames = np.where(mask)[0]
    intervals = []
    for _, group in groupby(enumerate(frames), lambda x: x[0] - x[1]):
        frames_group = list(map(itemgetter(1), group))
        start = frames_group[0] * hop_length
        end = min((frames_group[-1] + 1) * hop_length, len(y))
        if (end - start) / sr >= min_duration:
            intervals.append((start, end))
    return intervals


def extract_features(y, sr, start=None, end=None, hop_length=512):
    segment = y if start is None or end is None else y[start:end]
    return {
        "zcr": float(np.mean(librosa.feature.zero_crossing_rate(y=segment, hop_length=hop_length)[0])),  # zero-crossing rate
        "centroid": float(np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr, hop_length=hop_length)[0])),
        "bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr, hop_length=hop_length)[0])),  # spectral bandwidth
        "rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr, hop_length=hop_length)[0])),  # spectral rolloff
        "mfcc": np.mean(librosa.feature.mfcc(y=segment, sr=sr, hop_length=hop_length, n_mfcc=13), axis=1).round(2).tolist()
    }


def extract_background_features(y, sr, exclude_segments, hop_length=512):
    mask = np.ones(len(y), dtype=bool)
    for start_sec, end_sec in exclude_segments:
        mask[int(start_sec*sr):int(end_sec*sr)] = False
    return extract_features(y[mask], sr, hop_length=hop_length)


def build_comparative_prompt(background, focus, start_sec, end_sec, top_predictions):
    pred_text = "\n".join([f"- {label}: {score:.2%}" for label, score in top_predictions])
    return f"""
전체 오디오에서 특정 구간({start_sec:.2f}s ~ {end_sec:.2f}s)을 제외한 배경 소리의 특징:
- 평균 MFCC: {background['mfcc']}
- ZCR: {background['zcr']:.5f}
- Spectral Centroid: {background['centroid']:.2f} Hz
- Spectral Bandwidth: {background['bandwidth']:.2f} Hz
- Spectral Roll-off: {background['rolloff']:.2f} Hz

특정 구간({start_sec:.2f}s ~ {end_sec:.2f}s)의 특징:
- 평균 MFCC: {focus['mfcc']}
- ZCR: {focus['zcr']:.5f}
- Spectral Centroid: {focus['centroid']:.2f} Hz
- Spectral Bandwidth: {focus['bandwidth']:.2f} Hz
- Spectral Roll-off: {focus['rolloff']:.2f} Hz

YAMNet 상위 5개 후보:
{pred_text}
"""


def save_segment_graphs(y, sr, start, end, idx, output_dir="static"):
    segment = y[start:end]
    hop_length = 512
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, hop_length=hop_length, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y=segment, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr, hop_length=hop_length)[0]

    # feature 프레임 수에 맞춰 time axis 계산
    times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=hop_length)

    # 그리기
    librosa.display.waveshow(segment, sr=sr, ax=axs[0][0])
    axs[0][0].set_title('Waveform')
    img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=hop_length, ax=axs[0][1])
    axs[0][1].set_title('MFCC')
    fig.colorbar(img, ax=axs[0][1])
    axs[1][0].plot(times, zcr)
    axs[1][0].set_title('ZCR')
    axs[1][1].plot(times, centroid)
    axs[1][1].set_title('Centroid')
    axs[2][0].plot(times, bandwidth)
    axs[2][0].set_title('Bandwidth')
    axs[2][1].plot(times, rolloff)
    axs[2][1].set_title('Rolloff')
    
    output_path = os.path.join(output_dir, f"segment_{idx}.png")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    
    return output_path.replace("\\", "/")


def merge_similar_segments(segments, sr, max_gap=0.4, centroid_thresh=300, zcr_thresh=0.05):
    merged, curr = [], segments[0].copy()
    for seg in segments[1:]:
        time_gap = (seg['start'] - curr['end'])/sr
        if time_gap<=max_gap and abs(seg['centroid']-curr['centroid'])<=centroid_thresh and abs(seg['zcr']-curr['zcr'])<=zcr_thresh:
            curr['end']=seg['end']
        else:
            merged.append((curr['start'], curr['end'])); curr=seg.copy()
    merged.append((curr['start'], curr['end']))
    return merged


def analyze_yamnet_only(file_path):
    y, sr = load_audio(file_path)
    raw_intervals = detect_raw_intervals(y, sr)
    if not raw_intervals:
        print("의미 있는 구간이 탐지되지 않았습니다."); return []
    feats = [extract_features(y, sr, s, e) for s,e in raw_intervals]
    merged = merge_similar_segments(
        [dict(start=s, end=e, **feat) for (s,e), feat in zip(raw_intervals, feats)], sr
    )
    return classify_yamnet_segments(y, sr, merged)

def classify_yamnet_segments(y, sr, intervals):
    """
    각 (start, end) 구간을 YAMNet으로 분류하고
    top_predictions 리스트를 반환합니다.
    """
    predictions = []
    for start, end in intervals:
        segment = y[start:end]
        top_preds = classify_with_yamnet(segment, sr)
        predictions.append(top_preds)
    return predictions


def analyze_full_audio(file_path):
    y, sr = load_audio(file_path)
    raw_intervals = detect_raw_intervals(y, sr)
    feats = [extract_features(y, sr, s, e) for s,e in raw_intervals]
    merged = merge_similar_segments([dict(start=s,end=e,
        centroid=feat['centroid'], zcr=feat['zcr']) for (s,e),feat in zip(raw_intervals,feats)
    ], sr)
    results=[]
    for idx,(s,e) in enumerate(merged):
        bg = extract_background_features(y, sr, [(s/sr,e/sr)])
        focus = extract_features(y, sr, s, e)
        preds = classify_with_yamnet(y[s:e], sr)
        prompt = build_comparative_prompt(bg, focus, s/sr, e/sr, preds)
        resp = claude_api_call(prompt)
        graph_url = save_segment_graphs(y, sr, s, e, idx)
        results.append({
            "graph_url": graph_url,
            'segments':idx+1,
            'start':s/sr,'end':e/sr,
            'focus':focus,'background':bg,
            'yamnet_preds':preds,'prompt':prompt,'response':resp
        })
    segment_ranges = [ {"start": s/sr + 2.0, "end": e/sr + 2.0} for s, e in merged ]
    return {
        "results": results,
        "segments": segment_ranges,
    }
