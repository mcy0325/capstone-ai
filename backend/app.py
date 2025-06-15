from flask import Flask, request, jsonify
from flask_cors import CORS 
import os
from visual import analyze_visual
from auditory import analyze_yamnet_only
from auditory import analyze_full_audio
from moviepy.editor import VideoFileClip

app = Flask(__name__)
CORS(app)

def extract_audio(video_path, output_audio_path="./temp_audio.wav"):
    video = VideoFileClip(video_path)
    audio = video.audio
    if audio:
        audio.write_audiofile(output_audio_path)
    else:
        print("⚠️ 오디오가 없는 비디오입니다.")

@app.route("/analyze", methods=["POST"])
def analyze():
    video = request.files.get("video")
    if not video:
        return jsonify({"error": "No video uploaded"}), 400

    video_path = "./temp_video.mp4"
    audio_path = "./temp_audio.wav"
    video.save(video_path)
    
    # 오디오 추출
    extract_audio(video_path, output_audio_path=audio_path)

    # 오디오 분석
    attention_array = analyze_yamnet_only(audio_path)   
    
    # attention 확인 
    attention_result = db_attention_api(attention_array)
    
    if attention_result:
        audio_result=analyze_full_audio(audio_path)
        audio_segments = audio_result.get("segments", [])
        

        # 비주얼 분석 (오디오 구간 기반)
        visual_result, frames = analyze_visual(video_path, segments=audio_segments)

        # 정리된 JSON 결과 반환
        return jsonify({
            "visual": visual_result,
            "auditory": audio_result.get("results", [])
        })
    else:
         return jsonify({
            "visual": "attention 되지 않음",
            "auditory": "attention 되지 않음"
        })
    
def db_attention_api(attention_array):
    if attention_array:
        print("attention 감지됨")
        return True
    else:
        print("attnetion 감지되지 않음")
        return False

if __name__ == "__main__":
    app.run(debug=True)
