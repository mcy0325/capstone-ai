import React, { useState } from "react";
import axios from "axios";

function ImageUpload() {
  const [video, setVideo] = useState(null);
  const [videoResult, setVideoResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleVideoChange = (e) => {
    setVideo(e.target.files[0]);
  };

  const handleSubmitVideo = async (e) => {
    e.preventDefault();
    if (!video) return;
    const formData = new FormData();
    formData.append("video", video);
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/analyze", formData);
      setVideoResult(res.data);
    } catch (err) {
      console.error("영상 분석 실패:", err);
      alert("영상 분석 실패");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h2>🎥 비디오 시각 분석 도구</h2>
      {loading && <p style={{ color: "blue" }}>🌀 분석 중입니다...</p>}

      <form onSubmit={handleSubmitVideo} style={{ marginTop: "1rem" }}>
        <h3>비디오 업로드</h3>
        <input type="file" accept="video/*" onChange={handleVideoChange} />
        <button type="submit">분석 시작</button>
      </form>

      {videoResult && (
        <div style={{ marginTop: "2rem" }}>
          <h4>📊 프레임 단위 분석 결과</h4>
          {videoResult.visual?.frame_analysis?.map((frame, index) => (
            <div
              key={index}
              style={{
                marginBottom: "2rem",
                border: "1px solid gray",
                padding: "1rem",
              }}
            >
              <h5>프레임 #{frame.frame_index}</h5>
              <p>
                <strong>Caption:</strong> {frame.caption}
              </p>
              <p>
                <strong>키워드:</strong>{" "}
                {frame.description_features?.join(", ")}
              </p>
              <p>
                <strong>객체 수:</strong> {frame.object_count}
              </p>
              <p>
                <strong>객체 목록:</strong> {frame.detected_parts?.join(", ")}
              </p>
              <p>
                <strong>활성 뇌 영역:</strong> {frame.brain_regions?.join(", ")}
              </p>
              <p>
                <strong>색상 분석:</strong> {JSON.stringify(frame.colors)}
              </p>
              <p>
                <strong>대표 색상:</strong>
                <span
                  style={{
                    display: "inline-block",
                    width: "20px",
                    height: "20px",
                    backgroundColor: `rgb(${frame.dominant_color?.join(",")})`,
                    marginLeft: "0.5rem",
                    border: "1px solid #000",
                  }}
                />
                <span style={{ marginLeft: "0.5rem" }}>
                  ({frame.dominant_color?.map((v) => Math.round(v)).join(", ")})
                </span>
              </p>

              <p>
                <strong>Dominant Object Colors:</strong>
              </p>
              <div
                style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}
              >
                {frame.dominant_color_object_list?.map((color, idx) => (
                  <div key={idx} style={{ textAlign: "center" }}>
                    <div
                      style={{
                        width: "30px",
                        height: "30px",
                        backgroundColor: `rgb(${color
                          .map((v) => Math.round(v))
                          .join(",")})`,
                        border: "1px solid #000",
                      }}
                    />
                    <small>{color.map((v) => Math.round(v)).join(",")}</small>
                  </div>
                ))}
              </div>

              <p>
                <strong>Colorfulness:</strong> {frame.colorfulness?.toFixed(2)}
              </p>
              <p>
                <strong>Aspect Ratio:</strong> {frame.aspect_ratio?.toFixed(2)}
              </p>
              <p>
                <strong>Blur:</strong> {frame.blur?.toFixed(2)}
              </p>
              <p>
                <strong>Entropy:</strong> {frame.entropy?.toFixed(2)}
              </p>
              <p>
                <strong>Symmetry:</strong> {frame.symmetry?.toFixed(2)}
              </p>
              <p>
                <strong>밝은 영역 비율:</strong>{" "}
                {(frame.bright_region_ratio * 100).toFixed(2)}%
              </p>
              <p>
                <strong>Edge 밀도:</strong> {frame.edge_density}
              </p>
              <p>
                <strong>GLCM Contrast:</strong>{" "}
                {frame.glcm_contrast?.toFixed(2)}
              </p>
              <p>
                <strong>마스크 영역 비율:</strong>{" "}
                {(frame.mask_area_ratio * 100).toFixed(2)}%
              </p>
              <p>
                <strong>대비:</strong> {frame.contrast?.toFixed(2)}
              </p>
              <p>
                <strong>HOG 요약:</strong>{" "}
                {frame.hog_summary?.slice(0, 10).join(", ")} ...
              </p>
              <p>
                <strong>텍스처 히스토그램:</strong>{" "}
                {frame.texture_histogram?.slice(0, 10).join(", ")} ...
              </p>
              <p>
                <strong>추출된 특징:</strong> {JSON.stringify(frame.features)}
              </p>

              {/* ✅ Segmentation & Edge 이미지 */}
              <div style={{ display: "flex", gap: "1rem", marginTop: "1rem" }}>
                <div>
                  <p>
                    <strong>Segmentation</strong>
                  </p>
                  <img
                    src={`http://localhost:5000/${frame.segmentation_mask_url}`}
                    alt="Segmentation Mask"
                    width="300"
                  />
                </div>
                <div>
                  <p>
                    <strong>Edge</strong>
                  </p>
                  <img
                    src={`http://localhost:5000/${frame.edge_image_url}`}
                    alt="Edge Detection"
                    width="300"
                  />
                </div>
              </div>
            </div>
          ))}

          <h4>⚡ Optical Flow 분석</h4>
          {videoResult.visual?.optical_flow?.map((flow, index) => (
            <div
              key={index}
              style={{
                marginBottom: "1rem",
                borderTop: "1px dashed #aaa",
                paddingTop: "0.5rem",
              }}
            >
              <p>
                프레임 {flow.between_frames[0]} → {flow.between_frames[1]}
              </p>
              <p>평균 속도: {flow.average_magnitude?.toFixed(2)}</p>
              <p>평균 방향: {flow.average_angle?.toFixed(2)} rad</p>
              <p>해석: {flow.motion_label}</p>
            </div>
          ))}

          {/* 🔈 청각 분석 결과 */}
          {videoResult.auditory && (
            <div style={{ marginTop: "2rem" }}>
              <h4>🔈 청각 분석 결과</h4>
              {videoResult.auditory?.auditory_results?.map((segment, index) => (
                <div
                  key={index}
                  style={{
                    marginBottom: "2rem",
                    border: "1px solid #ccc",
                    padding: "1rem",
                    borderRadius: "8px",
                  }}
                >
                  <h5>🎧 구간 {segment.segment}</h5>
                  <p>
                    <strong>시간:</strong> {segment.start_sec}s ~{" "}
                    {segment.end_sec}s
                  </p>

                  <p>
                    <strong>YAMNet 상위 분류:</strong>
                  </p>
                  <ul>
                    {segment.yamnet_top_predictions?.map(
                      ([label, score], idx) => (
                        <li key={idx}>
                          {label} - {(score * 100).toFixed(2)}%
                        </li>
                      )
                    )}
                  </ul>

                  <p>
                    <strong>Claude 해석:</strong> {segment.claude_response}
                  </p>

                  <details>
                    <summary>🔍 고급 음향 특징 보기</summary>
                    <p>
                      <strong>🎯 구간 MFCC:</strong>{" "}
                      {JSON.stringify(segment.focus_features.mfcc)}
                    </p>
                    <p>
                      <strong>🎯 구간 ZCR:</strong>{" "}
                      {segment.focus_features.zcr.toFixed(5)}
                    </p>
                    <p>
                      <strong>🎯 구간 Spectral Centroid:</strong>{" "}
                      {segment.focus_features.centroid.toFixed(2)} Hz
                    </p>
                    <p>
                      <strong>🎯 구간 Spectral Bandwidth:</strong>{" "}
                      {segment.focus_features.bandwidth.toFixed(2)} Hz
                    </p>
                    <p>
                      <strong>🎯 구간 Spectral Rolloff:</strong>{" "}
                      {segment.focus_features.rolloff.toFixed(2)} Hz
                    </p>

                    <p>
                      <strong>🎧 배경 MFCC:</strong>{" "}
                      {JSON.stringify(segment.background_features.mfcc)}
                    </p>
                    <p>
                      <strong>🎧 배경 ZCR:</strong>{" "}
                      {segment.background_features.zcr.toFixed(5)}
                    </p>
                    <p>
                      <strong>🎧 배경 Spectral Centroid:</strong>{" "}
                      {segment.background_features.centroid.toFixed(2)} Hz
                    </p>
                    <p>
                      <strong>🎧 배경 Spectral Bandwidth:</strong>{" "}
                      {segment.background_features.bandwidth.toFixed(2)} Hz
                    </p>
                    <p>
                      <strong>🎧 배경 Spectral Rolloff:</strong>{" "}
                      {segment.background_features.rolloff.toFixed(2)} Hz
                    </p>
                  </details>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ImageUpload;
