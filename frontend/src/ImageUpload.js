import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement);

function ImageUpload() {
  const [image, setImage] = useState(null);
  const [image2, setImage2] = useState(null);
  const [video, setVideo] = useState(null);
  const [result, setResult] = useState(null);
  const [flowResult, setFlowResult] = useState(null);
  const [videoResult, setVideoResult] = useState(null);
  const [showRaw, setShowRaw] = useState(false);

  const canvasRef = useRef(null);

  const handleImageChange = (e) => setImage(e.target.files[0]);
  const handleImage2Change = (e) => setImage2(e.target.files[0]);
  const handleVideoChange = (e) => setVideo(e.target.files[0]);

  const handleSubmitImage = async (e) => {
    e.preventDefault();
    if (!image) return;
    const formData = new FormData();
    formData.append("image", image);
    const res = await axios.post("http://localhost:5000/analyze", formData);
    setResult(res.data);
  };

  const handleSubmitFlow = async (e) => {
    e.preventDefault();
    if (!image || !image2) return;
    const formData = new FormData();
    formData.append("image1", image);
    formData.append("image2", image2);
    const res = await axios.post("http://localhost:5000/opticalflow", formData);
    setFlowResult(res.data);
  };

  const handleSubmitVideo = async (e) => {
    e.preventDefault();
    if (!video) return;
    const formData = new FormData();
    formData.append("video", video);
    const res = await axios.post("http://localhost:5000/videoflow", formData);
    setVideoResult(res.data);
  };

  useEffect(() => {
    if (flowResult && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const magnitude = flowResult.optical_flow?.average_magnitude || 0;
      const angle = flowResult.optical_flow?.average_angle || 0;

      ctx.beginPath();
      ctx.arc(100, 100, magnitude * 20, 0, 2 * Math.PI);
      ctx.fillStyle = `hsl(${angle * (180 / Math.PI)}, 100%, 50%)`;
      ctx.fill();
      ctx.font = "14px Arial";
      ctx.fillStyle = "black";
      ctx.fillText(`Mag: ${magnitude.toFixed(2)}`, 10, 20);
      ctx.fillText(`Angle: ${angle.toFixed(2)}`, 10, 40);
    }
  }, [flowResult]);

  return (
    <div style={{ padding: "2rem" }}>
      <h2>시각 분석 도구</h2>

      <form onSubmit={handleSubmitImage}>
        <h3>이미지 분석</h3>
        <input type="file" accept="image/*" onChange={handleImageChange} />
        <button type="submit">분석</button>
      </form>

      <form onSubmit={handleSubmitFlow} style={{ marginTop: "1rem" }}>
        <h3>이미지 Optical Flow</h3>
        <input type="file" accept="image/*" onChange={handleImageChange} />
        <input type="file" accept="image/*" onChange={handleImage2Change} />
        <button type="submit">Flow 분석</button>
      </form>

      <form onSubmit={handleSubmitVideo} style={{ marginTop: "1rem" }}>
        <h3>비디오 Optical Flow</h3>
        <input type="file" accept="video/*" onChange={handleVideoChange} />
        <button type="submit">비디오 분석</button>
      </form>

      {result && (
        <div style={{ marginTop: "2rem" }}>
          <h4>결과 요약</h4>
          <p>Feature 개수: {result.features}</p>
          <p>Object 개수: {result.object_count}</p>
          <p>Caption: {result.caption}</p>
          <p>Keyword: {result.description_features?.join(", ")}</p>
          <p>활성 뇌영역: {result.brain_regions?.join(", ")}</p>

          <h4>수치 데이터</h4>
          <table border="1">
            <tbody>
              <tr>
                <td>평균 밝기</td>
                <td>{result.colors?.avg_brightness?.toFixed(2)}</td>
              </tr>
              <tr>
                <td>평균 채도</td>
                <td>{result.colors?.avg_saturation?.toFixed(2)}</td>
              </tr>
              <tr>
                <td>평균 색상 R</td>
                <td>{result.colors?.avg_color[0]?.toFixed(0)}</td>
              </tr>
              <tr>
                <td>평균 색상 G</td>
                <td>{result.colors?.avg_color[1]?.toFixed(0)}</td>
              </tr>
              <tr>
                <td>평균 색상 B</td>
                <td>{result.colors?.avg_color[2]?.toFixed(0)}</td>
              </tr>
              <tr>
                <td>Dominant Color R</td>
                <td>{result.dominant_color?.[0]?.toFixed(0)}</td>
              </tr>
              <tr>
                <td>Dominant Color G</td>
                <td>{result.dominant_color?.[1]?.toFixed(0)}</td>
              </tr>
              <tr>
                <td>Dominant Color B</td>
                <td>{result.dominant_color?.[2]?.toFixed(0)}</td>
              </tr>
              <tr>
                <td>Colorfulness</td>
                <td>{result.colorfulness?.toFixed(2)}</td>
              </tr>
              <tr>
                <td>Aspect Ratio</td>
                <td>{result.aspect_ratio?.toFixed(2)}</td>
              </tr>
              <tr>
                <td>Blur</td>
                <td>{result.blur?.toFixed(2)}</td>
              </tr>
              <tr>
                <td>Entropy</td>
                <td>{result.entropy?.toFixed(2)}</td>
              </tr>
              <tr>
                <td>Symmetry</td>
                <td>{result.symmetry?.toFixed(2)}</td>
              </tr>
              <tr>
                <td>Bright Region Ratio</td>
                <td>{result.bright_region_ratio?.toFixed(2)}</td>
              </tr>
              <tr>
                <td>Edge Density</td>
                <td>{result.edge_density}</td>
              </tr>
              <tr>
                <td>Contrast</td>
                <td>{result.contrast?.toFixed(2)}</td>
              </tr>
            </tbody>
          </table>

          <h4>색상 시각화</h4>
          <div style={{ display: "flex", gap: "1rem" }}>
            <div>
              <h5>평균 색상</h5>
              <div
                style={{
                  width: "50px",
                  height: "50px",
                  backgroundColor: `rgb(${result.colors?.avg_color[0]}, ${result.colors?.avg_color[1]}, ${result.colors?.avg_color[2]})`,
                  border: "1px solid black",
                }}
              ></div>
            </div>
            <div>
              <h5>Dominant Color</h5>
              <div
                style={{
                  width: "50px",
                  height: "50px",
                  backgroundColor: `rgb(${result.dominant_color?.[0]}, ${result.dominant_color?.[1]}, ${result.dominant_color?.[2]})`,
                  border: "1px solid black",
                }}
              ></div>
            </div>
            <h4>
              객체 Dominant Colors (Top{" "}
              {result.dominant_color_object_list?.length})
            </h4>
            <div style={{ display: "flex", gap: "1rem" }}>
              {result.dominant_color_object_list?.map((color, index) => (
                <div key={index}>
                  <h5>Color {index + 1}</h5>
                  <div
                    style={{
                      width: "50px",
                      height: "50px",
                      backgroundColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
                      border: "1px solid black",
                    }}
                  ></div>
                  <p>R: {color[0]?.toFixed(0)}</p>
                  <p>G: {color[1]?.toFixed(0)}</p>
                  <p>B: {color[2]?.toFixed(0)}</p>
                </div>
              ))}
            </div>
          </div>

          <h4>이미지 결과</h4>
          {result.segmentation_mask_url && (
            <div>
              <h5>Segmentation 결과</h5>
              <img
                src={`http://localhost:5000/${
                  result.segmentation_mask_url
                }?t=${Date.now()}`}
                alt="Segmentation Result"
                width="300"
              />
            </div>
          )}
          {result.edge_image_url && (
            <div>
              <h5>엣지 디텍션 결과</h5>
              <img
                src={`http://localhost:5000/${
                  result.edge_image_url
                }?t=${Date.now()}`}
                alt="Edge Detection"
                width="300"
              />
            </div>
          )}

          <h4>그래프</h4>
          {result.texture_histogram && (
            <div style={{ width: "300px" }}>
              <h5>Texture Histogram</h5>
              <Bar
                data={{
                  labels: result.texture_histogram.map((_, i) => `Bin ${i}`),
                  datasets: [
                    {
                      label: "LBP Histogram",
                      data: result.texture_histogram,
                      backgroundColor: "rgba(75,192,192,0.6)",
                    },
                  ],
                }}
              />
            </div>
          )}
          {result.hog_summary && (
            <div style={{ width: "300px" }}>
              <h5>HOG Summary (Top 20)</h5>
              <Bar
                data={{
                  labels: result.hog_summary.map((_, i) => `F${i}`),
                  datasets: [
                    {
                      label: "HOG Features",
                      data: result.hog_summary,
                      backgroundColor: "rgba(153,102,255,0.6)",
                    },
                  ],
                }}
              />
            </div>
          )}

          <h4>객체 탐지</h4>
          {result.detected_parts && (
            <table border="1">
              <thead>
                <tr>
                  <th>Label</th>
                  <th>Confidence</th>
                  <th>Bounding Box</th>
                </tr>
              </thead>
              <tbody>
                {result.detected_parts.map((obj, index) => (
                  <tr key={index}>
                    <td>{obj.label}</td>
                    <td>{obj.confidence.toFixed(2)}</td>
                    <td>{obj.bbox.map((v) => v.toFixed(1)).join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <button onClick={() => setShowRaw(!showRaw)}>
            {showRaw ? "JSON 숨기기" : "JSON 전체 보기"}
          </button>
          {showRaw && <pre>{JSON.stringify(result, null, 2)}</pre>}
        </div>
      )}

      {flowResult && (
        <div style={{ marginTop: "2rem" }}>
          <h4>Optical Flow 분석 결과 (이미지 쌍)</h4>
          {flowResult.optical_flow?.flow_image_path && (
            <div>
              <h5>Optical Flow 이미지</h5>
              <img
                src={`http://localhost:5000/${
                  flowResult.optical_flow.flow_image_path
                }?t=${Date.now()}`}
                alt="Optical Flow"
                width="300"
              />
            </div>
          )}
          <canvas
            ref={canvasRef}
            width={200}
            height={200}
            style={{ border: "1px solid black" }}
          />
          <pre>{JSON.stringify(flowResult, null, 2)}</pre>
        </div>
      )}

      {videoResult && (
        <div style={{ marginTop: "2rem" }}>
          <h4>비디오 Optical Flow 분석 결과</h4>
          <pre>{JSON.stringify(videoResult, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default ImageUpload;
