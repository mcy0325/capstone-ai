import React, { useState, useRef, useEffect } from "react";
import axios from "axios";

function ImageUpload() {
  const [image, setImage] = useState(null);
  const [image2, setImage2] = useState(null);
  const [video, setVideo] = useState(null);
  const [result, setResult] = useState(null);
  const [flowResult, setFlowResult] = useState(null);
  const [videoResult, setVideoResult] = useState(null);

  const canvasRef = useRef(null);

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handleImage2Change = (e) => {
    setImage2(e.target.files[0]);
  };

  const handleVideoChange = (e) => {
    setVideo(e.target.files[0]);
  };

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

  // 시각화 예시 (간단한 magnitude-based 색상 표시)
  useEffect(() => {
    if (flowResult && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const magnitude = flowResult.optical_flow?.average_magnitude || 0;
      const angle = flowResult.optical_flow?.average_angle || 0;

      // 간단한 원 시각화
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
          <h4>이미지 분석 결과</h4>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}

      {flowResult && (
        <div style={{ marginTop: "2rem" }}>
          <h4>Optical Flow 분석 결과 (이미지 쌍)</h4>
          <pre>{JSON.stringify(flowResult, null, 2)}</pre>
          <canvas
            ref={canvasRef}
            width={200}
            height={200}
            style={{ border: "1px solid black" }}
          />
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
