import React, { useState } from "react";
import axios from "axios";

function ImageUpload() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return;

    const formData = new FormData();
    formData.append("image", image);

    try {
      const res = await axios.post("http://localhost:5000/analyze", formData);
      setResult(res.data);
    } catch (error) {
      console.error("분석 중 오류 발생:", error);
    }
  };

  return (
    <div style={{ padding: "2rem", maxWidth: "600px", margin: "auto" }}>
      <h2>감정 분석 보조 도구</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleImageChange} />
        <button type="submit" style={{ marginTop: "1rem" }}>
          이미지 분석
        </button>
      </form>

      {preview && (
        <img
          src={preview}
          alt="preview"
          style={{ marginTop: "1rem", width: "100%" }}
        />
      )}

      {result && (
        <div style={{ marginTop: "2rem" }}>
          <h3>분석 결과</h3>
          <p>
            <strong>이미지 설명:</strong> {result.caption}
          </p>
          <p>
            <strong>감정 추론:</strong> {result.emotion[0].label} (
            {(result.emotion[0].score * 100).toFixed(2)}%)
          </p>
          <p>
            <strong>ResNet 분류 ID:</strong> {result.classification}
          </p>
          <p>
            <strong>특징점 수:</strong> {result.features}
          </p>
          <div>
            <strong>색상 정보:</strong>
            <pre>{JSON.stringify(result.colors, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default ImageUpload;
