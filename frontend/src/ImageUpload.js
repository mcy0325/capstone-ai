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
    <div style={{ padding: "2rem", maxWidth: "700px", margin: "auto" }}>
      <h2>시각 정보 분석 도구</h2>
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
            <strong>특징점 수:</strong> {result.features}
          </p>

          <div>
            <strong>색상 정보:</strong>
            <pre>{JSON.stringify(result.colors, null, 2)}</pre>
          </div>

          <div>
            <strong>추출된 특징 키워드:</strong>
            <ul>
              {result.description_features.map((kw, i) => (
                <li key={i}>{kw}</li>
              ))}
            </ul>
          </div>

          <div>
            <strong>활성화된 뇌 영역:</strong>
            <ul>
              {result.brain_regions.map((region, i) => (
                <li key={i}>{region}</li>
              ))}
            </ul>
          </div>

          <div>
            <strong>탐지된 객체 (YOLO):</strong>
            <ul>
              {result.detected_parts.map((obj, i) => (
                <li key={i}>
                  {obj.label} – {(obj.confidence * 100).toFixed(1)}%
                </li>
              ))}
            </ul>
          </div>

          {result.edge_image_url && (
            <div>
              <strong>엣지 감지 결과:</strong>
              <img
                src={`http://localhost:5000/${result.edge_image_url}`}
                alt="Edge Detection"
                style={{ marginTop: "1rem", width: "100%" }}
              />
            </div>
          )}

          {result.segmentation_mask_url && (
            <div>
              <strong>Segmentation 결과:</strong>
              <img
                src={`http://localhost:5000/${result.segmentation_mask_url}`}
                alt="Segmentation Mask"
                style={{ marginTop: "1rem", width: "100%" }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ImageUpload;
