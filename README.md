# 🧠 인공뇌 기반 시청각 인지 시스템 (Artificial Brain Simulation)

> 본 프로젝트는 인간의 인지 메커니즘을 모사하는 **인공뇌(AI Brain)** 구현을 목표로 하며, 그중에서도 **시각(Visual)** 및 **청각(Auditory)** 정보의 처리 과정을 중점적으로 개발하였습니다.



## 🔬 프로젝트 개요

이 프로젝트는 인공신경망 기반 인지 모델을 활용하여, **인간의 시각 및 청각 정보 처리 흐름을 디지털 환경에서 구현**하고자 합니다. 사용자가 업로드한 비디오에서 시청각 정보를 추출하여 **움직임, 소리, 시각적 변화** 등을 감지하고, 이를 바탕으로 **Claude API를 통해 고차원적 해석을 수행**합니다. 이는 궁극적으로 '인공 뇌'의 핵심 처리 회로 중 일부를 기능적으로 구현한 것입니다.



## 🧩 담당 영역 및 역할

| 이름 | 담당 모듈 | 주요 역할 |
|------|------------|-------------|
| 문채영 | 시각 처리 | YOLO, Optical Flow, Segmentation, 시각 특징 추출 및 해석 |
| 이서정 | 청각 처리 | YAMNet 기반 소리 감지, MFCC 분석, Claude 설명 생성 |

각 모듈은 독립적으로 동작하며, 특정 시간대 정보를 기준으로 상호보완적 해석을 생성합니다.



## 🔧 기술 구성

### 🧠 인지 시뮬레이션 흐름

```
입력 비디오
  ├─> 오디오 추출 및 청각 특징 분석 (이서정)
  ├─> 특정 시간 구간 주의 집중 여부 판단
  ├─> 해당 구간 영상에 대한 시각 분석 (문채영)
  └─> Claude API를 통한 통합 해석 생성
```



## 🧪 적용 기술 요약

- **청각 분석 (Auditory)**
  - YAMNet을 활용한 음향 이벤트 분류
  - MFCC, Spectral Features 기반 배경-포커스 비교
  - 에너지 기반 구간 분할 및 통합 설명 프롬프트 생성

- **시각 분석 (Visual)**
  - YOLOv5 객체 감지 및 Optical Flow 기반 움직임 분석
  - 이미지 캡션(BLIP), 세분화(U²-Net), 시각 피처 추출
  - 뇌 영역 매핑 및 감지 정보 요약

- **통합 분석**
  - Claude API 호출을 통한 자연어 요약 및 동작 해석
  - 움직임 정보 기반 주의 및 행동 해석



## 📌 예시 출력 (JSON 요약)

```json
{
  "auditory": [...],
  "visual": {
    "frame_analysis": [...],
    "optical_flow": [...]
  }
}
```



## 🔗 참고 자료

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [U²-Net](https://github.com/NathanUA/U-2-Net)
- [BLIP (HuggingFace)](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [YAMNet (TFHub)](https://tfhub.dev/google/yamnet/1)
- [Anthropic Claude](https://www.anthropic.com/index/claude)



> 해당 프로젝트는 시청각 기반 인공뇌의 일부분을 기능적으로 시뮬레이션하며, 인지과학, UX보안, 감정 AI 등의 다양한 분야로 확장 가능한 기반 기술을 제안합니다.
