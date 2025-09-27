# 🌐 Public Data Sources for SafeDriving Project

## 📊 현재 활용 중인 공개 데이터

### 1. Kaggle Safe Driving 데이터셋
- **출처**: [Porto Seguro's Safe Driver Prediction](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)
- **데이터 규모**: ~595,212 샘플, 57개 피처
- **주요 피처**: 운전자 정보, 차량 정보, 보험 이력 및 클레임 데이터
- **활용 현황**: ✅ 이미 통합됨 (`src/data/data_loader.py`)

### 2. 스마트폰 센서 데이터
- **출처**: Android Sensor Framework, iOS Core Motion
- **제공 데이터**: 가속도계(3축), 자이로스코프(3축), GPS 위치 및 속도
- **활용 현황**: ✅ 이미 통합됨 (합성 데이터 생성)

---

## 🚀 실제 활용 예정 공개 데이터

### 3. Kaggle 교통사고 데이터셋
- **출처**: [Road Traffic Accidents](https://www.kaggle.com/datasets/sohier/us-accidents-dataset)
- **제공 데이터**: 미국 교통사고 데이터 (300만+ 건)
- **주요 피처**: 사고 위치, 시간, 기상 조건, 도로 유형, 심각도
- **활용 방안**: 위험 지역 및 조건별 가중치 계산

### 4. Kaggle 차량 센서 데이터셋
- **출처**: [Vehicle Sensor Data](https://www.kaggle.com/datasets/sobhanmoosavi/vehicle-sensor-data)
- **제공 데이터**: 실제 차량 센서 로그 데이터
- **주요 피처**: 가속도, 속도, GPS, 브레이크 압력, 엔진 상태
- **활용 방안**: 운전 패턴 분석 및 위험 행동 탐지

### 5. Kaggle 운전자 행동 데이터셋
- **출처**: [Driver Behavior Analysis](https://www.kaggle.com/datasets/outofskills/driving-behavior)
- **제공 데이터**: 운전자별 행동 패턴 분류 데이터
- **주요 피처**: 급가속, 급정지, 급회전, 속도 위반 패턴
- **활용 방안**: 안전 점수 예측 모델 학습

---

## 🔄 데이터 활용 전략

### 교통사고 데이터 활용 (Kaggle)
- **위험도 매핑**: 사고 다발 지역 및 시간대 분석
- **기상 조건**: 사고 데이터 내 기상 정보로 위험도 가중치 계산
- **도로 유형별 분석**: 고속도로, 시내도로별 사고 패턴 학습

### 차량 센서 데이터 활용 (Kaggle)
- **실제 센서 패턴**: 브레이크 압력, 엔진 상태 등 추가 피처 활용
- **운전 스타일 분류**: 실제 차량 데이터로 운전 패턴 검증
- **이상 상황 탐지**: 엔진 이상, 급제동 등 위험 상황 감지

### 운전자 행동 데이터 활용 (Kaggle)
- **행동 패턴 학습**: 급가속, 급정지 등 위험 행동 분류 모델
- **안전 점수 보정**: 실제 운전자 행동 데이터로 점수 알고리즘 개선
- **예측 모델 검증**: 기존 모델의 정확도 검증 및 튜닝

---

*문서 작성일: 2025-09-27*