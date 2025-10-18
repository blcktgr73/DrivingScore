# Phase 4G 계획서 (Kaggle 실제 사고 데이터 통합)

## 📋 프로젝트 개요

**목표**: Kaggle 실제 사고 데이터를 활용하여 고품질의 Combined Dataset 20K를 생성하고, 향상된 예측 모델을 개발한다.

**주요 개선점**:
- Phase 4F: 100% 시뮬레이션 데이터 (실제 사고 없음)
- Phase 4G: **Kaggle 실제 사고 데이터 매칭** (높은 신뢰도)

---

## 🎯 핵심 목표

### 1. 데이터 품질 향상
- **목표 라벨 정확도**: 70-80% → 85-90%
- **매칭 기준 강화**: 거리 50km 이내, 시간 ±3일, 도시 필수 일치
- **Risk:Safe 사고 비율**: 4:1 (실제 연구 기반 3-5배)

### 2. MDPI 연구 기반 이벤트 생성
- 급가속/급정거: MDPI k-means 분석 결과 적용
- 급회전: 가상 생성 (명확한 근거 명시)
- 과속: Phase 4F 기준 유지

### 3. 모델 성능 개선
- LR 모델 + Class Weight + Threshold 조정
- Voting Ensemble (LR + RF + GBM)
- Scenario A (과속 고려) vs B (과속 미고려)

---

## 📊 데이터 생성 사양

### Combined Dataset 목표
- **총 데이터 수**: 20,000개
- **Risk:Safe 비율**: 25:75 (CMT White Paper 기준)
  - Risk group: 5,000개 (25%)
  - Safe group: 15,000개 (75%)
- **사고 비율**: Risk 그룹이 Safe 그룹보다 4배 높음
  - Risk group 사고율: ~16%
  - Safe group 사고율: ~4%
  - 전체 사고율: ~7% (Kaggle 데이터 기반)

### Kaggle 사고 데이터 매칭 기준

| 항목 | Phase 4F | Phase 4G | 변경 이유 |
|------|----------|----------|-----------|
| **거리** | 100km 이내 | **50km 이내** | 매칭 품질 향상 |
| **시간** | ±7일 | **±3일** | 시간적 연관성 강화 |
| **도시** | 선택적 | **필수 일치** | 지역적 특성 반영 |
| **예상 정확도** | 70-80% | **85-90%** | 고품질 레이블링 |

### 오버샘플링 방지
- 동일한 Kaggle 사고 데이터를 중복 매칭하지 않음
- 각 사고 데이터는 최대 1회만 사용
- 실제 매칭된 비율을 명시적으로 기록

---

## 🚗 이벤트 생성 방법론

### 1. 급가속/급정거 (MDPI 연구 기반)

**출처**: `docs/Phase4G_MDPI_Harsh_Driving_Events_Study.md`

#### K-means 클러스터링 결과
- **데이터셋**: 356,162 trips
- **그룹 분류**:
  - Non-Dangerous: 93% (평균 급가속 11.95/100km, 급정거 16.39/100km)
  - Dangerous: 7% (위험 임계값 급가속 48.82/100km, 급정거 45.40/100km)

#### Phase 4G 적용 방법

**Safe 그룹 (75%)**:
```
급가속 기본값 = 11.95 × 0.65 = 7.77 events/100km
급정거 기본값 = 16.39 × 0.65 = 10.65 events/100km
표준편차: 급가속 27.86, 급정거 29.76
```

**Risk 그룹 (25%)**:
```
급가속 기본값 = 48.82 × 0.85 = 41.50 events/100km
급정거 기본값 = 45.40 × 0.85 = 38.59 events/100km
표준편차: 급가속 27.86, 급정거 29.76
```

**야간 보정**:
```
야간 운전 (18:00-06:00): 기본값 × 1.5
주간 운전 (06:00-18:00): 기본값 × 1.0
```

**거리 스케일링**:
```python
이벤트_횟수 = int((기본값/100km × 야간배율 × 정규분포) × 실제거리km)
```

### 2. 급회전 (가상 생성)

**생성 근거**: 공개된 급회전 통계 부재, 급정거와 상관관계 가정

**생성 로직**:
```
급회전 = 급정거 × 랜덤(0.3~0.5)

이유:
1. 급정거 시 급회전이 동반될 가능성 높음
2. 급회전은 급정거보다 덜 빈번함
3. 비율 범위: 30-50% (보수적 추정)
```

**그룹별 기대값**:
```
Safe 그룹:
- 급정거: 10.65/100km
- 급회전: 3.2~5.3/100km (평균 4.25)

Risk 그룹:
- 급정거: 38.59/100km
- 급회전: 11.6~19.3/100km (평균 15.4)
```

### 3. 과속 (Phase 4F 기준 유지)

**생성 비율**:
```
Safe 그룹: 5-10% 확률
Risk 그룹: 30-50% 확률

과속 횟수 (발생 시):
Safe: 1-3회
Risk: 5-15회
```

---

## 🔬 데이터 생성 프로세스

### Phase 1: Kaggle 사고 데이터 로드
```python
# 데이터 출처: data/us_accidents/US_Accidents_March23.csv
# 레코드 수: ~7.7M
# 사용 컬럼: City, Severity, Start_Time, Start_Lat, Start_Lng, Weather_Condition
```

### Phase 2: 시뮬레이션 데이터 생성
```python
# 20,000개 trip 생성
# - Risk: 5,000 (25%)
# - Safe: 15,000 (75%)
# - 거리: 5-200km (정규분포)
# - 시간대: 주간 60%, 야간 40%
# - 도시: Kaggle 데이터의 상위 50개 도시
```

### Phase 3: 사고 데이터 매칭
```python
# 매칭 알고리즘:
# 1. 도시 필수 일치
# 2. 시간 차이 ±3일 이내
# 3. 거리 50km 이내 (Haversine 공식)
# 4. Risk 그룹 우선 매칭 (높은 사고율 반영)
# 5. 매칭 실패 시 사고 없음으로 기록
```

### Phase 4: 이벤트 생성
```python
# MDPI 기반:
# - 급가속: generate_harsh_accel(driver_type, distance, is_night)
# - 급정거: generate_harsh_brake(driver_type, distance, is_night)

# 가상 생성:
# - 급회전: int(급정거 × random.uniform(0.3, 0.5))
# - 과속: generate_speeding(driver_type, distance)
```

### Phase 5: 검증
```python
# 1. Risk:Safe 사고 비율 = 4:1 검증
# 2. 평균 급가속/급정거가 MDPI 기대값과 일치하는지 확인
# 3. 오버샘플링 체크 (동일 사고 ID 중복 확인)
```

---

## 🧪 모델 학습 및 평가

### 모델 1: Logistic Regression + Class Weight + Threshold 조정

**특징**:
- 단순하고 해석 가능한 모델
- Class imbalance 해결: `class_weight='balanced'`
- Threshold 조정: 0.5 → 최적값 탐색 (Recall 극대화)

**하이퍼파라미터**:
```python
LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

# Threshold 범위: 0.3 ~ 0.7 (0.05 간격)
```

### 모델 2: Voting Ensemble (LR + RF + GBM)

**특징**:
- 다양한 알고리즘의 강점 결합
- Soft voting (확률 평균)
- 견고성 향상

**구성 모델**:
```python
1. Logistic Regression (선형 관계)
   - class_weight='balanced'

2. Random Forest (비선형 관계, 변수 상호작용)
   - n_estimators=100
   - max_depth=10
   - class_weight='balanced'

3. Gradient Boosting (순차적 학습)
   - n_estimators=100
   - max_depth=5
   - learning_rate=0.1
```

### Scenario A: 과속 고려

**특징 가중치**:
```python
# 모든 이벤트 포함
features = [
    'Rapid_Accel_Count',    # 급가속
    'Sudden_Stop_Count',    # 급정거
    'Sharp_Turn_Count',     # 급회전
    'Speeding_Count',       # 과속
    'Distance_km',          # 거리
    'Is_Night'              # 야간 여부
]

# 가중치 (상대적 중요도):
weights = {
    'Speeding_Count': 2.0,       # 과속 (가장 위험)
    'Sudden_Stop_Count': 1.5,    # 급정거
    'Sharp_Turn_Count': 1.3,     # 급회전
    'Rapid_Accel_Count': 1.2,    # 급가속
    'Is_Night': 1.5,             # 야간
    'Distance_km': 1.0           # 거리 (노출)
}
```

### Scenario B: 과속 미고려

**특징 가중치**:
```python
# 과속 제외 (센서 기반만)
features = [
    'Rapid_Accel_Count',    # 급가속
    'Sudden_Stop_Count',    # 급정거
    'Sharp_Turn_Count',     # 급회전
    'Distance_km',          # 거리
    'Is_Night'              # 야간 여부
]

# 가중치 (상대적 중요도):
weights = {
    'Sudden_Stop_Count': 2.0,    # 급정거 (가장 중요)
    'Sharp_Turn_Count': 1.5,     # 급회전
    'Rapid_Accel_Count': 1.3,    # 급가속
    'Is_Night': 1.5,             # 야간
    'Distance_km': 1.0           # 거리 (노출)
}
```

**비교 목적**:
- GPS 기반 과속 데이터가 없는 경우 성능 평가
- 순수 센서 기반 (가속도계) 예측 가능성 검증

---

## 📈 평가 지표

### 주요 메트릭
1. **Recall (재현율)**: 실제 사고 중 예측 성공률 (가장 중요!)
2. **Precision (정밀도)**: 사고 예측 중 실제 사고 비율
3. **F1-Score**: Precision과 Recall의 조화 평균
4. **Accuracy (정확도)**: 전체 예측 정확도
5. **AUC-ROC**: 모델의 전반적 분류 성능

### 목표 성능
```
Phase 4F 결과:
- Recall: 67-68%
- Precision: 50-55%
- F1-Score: 57-60%

Phase 4G 목표:
- Recall: 75-80% (↑ 10-12%p)
- Precision: 60-65% (↑ 10%p)
- F1-Score: 67-72% (↑ 10%p)

개선 근거: 고품질 Kaggle 실제 사고 데이터 + MDPI 연구 기반 이벤트
```

---

## 🗂️ 산출물

### 1. Phase4G_Plan.md (현재 문서)
- 프로젝트 계획 및 방법론

### 2. Phase4G_Data_Sample_Report.md
- 생성된 20K 데이터 통계 및 분석
- Risk/Safe 그룹별 사고 비율 검증
- 실제 샘플 예시 (4가지 케이스)
- 오버샘플링 검증 결과

### 3. Phase4G_Final_Report.md
- 모델 학습 결과 (LR, Voting Ensemble)
- Scenario A vs B 비교 분석
- Phase 4F vs Phase 4G 성능 비교
- Feature Importance 분석
- 결론 및 향후 개선 방향

### 4. research/phase4g_*.py
- `phase4g_step1_data_generation.py`: 20K 데이터 생성
- `phase4g_step2_data_report.py`: 데이터 샘플 리포트 작성
- `phase4g_step3_model_training.py`: 모델 학습 및 평가
- `phase4g_step4_final_report.py`: 최종 결과 리포트 작성

### 5. research/phase4g_combined_20k.json
- 생성된 20,000개 Combined Dataset

---

## 📅 실행 계획

### Step 1: 데이터 생성 (예상 시간: 1-2시간)
- Kaggle 사고 데이터 로드
- 20K 시뮬레이션 데이터 생성
- MDPI 기반 이벤트 생성
- 사고 데이터 매칭 (50km, ±3일, 도시 일치)
- JSON 저장

### Step 2: 데이터 분석 및 리포트 (예상 시간: 30분)
- 통계 분석
- Risk:Safe 사고 비율 검증 (4:1)
- 샘플 추출 및 포맷팅
- Markdown 리포트 작성

### Step 3: 모델 학습 (예상 시간: 1-2시간)
- LR + Class Weight + Threshold 조정
- Voting Ensemble (LR+RF+GBM)
- Scenario A (과속 포함) 학습
- Scenario B (과속 제외) 학습
- Cross-validation

### Step 4: 최종 리포트 작성 (예상 시간: 30분)
- 결과 정리
- 시각화 (혼동 행렬, ROC 곡선, Feature Importance)
- Phase 4F 대비 개선률 분석
- Markdown 리포트 작성

**총 예상 시간**: 3-5시간

---

## 🎓 기대 효과

### 1. 데이터 품질
- ✅ 실제 사고 데이터 기반 → 라벨 정확도 85-90%
- ✅ 엄격한 매칭 기준 → 높은 신뢰도
- ✅ MDPI 연구 기반 이벤트 → 현실적 분포

### 2. 모델 성능
- ✅ Recall 75-80% → 실제 사고 감지율 향상
- ✅ Ensemble 학습 → 견고성 증가
- ✅ Threshold 조정 → Recall/Precision 균형

### 3. 실용성
- ✅ 과속 유무에 따른 시나리오 분석
- ✅ 센서 기반 예측 가능성 검증
- ✅ Risk:Safe 4:1 비율 → 실제 연구와 일치

---

**작성자**: Claude Code
**작성일**: 2025-10-17
**버전**: 1.0
**프로젝트**: DrivingScore Phase 4G
