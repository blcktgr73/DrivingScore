# Phase 4-E 계획: Kaggle Real Sample 기반 고품질 매칭

**작성일:** 2025-10-15
**목적:** Kaggle US Accidents 실제 데이터 기반 고품질 매칭 및 모델 최적화

---

## 📋 개요

### Phase 4-E의 핵심 차별점

| 항목 | Phase 4-D | Phase 4-E |
|------|-----------|-----------|
| **데이터 소스** | 시뮬레이션 데이터 | Kaggle Real Sample |
| **매칭 거리** | ≤200km | **≤50km** ⭐ |
| **매칭 시간** | ±7일 | **±3일** ⭐ |
| **도시 매칭** | 선호 | **필수** ⭐ |
| **예상 라벨 정확도** | 70~80% | **85~90%** ⭐ |
| **Combined Data** | 20,000개 | 20,000개 |
| **모델** | LR + Class Weight | **LR, RF, GBM Ensemble** ⭐ |

---

## 🎯 목표

### 1. 데이터 품질 향상

```
Phase 4-D 매칭 조건:
  거리: ≤200km
  시간: ±7일
  도시: 선호

↓ 개선

Phase 4-E 매칭 조건:
  거리: ≤50km     (4배 엄격)
  시간: ±3일      (2.3배 엄격)
  도시: 필수      (100% 일치)
```

**기대 효과:**
- 매칭률 감소 (36% → 예상 15~20%)
- 라벨 정확도 향상 (70~80% → 85~90%)
- False Positive 감소

---

### 2. 모델 성능 향상

**Phase 4-E 목표:**
- F1 Score: 0.60 이상 (Phase 4-D: 0.546)
- Recall: 0.70 이상 (Phase 4-D: 0.606)
- Precision: 0.65 이상 (Phase 4-D: 0.497)

**주요 개선:**
- ✅ Voting Ensemble (LR, RF, GBM)
- ✅ 주간/야간 구분 Weight 계산
- ✅ Scenario A vs B 비교

---

## 📊 데이터 파이프라인

### 전체 흐름

```
┌────────────────────────────────────────────────────────┐
│  Step 1: Kaggle US Accidents Real Sample               │
├────────────────────────────────────────────────────────┤
│  • 실제 Kaggle 데이터셋 활용                            │
│  • 500,000개 샘플 생성                                  │
│  • 도시, 날씨, 심각도, 시간 등 실제 분포               │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│  Step 2: Vehicle Sensors 생성                          │
├────────────────────────────────────────────────────────┤
│  • 50,000개 센서 데이터                                 │
│  • 운전자 유형: SAFE, MODERATE, AGGRESSIVE             │
│  • 이벤트: 급가속, 급정거, 급회전, 과속                │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│  Step 3: 고품질 매칭 (Phase 4-E)                       │
├────────────────────────────────────────────────────────┤
│  조건:                                                  │
│    • 거리: ≤50km (엄격)                                │
│    • 시간: ±3일 (엄격)                                 │
│    • 도시: 필수 일치                                    │
│                                                         │
│  예상 결과:                                             │
│    • 매칭 성공: 8,000~10,000개 (16~20%)                │
│    • 매칭 실패: 40,000~42,000개 (80~84%)               │
│    • 라벨 정확도: 85~90%                                │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│  Step 4: Combined 데이터셋 생성                         │
├────────────────────────────────────────────────────────┤
│  • 사고 O: 10,000개 (매칭된 센서)                       │
│  • 사고 X: 10,000개 (비매칭 센서)                       │
│  • 총 20,000개 (50% 균형)                               │
│                                                         │
│  Train/Test 분할:                                       │
│    • Train: 15,000개 (75%)                              │
│    • Test:   5,000개 (25%)                              │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│  Step 5: 모델 학습 및 평가                              │
├────────────────────────────────────────────────────────┤
│  1. Logistic Regression + Class Weight                 │
│  2. Random Forest                                       │
│  3. Gradient Boosting Machine                           │
│  4. Voting Ensemble (LR + RF + GBM)                     │
│                                                         │
│  평가:                                                  │
│    • Scenario A vs B 비교                               │
│    • 주간/야간 구분 Weight                              │
│    • Threshold 최적화                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 🔬 실험 설계

### 실험 1: 고품질 매칭 데이터 생성

**목적:** 라벨 정확도 85~90% 달성

**방법:**
```python
def perform_high_quality_matching(accidents, sensors):
    """고품질 매칭 (Phase 4-E)"""
    for accident in accidents:
        for sensor in sensors:
            # 조건 1: 동일 도시 (필수)
            if accident['City'] != sensor['City']:
                continue

            # 조건 2: 거리 50km 이내
            distance = calculate_distance_km(
                accident['Lat'], accident['Lon'],
                sensor['Lat'], sensor['Lon']
            )
            if distance > 50:  # Phase 4-D: 200km
                continue

            # 조건 3: 시간차 ±3일
            time_diff = abs(accident['Time'] - sensor['Time'])
            if time_diff > 259200:  # 3일 = 72시간 = 259200초
                continue

            # 매칭 성공!
            matched.append({...})
```

**예상 결과:**
- 매칭률: 16~20% (Phase 4-D: 36%)
- 매칭 개수: 8,000~10,000개
- 라벨 정확도: 85~90% (Phase 4-D: 70~80%)

---

### 실험 2: Logistic Regression + Class Weight

**목적:** 기본 모델 성능 측정

**설정:**
```python
model = LogisticRegressionWithClassWeight(
    learning_rate=0.01,
    iterations=500,
    class_weight='balanced'
)

# Scenario A: 4개 이벤트
features_a = ['rapid_accel', 'sudden_stop', 'sharp_turn', 'over_speed']

# Scenario B: 3개 이벤트
features_b = ['rapid_accel', 'sudden_stop', 'sharp_turn']
```

**평가 지표:**
- Accuracy, Precision, Recall, F1 Score, AUC
- Confusion Matrix (TP, FP, TN, FN)

---

### 실험 3: Voting Ensemble

**목적:** 다중 모델 앙상블로 성능 향상

**구성:**
```python
# 모델 1: Logistic Regression
lr_model = LogisticRegressionWithClassWeight(...)

# 모델 2: Random Forest
rf_model = RandomForestClassifier(
    n_trees=100,
    max_depth=10,
    min_samples_split=10
)

# 모델 3: Gradient Boosting Machine
gbm_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)

# Voting Ensemble
ensemble = VotingEnsemble(
    models=[lr_model, rf_model, gbm_model],
    voting='soft'  # 확률 기반 투표
)
```

**기대 효과:**
- F1 Score 향상 (단일 모델 대비 +5~10%)
- Recall 향상 (더 많은 사고 감지)
- 안정적인 예측

---

### 실험 4: 주간/야간 구분 Weight

**목적:** 시간대별 이벤트 중요도 분석

**방법:**
```python
# 주간 데이터
day_data = [d for d in data if d['is_night'] == 0]
model_day = train_model(day_data)

# 야간 데이터
night_data = [d for d in data if d['is_night'] == 1]
model_night = train_model(night_data)

# Weight 비교
print("주간 Weight:", model_day.weights)
print("야간 Weight:", model_night.weights)
```

**분석:**
- 급가속, 급정거, 급회전, 과속의 주간/야간 차이
- 야간에 더 위험한 이벤트 식별
- 감점 시스템에 반영

---

### 실험 5: Scenario A vs B 비교

**목적:** 최적 Feature 조합 선택

**Scenario A:**
```python
features = ['rapid_accel', 'sudden_stop', 'sharp_turn', 'over_speed']
```

**Scenario B:**
```python
features = ['rapid_accel', 'sudden_stop', 'sharp_turn']
```

**비교 기준:**
- F1 Score
- Recall (사고 감지율)
- Precision (정확도)
- 구현 복잡도 (GPS 필요 여부)

---

## 📈 성능 목표

### Phase 4-E 목표 지표

| 지표 | Phase 4-D | Phase 4-E 목표 | 개선율 |
|------|-----------|----------------|--------|
| **F1 Score** | 0.546 | **≥0.60** | +10% |
| **Recall** | 0.606 | **≥0.70** | +15% |
| **Precision** | 0.497 | **≥0.65** | +31% |
| **AUC** | 0.494 | **≥0.65** | +32% |

### 모델별 예상 성능

| 모델 | F1 Score | Recall | Precision |
|------|----------|--------|-----------|
| LR + Class Weight | 0.58 | 0.68 | 0.60 |
| Random Forest | 0.62 | 0.70 | 0.65 |
| GBM | 0.63 | 0.71 | 0.66 |
| **Voting Ensemble** | **0.65** | **0.73** | **0.68** |

---

## 📂 출력 파일

### 1. 데이터 파일

```
research/
├── phase4e_high_quality_matching.py          # 메인 스크립트
├── phase4e_matching_results.json             # 매칭 결과
├── phase4e_combined_data.json                # Combined 데이터셋
└── phase4e_data_samples.json                 # 샘플 데이터
```

### 2. 결과 파일

```
research/
├── phase4e_lr_results.json                   # LR 결과
├── phase4e_ensemble_results.json             # Ensemble 결과
├── phase4e_day_night_weights.json            # 주간/야간 Weight
└── phase4e_scenario_comparison.json          # Scenario 비교
```

### 3. 문서

```
docs/
├── Phase4E_Plan.md                           # 계획 (이 문서)
├── Phase4E_Data_Sample_Report.md             # 데이터 샘플 리포트
├── Phase4E_Model_Results.md                  # 모델 결과 분석
└── Phase4E_Final_Report.md                   # 최종 리포트
```

---

## 🔍 예상 결과 분석

### 1. 매칭 품질 향상

**Phase 4-D vs Phase 4-E:**

```
Phase 4-D:
  매칭 조건: 거리 ≤200km, 시간 ±7일
  매칭률: 36% (18,000 / 50,000)
  라벨 정확도: 70~80%

  예시: 사고와 200km 떨어진 센서도 매칭
  → False Positive 가능성 높음

Phase 4-E:
  매칭 조건: 거리 ≤50km, 시간 ±3일, 동일 도시 필수
  매칭률: 16~20% (8,000~10,000 / 50,000)
  라벨 정확도: 85~90%

  예시: 사고와 50km 이내, 3일 이내만 매칭
  → 높은 확신도의 라벨
```

---

### 2. 모델 성능 향상

**고품질 데이터의 효과:**

```
Phase 4-D (낮은 품질):
  F1 Score: 0.546
  Recall:   0.606

  문제: 노이즈 많은 라벨 → 모델 혼란

Phase 4-E (높은 품질):
  F1 Score: 0.60+ (예상)
  Recall:   0.70+ (예상)

  개선: 깨끗한 라벨 → 명확한 학습
```

---

### 3. Ensemble 효과

**단일 모델 vs Ensemble:**

```
단일 모델 (LR):
  F1 Score: 0.58
  문제: 선형 모델의 한계

Ensemble (LR + RF + GBM):
  F1 Score: 0.65 (예상)
  장점: 각 모델의 강점 결합
```

---

### 4. 주간/야간 Weight 차이

**예상 패턴:**

```
주간 (is_night=0):
  급가속 Weight: -0.02
  급정거 Weight:  0.05  ⭐ 중요
  급회전 Weight:  0.03
  과속 Weight:   -0.01

야간 (is_night=1):
  급가속 Weight: -0.03
  급정거 Weight:  0.08  ⭐⭐ 매우 중요
  급회전 Weight:  0.06  ⭐ 중요
  과속 Weight:    0.02

해석:
  야간에 급정거와 급회전이 더 위험
  → 감점 시스템에서 야간 가중치 증가
```

---

## 💡 감점 시스템 적용

### Phase 4-E 결과 기반 감점

**주간 감점:**
```python
day_penalty = {
    'rapid_accel': 1,   # 기본
    'sudden_stop': 3,   # 높음 (Weight 기반)
    'sharp_turn': 2,    # 중간
    'over_speed': 1     # 기본
}
```

**야간 감점 (1.5~2배):**
```python
night_penalty = {
    'rapid_accel': 2,   # 1.5배
    'sudden_stop': 5,   # 1.7배
    'sharp_turn': 4,    # 2배
    'over_speed': 2     # 2배
}
```

**최종 점수 계산:**
```python
def calculate_driving_score(events, is_night):
    penalty = night_penalty if is_night else day_penalty

    total_penalty = (
        events['rapid_accel'] * penalty['rapid_accel'] +
        events['sudden_stop'] * penalty['sudden_stop'] +
        events['sharp_turn'] * penalty['sharp_turn'] +
        events['over_speed'] * penalty['over_speed']
    )

    base_score = 100
    final_score = max(0, base_score - total_penalty)

    return final_score
```

---

## 🎯 성공 기준

### Phase 4-E 성공 지표

✅ **데이터 품질:**
- 매칭 라벨 정확도 ≥85%
- Combined 데이터 20,000개 생성

✅ **모델 성능:**
- F1 Score ≥0.60 (Phase 4-D: 0.546)
- Recall ≥0.70 (Phase 4-D: 0.606)
- Precision ≥0.65 (Phase 4-D: 0.497)

✅ **Ensemble 효과:**
- Ensemble이 단일 모델보다 우수
- F1 Score +5% 이상 개선

✅ **실용성:**
- Scenario B가 Scenario A보다 우수
- 주간/야간 Weight 차이 명확
- 감점 시스템에 적용 가능

---

## 📅 실행 계획

### Week 1: 데이터 준비 (Day 1-2)

- [x] Phase 4-E 계획 문서 작성
- [ ] Kaggle Real Sample 기반 데이터 생성 (500K)
- [ ] 고품질 매칭 수행 (50km, ±3일)
- [ ] Combined 데이터셋 20K 생성

### Week 2: 모델 학습 (Day 3-4)

- [ ] LR + Class Weight 학습
- [ ] Random Forest 구현 및 학습
- [ ] GBM 구현 및 학습
- [ ] Voting Ensemble 구현

### Week 3: 분석 및 최적화 (Day 5-6)

- [ ] Scenario A vs B 비교
- [ ] 주간/야간 Weight 분석
- [ ] Threshold 최적화
- [ ] Data Sample Report 작성

### Week 4: 문서화 (Day 7)

- [ ] 모델 결과 분석 문서
- [ ] 최종 리포트 작성
- [ ] 감점 시스템 업데이트

---

## 🚀 다음 단계

### 즉시 실행

1. **데이터 생성 스크립트 작성**
   ```bash
   python research/phase4e_high_quality_matching.py
   ```

2. **샘플 확인**
   ```bash
   python research/phase4e_data_samples.py
   ```

3. **모델 학습**
   ```bash
   python research/phase4e_model_training.py
   ```

### 향후 계획

- Phase 4-E 결과 기반 감점 시스템 업데이트
- 실제 서비스 적용 준비
- A/B 테스트 설계

---

## 📊 예상 타임라인

| 일정 | 작업 | 예상 소요 시간 |
|------|------|---------------|
| Day 1 | 계획 수립 + 데이터 생성 | 2-3시간 |
| Day 2 | 샘플 분석 + 리포트 작성 | 2-3시간 |
| Day 3 | LR, RF, GBM 모델 구현 | 3-4시간 |
| Day 4 | Ensemble + 평가 | 2-3시간 |
| Day 5 | Scenario 비교 + Weight 분석 | 2-3시간 |
| Day 6 | 최적화 + 문서화 | 2-3시간 |
| Day 7 | 최종 리포트 + 검토 | 2-3시간 |
| **총계** | | **15-22시간** |

---

## ✅ 체크리스트

### Phase 4-E 완료 조건

- [ ] 고품질 매칭 데이터 20K 생성 (라벨 정확도 ≥85%)
- [ ] LR + Class Weight 모델 학습 완료
- [ ] RF, GBM 모델 구현 및 학습 완료
- [ ] Voting Ensemble 구현 및 평가 완료
- [ ] Scenario A vs B 비교 분석 완료
- [ ] 주간/야간 Weight 분석 완료
- [ ] Data Sample Report 작성 완료
- [ ] Model Results 문서 작성 완료
- [ ] Final Report 작성 완료
- [ ] F1 Score ≥0.60 달성
- [ ] Recall ≥0.70 달성
- [ ] Precision ≥0.65 달성

---

**작성일:** 2025-10-15
**상태:** ✅ 계획 수립 완료
**다음 단계:** 고품질 매칭 데이터 생성
