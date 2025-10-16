# Phase 4F: Behavior-based Approach 개선 계획

**작성일**: 2025년 10월 16일
**기반**: Phase 4F 실제 데이터 분석 및 Cross-Validation 결과

---

## 요약

Phase 4F와 Phase 4D 교차 검증 결과, **단순 IMU 센서 기반 점수 시스템의 심각한 한계**가 드러났습니다. README.md의 **Behavior-based approach** 원칙에 맞춰, 실제 데이터를 기반으로 실행 가능한 개선 방향을 제시합니다.

### 핵심 문제점

| 문제 | 현상 | 근본 원인 |
|------|------|-----------|
| **🔴 모델 성능 붕괴** | Precision 94% → 13.7%, Recall 90% → 0.6% | 합성 데이터와 실제 데이터의 괴리 |
| **🔴 변별력 부족** | AGGRESSIVE vs SAFE 사고율 1.2배 (목표: 4배) | 단순 이벤트 카운트의 한계 |
| **🔴 과도한 SAFE 분류** | 86.1%가 SAFE (Risk Group의 72.8% 포함) | 임계값 및 가중치 불일치 |
| **🔴 낮은 Recall** | 실제 위험 운전자의 98.8% 미탐지 | 특징 공간 부족 |

---

## 1. 실제 데이터 분석 (Phase 4F)

### 1.1 데이터 품질 검증 ✅

**긍정적인 결과:**

```
총 샘플: 20,000개
Risk Group: 10,000 (50.0%)
Safe Group: 10,000 (50.0%)

사고율:
  Risk: 2,000/10,000 = 20.0%
  Safe: 500/10,000 = 5.0%
  비율: 4.00:1 (목표 달성!)
```

**핵심 인사이트:**
- ✅ 4:1 사고율 비율 정확히 달성 (현실적)
- ✅ 라벨 정확도 85-90% 추정
- ✅ 오버샘플링 없음 (20,000 unique IDs)

### 1.2 이벤트 패턴 분석

**Risk Group vs Safe Group 평균 이벤트 수:**

| 이벤트 | Risk Group | Safe Group | **비율** |
|--------|------------|------------|----------|
| 급가속 | 2.79 | 0.87 | **3.21x** |
| 급정거 | 2.26 | 0.83 | **2.72x** |
| 급회전 | 1.90 | 0.74 | **2.55x** |
| 과속 | 1.38 | 0.51 | **2.69x** |

**핵심 발견:**
- ✅ Risk Group이 모든 이벤트에서 **2.5~3.2배 높음**
- ✅ 급가속이 가장 높은 차별력 (3.21x)
- ⚠️ **하지만 현재 모델은 이를 제대로 활용하지 못함**

### 1.3 현재 모델 성능 (Phase 4F LR 모델)

```
Scenario A & B (둘 다 Precision-focused):
  Threshold: 0.76
  Precision: 50.0%
  Recall: 0.5%  ← 거의 아무것도 못 찾음!
  F1: 1.0%

혼동 행렬:
  True Positive: 4
  False Positive: 4
  False Negative: 751  ← 실제 위험자의 99.5%를 놓침!
  True Negative: 5,241
```

**문제점:**
- 6,000명 테스트 중 **단 8명만 Risk로 예측**
- 실제 위험 운전자 755명 중 **4명만 감지** (0.5%)
- **사실상 모든 사람을 Safe로 분류**

---

## 2. README.md의 Behavior-based Approach 원칙

### 원문 인용:

> "단순 결과(Outcome)에 기반한 점수만으로는 보험료 조정 외 실질적인 개선이 어렵습니다. DrivingScore는 급가속·급제동·야간 주행 등 **행동 데이터를 직접 계량화**해 **즉각적인 피드백**을 제공하는 Behavior-based 접근을 채택합니다."

> "**Calibrate by Truth, Feedback by Behavior** 원칙을 운영 전략으로 삼습니다."

### 현재 상황 평가:

| 원칙 | 현재 상태 | 문제점 |
|------|-----------|--------|
| **행동 데이터 계량화** | ⚠️ 부분적 | IMU만 사용, 맥락 정보 부족 |
| **즉각적 피드백** | ❌ 불가능 | 모델이 작동하지 않음 (Recall 0.5%) |
| **Calibrate by Truth** | ⚠️ 부분적 | 4:1 비율은 맞지만 모델이 반영 못 함 |
| **Feedback by Behavior** | ❌ 실패 | 86%가 SAFE → 차별화 없음 |

---

## 3. 개선 방향: 3단계 접근법

### Phase A: 즉시 실행 가능 (1-2주)
### Phase B: 중기 개선 (1-3개월)
### Phase C: 장기 전략 (3-6개월)

---

## Phase A: 즉시 실행 가능 개선 (1-2주)

### A1. 클래스 불균형 처리 강화

**문제**: 현재 Class Weight (Positive: 4.01, Negative: 0.57)로도 부족

**개선안**:
```python
# 현재
class_weight = 'balanced'  # 자동 계산

# 제안 1: Manual Override
class_weight = {0: 1, 1: 10}  # 양성 클래스 가중치 대폭 증가

# 제안 2: Custom Balanced
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train) * 2
```

**예상 효과**: Recall 0.5% → 30%+ 향상

### A2. 임계값 대폭 하향 조정

**문제**: Threshold 0.76은 너무 높음 (Recall 0.5%)

**제안**:
```python
# 현재 최적화
scenario_weights = (0.7, 0.2, 0.1)  # Precision 중심

# 제안: Balanced 접근
scenario_weights = (0.3, 0.5, 0.2)  # F1 중심

# 또는 Recall 우선
scenario_weights = (0.2, 0.6, 0.2)  # Recall 중심
```

**근거**:
- **행동 변화 유도**가 목표 → **위험자를 더 많이 찾아야** 함
- False Positive는 감수 가능 (피드백 제공 기회)
- False Negative는 치명적 (개선 기회 상실)

**예상 Threshold**: 0.76 → 0.3~0.4

### A3. 특징 엔지니어링 (기존 데이터 활용)

**현재 특징 (5개)**:
- rapid_accel, sudden_stop, sharp_turn, over_speed, is_night

**추가 특징 (데이터 가공)**:
```python
# 1. 이벤트 총합
'total_events': rapid_accel + sudden_stop + sharp_turn + over_speed

# 2. 위험 이벤트 비율
'risky_event_ratio': (급가속 + 급정거) / max(total_events, 1)

# 3. 야간 위험 이벤트
'night_risky_events': (급가속 + 급정거) * is_night * 1.5

# 4. 이벤트 조합 (급가속 + 급정거)
'emergency_maneuvers': min(급가속, 급정거)  # 급정거 직전 급가속

# 5. 과속 중 급회전
'overspeed_turn': over_speed * sharp_turn

# 6. 이벤트 표준편차 (변동성)
# 여러 trip이 있다면 표준편차 계산
```

**예상 효과**: AUC +0.05, F1 +10%p

### A4. SMOTE 대신 Under-sampling + Class Weight

**문제**: 현재 SMOTE 미사용 (오버샘플링 금지 원칙)

**제안**: Under-sampling 병행
```python
from imblearn.under_sampling import RandomUnderSampler

# Safe Group 일부만 샘플링
rus = RandomUnderSampler(sampling_strategy=0.5)  # Risk:Safe = 1:2
X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)

# Class Weight 함께 사용
model = LogisticRegression(class_weight={0: 1, 1: 5})
```

**장점**:
- 오버샘플링 금지 원칙 준수
- Class 불균형 완화
- 학습 속도 향상

---

## Phase B: 중기 개선 (1-3개월)

### B1. 앙상블 모델 고도화

**현재**: LR + 규칙 기반 (간소화)

**제안**: 진짜 앙상블
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

# 모델 구성
lr = LogisticRegression(class_weight={0:1, 1:10})
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced_subsample'
)
gbm = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)

# Soft Voting
ensemble = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('gbm', gbm)],
    voting='soft'
)
```

**예상 성능**:
- Phase 4D 수준 회복 (F1 50%+)
- RandomForest가 비선형 패턴 포착

### B2. XGBoost + 하이퍼파라미터 튜닝

**제안**:
```python
import xgboost as xgb

# XGBoost with class imbalance
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=10,  # Class imbalance 처리
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc'
)

# Grid Search
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'scale_pos_weight': [5, 10, 15]
}

grid_search = GridSearchCV(
    model, param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
```

**예상 효과**: F1 1% → 60%+

### B3. 맥락 정보 추가 (메타데이터 활용)

**현재 메타데이터**:
- sensor_id, city, trip_duration

**활용 방안**:
```python
# 1. 도시별 위험도 (통계 기반)
city_accident_rate = {
    'New York': 0.15,
    'Los Angeles': 0.12,
    # ... 실제 사고 데이터로 계산
}
features['city_risk'] = city_accident_rate.get(city, 0.1)

# 2. 주행 시간 구간
features['trip_length_category'] = (
    0 if trip_duration < 30 else
    1 if trip_duration < 60 else
    2  # 장거리
)

# 3. 이벤트 밀도
features['event_density'] = total_events / max(trip_duration, 1)

# 4. 도시 규모
big_cities = ['New York', 'Los Angeles', 'Chicago']
features['is_big_city'] = 1 if city in big_cities else 0
```

### B4. 상대 평가 시스템 (Percentile-based)

**문제**: 절대 점수로는 변별력 부족

**제안**:
```python
# 1. 점수 계산
scores = model.predict_proba(X_test)[:, 1]

# 2. 백분위수 변환
from scipy.stats import percentileofscore

percentiles = [percentileofscore(scores, s) for s in scores]

# 3. 등급 부여
def assign_grade(percentile):
    if percentile >= 90:
        return 'SAFE'
    elif percentile >= 75:
        return 'MODERATE'
    else:
        return 'AGGRESSIVE'

# 4. 자동으로 분포 조정
# SAFE 65%, MODERATE 25%, AGGRESSIVE 10% 달성
```

**장점**:
- 분포 자동 조정
- 시간 경과에 따른 기준 변화 대응
- 사용자 간 상대 비교 가능

---

## Phase C: 장기 전략 (3-6개월)

### C1. 외부 데이터 통합

**1) 날씨 데이터 (API)**
```python
# OpenWeatherMap API
weather_features = {
    'rain': 1 if 'rain' in weather else 0,
    'snow': 1 if 'snow' in weather else 0,
    'temperature': temp,
    'visibility': visibility_km
}
```

**2) 교통 정보**
```python
# Google Maps Traffic API
traffic_features = {
    'traffic_level': 0-3,  # 0=원활, 3=정체
    'rush_hour': 1 if 7-9시 or 17-19시 else 0
}
```

**3) 도로 유형 (OSM)**
```python
# OpenStreetMap
road_features = {
    'highway': 1,
    'urban': 1,
    'residential': 0
}
```

### C2. 시계열 모델 (LSTM/Transformer)

**현재**: 각 trip 독립적 평가

**제안**: 시계열 패턴 학습
```python
import torch
import torch.nn as nn

class DrivingScoreLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x: (batch, sequence_length, features)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# 여러 trip을 sequence로 학습
# 운전 패턴의 시간적 변화 포착
```

**활용**:
- 최근 N개 trip의 패턴 분석
- 개선/악화 추세 감지
- 장기 행동 패턴 학습

### C3. 설명 가능한 AI (Explainable AI)

**목적**: 사용자에게 "왜 이 점수인가?" 설명

**방법 1: SHAP**
```python
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# 사용자별 설명
for i, sample in enumerate(X_test[:5]):
    print(f"\n사용자 {i}의 점수 요인:")
    for feature, shap_val in zip(feature_names, shap_values[i]):
        print(f"  {feature}: {shap_val:+.2f}")
```

**방법 2: Feature Importance 시각화**
```python
# XGBoost Feature Importance
importance = xgb_model.feature_importances_

# 사용자 피드백
feedback = {
    'score': 45,
    'grade': 'AGGRESSIVE',
    'top_issues': [
        '급가속 횟수가 평균보다 3배 높습니다',
        '야간 주행 중 급정거 빈번',
        '과속 구간에서 급회전 발생'
    ],
    'recommendations': [
        '부드러운 가속 연습',
        '야간 주행 시 속도 줄이기',
        '커브 전 감속'
    ]
}
```

### C4. 강화학습 기반 동적 가중치

**아이디어**: 사용자 행동 변화에 따라 가중치 조정

```python
# Q-Learning 기반 가중치 최적화
class AdaptiveWeightOptimizer:
    def __init__(self):
        self.weights = {
            'rapid_accel': 3.0,
            'sudden_stop': 3.0,
            'sharp_turn': 2.0,
            'over_speed': 2.0
        }

    def update(self, user_feedback, accident_occurred):
        # 사고 발생 시 해당 이벤트 가중치 증가
        if accident_occurred:
            for event, count in user_feedback.items():
                if count > threshold:
                    self.weights[event] *= 1.1

        # 개선 시 가중치 안정화
        else:
            for event in self.weights:
                self.weights[event] *= 0.99

# 개인화된 점수 시스템
```

---

## 4. 실행 우선순위 및 로드맵

### Week 1-2: Quick Wins (Phase A)

**목표**: Recall 0.5% → 30%+ 달성

| 번호 | 작업 | 예상 시간 | 예상 효과 |
|------|------|-----------|-----------|
| 1 | Class Weight 증가 (1→10) | 1시간 | Recall +20%p |
| 2 | Threshold 하향 (0.76→0.35) | 2시간 | Recall +10%p |
| 3 | 특징 엔지니어링 (6개 추가) | 1일 | F1 +10%p |
| 4 | Scenario B를 Recall-focused로 복원 | 2시간 | 선택지 제공 |

**검증**:
```bash
cd research
python phase4f_step3_model_training_improved.py
```

### Week 3-4: 앙상블 적용 (Phase B1-B2)

**목표**: F1 1% → 50%+ 달성

| 번호 | 작업 | 예상 시간 | 예상 효과 |
|------|------|-----------|-----------|
| 1 | RandomForest 추가 | 1일 | +15%p F1 |
| 2 | XGBoost 적용 | 2일 | +20%p F1 |
| 3 | Voting Ensemble | 1일 | +5%p F1 |
| 4 | 하이퍼파라미터 튜닝 | 2일 | +10%p F1 |

### Month 2: 맥락 정보 (Phase B3-B4)

**목표**: 실제 배포 준비

| 번호 | 작업 | 예상 시간 | 예상 효과 |
|------|------|-----------|-----------|
| 1 | 메타데이터 특징 추가 | 3일 | +5%p AUC |
| 2 | 상대 평가 시스템 | 5일 | 분포 개선 |
| 3 | A/B 테스트 설계 | 3일 | 검증 체계 |
| 4 | 최종 검증 | 4일 | - |

### Month 3-6: 장기 전략 (Phase C)

**선택적 진행**

---

## 5. 성공 지표 (Success Metrics)

### 5.1 모델 성능 지표

**최소 목표 (Phase A 완료 시)**:
- Precision: ≥ 40%
- Recall: ≥ 30%
- F1: ≥ 30%

**목표 (Phase B 완료 시)**:
- Precision: ≥ 70%
- Recall: ≥ 60%
- F1: ≥ 65%

**최종 목표 (Phase C)**:
- Precision: ≥ 80%
- Recall: ≥ 70%
- F1: ≥ 75%
- AUC: ≥ 0.85

### 5.2 비즈니스 지표

**변별력**:
- AGGRESSIVE / SAFE 사고율 비율: ≥ 3:1 (현재 1.2:1)

**분포**:
- SAFE: 60-70% (현재 86.1%)
- MODERATE: 20-30% (현재 13.4%)
- AGGRESSIVE: 5-15% (현재 0.6%)

**행동 변화**:
- 위험 운전자 탐지율: ≥ 70% (현재 1.2%)
- 안전 운전자 정확도: ≥ 90% (현재 99.4% - 유지)

---

## 6. 위험 요인 및 대응 방안

### 6.1 데이터 품질

**위험**: 라벨 노이즈 10-15%

**대응**:
- 앙상블 모델로 노이즈 robust
- Cross-validation으로 검증
- 이상치 탐지 및 제거

### 6.2 과적합

**위험**: 학습 데이터 20K로 제한적

**대응**:
- Regularization (L1/L2)
- Early stopping
- K-fold CV

### 6.3 특징 부족

**위험**: IMU 센서만으로 한계

**대응**:
- 특징 엔지니어링 우선
- 외부 데이터는 선택적
- 메타데이터 최대 활용

---

## 7. 핵심 권장사항

### 즉시 실행 (이번 주)

1. **Class Weight를 10으로 증가**
   ```python
   class_weight = {0: 1, 1: 10}
   ```

2. **Scenario B를 Recall-focused로 복원**
   ```python
   scenario_b_weights = (0.2, 0.7, 0.1)  # Precision, Recall, F1
   ```

3. **Threshold 범위 확장**
   ```python
   thresholds = [i / 100 for i in range(5, 91)]  # 0.05 ~ 0.90
   ```

4. **특징 엔지니어링 3개 추가**
   - total_events
   - risky_event_ratio
   - night_risky_events

### 다음 달

5. **XGBoost 적용**
   - scale_pos_weight=10
   - max_depth=6
   - GridSearchCV

6. **메타데이터 활용**
   - city_risk
   - trip_duration_category
   - event_density

7. **상대 평가 시스템**
   - Percentile 기반
   - 자동 분포 조정

---

## 8. 결론

### 현재 상황 요약

**✅ 잘 된 점**:
- 4:1 사고율 비율 정확히 달성
- 고품질 데이터 20K 확보
- Risk Group 이벤트 2.5~3.2배 차이 확인

**🔴 문제점**:
- 모델 성능 붕괴 (Recall 0.5%)
- 변별력 부족 (1.2배)
- 실용성 없음 (86%가 SAFE)

### Behavior-based Approach 실현을 위한 핵심

**원칙 재확인**:
> "급가속·급제동·야간 주행 등 **행동 데이터를 직접 계량화**해 **즉각적인 피드백**을 제공"

**실현 방안**:

1. **Recall 우선 최적화** → 위험 운전자를 먼저 찾아야 행동 변화 유도 가능
2. **특징 엔지니어링** → 행동의 맥락(context) 반영
3. **앙상블 모델** → 비선형 패턴 포착
4. **상대 평가** → 지속적인 개선 동기 부여
5. **설명 가능성** → "왜 이 점수?" 명확히 전달

### 다음 단계

**Week 1-2 (즉시)**:
```bash
cd research
# 1. 스크립트 수정
# - Class Weight: 10
# - Scenario B: Recall-focused
# - Feature Engineering: 3개 추가

python phase4f_step3_model_training_v2.py
python phase4f_step4_final_report.py
```

**Week 3-4 (앙상블)**:
```bash
python phase4f_step3_ensemble.py  # XGBoost + RF + LR
```

**Month 2 (검증)**:
```bash
python phase4f_ab_test.py  # 실제 사용자 테스트
```

---

## 부록

### A. 참고 문서

- README.md: Behavior-based approach 원칙
- Phase4F_Final_Report.md: 현재 모델 성능
- Phase4D_4F_Cross_Validation_Report.md: 성능 격차 분석
- Phase4D_Model_Improvement.md: Phase 4D 성공 사례

### B. 코드 템플릿

**즉시 적용 가능한 개선 코드**:

```python
# phase4f_step3_model_training_v2.py

# 1. Class Weight 증가
weight_positive = n_samples / (2 * n_positive) * 2.5  # 2.5배 증가

# 2. Scenario B를 Recall-focused로
scenario_b_weights = (0.2, 0.7, 0.1)

# 3. 특징 엔지니어링
def add_engineered_features(sample):
    f = sample['features']

    # Total events
    f['total_events'] = (f['rapid_accel'] + f['sudden_stop'] +
                        f['sharp_turn'] + f['over_speed'])

    # Risky event ratio
    f['risky_ratio'] = (f['rapid_accel'] + f['sudden_stop']) / max(f['total_events'], 1)

    # Night risky events
    f['night_risky'] = (f['rapid_accel'] + f['sudden_stop']) * f['is_night'] * 1.5

    return sample

# 4. Threshold 범위 확장
thresholds = [i / 100 for i in range(5, 91)]  # 0.05 ~ 0.90
```

---

**마지막 업데이트**: 2025년 10월 16일
**다음 리뷰**: Week 1-2 개선 후
