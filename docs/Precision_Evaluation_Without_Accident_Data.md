# 사고 데이터 없이 Precision 평가하는 방법

**작성일:** 2025-10-15
**상황:** 실제 사고 데이터를 확보할 수 없는 상황에서 모델 성능 평가 방법

---

## 🎯 문제 정의

### 현재 상황
```
보유 데이터:
  ✅ 사용자의 운전 이벤트 (급가속, 급정거, 급회전)
  ✅ GPS, 시간, 속도 등 센서 데이터
  ❌ 실제 사고 발생 여부 (알 수 없음)

문제:
  - Precision = TP / (TP + FP)를 계산할 수 없음
  - Ground Truth (정답 라벨)이 없음
  - 모델이 정확한지 검증 불가능
```

### 왜 사고 데이터를 확보할 수 없나?

1. **개인정보 보호법**
   - 사고 이력은 민감 정보
   - 보험사 데이터 접근 제한

2. **데이터 수집 기간**
   - 사고는 드문 이벤트 (연간 2-3%)
   - 10,000명 × 1년 = 200-300건 (부족)

3. **신뢰성 문제**
   - 사용자 자체 보고 → 거짓말 가능
   - 경미한 접촉 사고 누락
   - 기억 오류

---

## 💡 해결 방안: 대체 평가 지표

### 방법 1: **Proxy Label (대리 라벨) 사용** ⭐ 추천

실제 사고 대신 **사고와 상관관계가 높은 대리 지표**를 사용

#### 1-1. 급정거 후 충격 감지

```python
# 스마트폰 가속도계로 충격 감지 가능
def detect_impact(accelerometer_data):
    """
    급정거 후 1초 내 강한 충격 = 사고 가능성
    """
    threshold_g = 3.0  # 3G 이상 충격

    for event in events:
        if event['type'] == 'sudden_stop':
            # 급정거 후 1초 내 데이터 확인
            next_second = accelerometer_data[event['time']:event['time']+1]

            max_impact = max(abs(next_second))
            if max_impact > threshold_g:
                return True  # 충돌 가능성

    return False

# Proxy Label
proxy_accident = detect_impact(data)
```

**장점:**
- 실시간 감지 가능
- 개인정보 문제 없음
- 데이터 즉시 수집 가능

**단점:**
- 과속방지턱도 감지될 수 있음 (False Positive)
- 경미한 접촉 사고는 놓칠 수 있음 (False Negative)
- Proxy Label 자체의 정확도 문제 (70-80%)

#### 1-2. 급제동 + 비상등 조합

```python
def detect_emergency_stop(events, hazard_light):
    """
    급제동 + 비상등 켜짐 = 위험 상황
    """
    for event in events:
        if (event['type'] == 'sudden_stop' and
            event['deceleration'] > 0.6 and  # 0.6G 이상
            hazard_light['on_time'] - event['time'] < 2):  # 2초 내 비상등
            return True

    return False

# Proxy Label
proxy_accident = detect_emergency_stop(events, hazard_light)
```

**장점:**
- 더 정확한 위험 상황 감지
- False Positive 감소

**단점:**
- 비상등 데이터 수집 필요 (OBD-II 동글 필요)
- 복잡도 증가

#### 1-3. ABS 작동 감지 (고급)

```python
def detect_abs_activation(obd_data):
    """
    ABS 작동 = 긴급 제동 상황
    차량 CAN 버스에서 직접 읽음
    """
    if obd_data['abs_active'] == True:
        return True

    return False
```

**장점:**
- 매우 정확한 긴급 상황 감지
- False Positive 최소화

**단점:**
- OBD-II 동글 필수
- 차량 호환성 문제
- 비용 증가

---

### 방법 2: **전문가 라벨링 (Expert Labeling)** ⭐⭐ 추천

운전 전문가가 주행 영상을 보고 위험도 평가

#### 2-1. 블랙박스 영상 라벨링

```python
# 프로세스
1. 사용자 동의 하에 블랙박스 영상 수집 (1분 클립)
2. 운전 강사, 보험 조사관 등 전문가 3명에게 라벨링 의뢰
3. 5점 척도 평가: 1(매우 안전) ~ 5(매우 위험)
4. 3명의 평균 점수 사용

# 라벨링 기준
labeling_criteria = {
    1: "매우 안전 - 방어 운전, 여유 있는 거리 유지",
    2: "안전 - 일반적인 운전, 특별한 문제 없음",
    3: "보통 - 약간 공격적이지만 사고 위험 낮음",
    4: "위험 - 급가속/급정거 많음, 거리 부족",
    5: "매우 위험 - 사고 직전 상황, 명백한 위험 운전"
}

# Ground Truth
ground_truth = expert_average_score >= 4  # 4점 이상 = 위험
```

**장점:**
- 높은 정확도 (전문가 판단)
- 맥락 이해 (날씨, 도로, 교통 상황)
- 설명 가능성 (왜 위험한지 기록)

**단점:**
- 비용 높음 (시간당 $50-100)
- 시간 소요 (1분 영상 = 5분 라벨링)
- 확장성 낮음 (대량 처리 어려움)

**비용 예측:**
```
1,000개 샘플 라벨링:
  - 1분 영상 × 1,000개 = 1,000분
  - 라벨링 시간 = 5,000분 (83시간)
  - 전문가 3명 × $50/시간 × 83시간 = $12,450
  - 전문가 1명으로 단축 시 = $4,150
```

#### 2-2. 크라우드소싱 (저비용 대안)

```python
# Amazon Mechanical Turk, 크라우드웍스 등 활용

labeling_task = {
    'platform': 'Amazon MTurk',
    'task': '운전 영상 보고 위험도 평가 (1-5점)',
    'workers_per_video': 5,  # 신뢰도 확보
    'payment': '$0.50 per video',
    'quality_control': 'Gold standard questions'
}

# 비용
cost = 1000 videos × $0.50 × 5 workers = $2,500
```

**장점:**
- 비용 절감 (전문가 대비 1/5)
- 빠른 처리 (병렬 작업)
- 대량 처리 가능

**단점:**
- 정확도 낮음 (비전문가)
- 품질 관리 필요
- 다수결로 보완 필요 (5명 이상)

---

### 방법 3: **자체 보고 + 검증 (Self-Report with Validation)**

사용자가 직접 보고하되, 여러 방법으로 검증

#### 3-1. 인센티브 기반 자체 보고

```python
# 정직한 보고를 유도하는 인센티브 설계

def incentive_based_reporting():
    """
    사용자가 사고/위험 상황을 정직하게 보고하도록 유도
    """

    # 1. 보고 시 리워드 (처벌 X, 보상 O)
    if user.report_near_miss():
        user.add_points(100)  # 포인트 적립
        user.add_badge("정직한 운전자")

    # 2. 익명성 보장
    report = {
        'timestamp': '2025-01-15 10:30',
        'type': 'near_miss',
        'description': '앞차 급정거로 인한 급제동',
        'user_id': 'anonymized',  # 익명 처리
        'used_for': 'research_only'  # 평가에 사용 안 함
    }

    # 3. 학습 기회 제공
    if report['type'] == 'near_miss':
        show_educational_content("급정거 대응법")
        offer_discount_coupon("방어 운전 교육 50% 할인")

    return report
```

**장점:**
- 비용 무료
- 대량 수집 가능
- 사용자 참여 유도

**단점:**
- 신뢰도 문제 (거짓말 가능)
- 선택 편향 (심각한 사고만 보고)
- 검증 필요

#### 3-2. 교차 검증 (Cross-Validation)

```python
def cross_validate_self_report(report, sensor_data, nearby_users):
    """
    자체 보고를 센서 데이터 및 주변 사용자 데이터로 검증
    """

    credibility_score = 0

    # 1. 센서 데이터와 일치 확인
    if report['type'] == 'sudden_stop':
        actual_deceleration = sensor_data['deceleration']
        if actual_deceleration > 0.5:  # 0.5G 이상
            credibility_score += 30

    # 2. GPS 위치 확인 (사고 다발 지역?)
    accident_prone_area = check_accident_database(report['location'])
    if accident_prone_area:
        credibility_score += 20

    # 3. 주변 사용자 데이터 확인
    nearby_reports = nearby_users.get_reports(
        location=report['location'],
        time_window=300  # ±5분
    )
    if len(nearby_reports) > 0:
        credibility_score += 30  # 다른 사용자도 보고

    # 4. 사용자 신뢰도 점수
    user_trust_score = user.get_trust_score()  # 과거 보고 정확도
    credibility_score += user_trust_score * 0.2

    # 최종 판정
    if credibility_score > 60:
        return True  # 신뢰 가능
    else:
        return False  # 의심스러움
```

**장점:**
- 거짓 보고 필터링
- 신뢰도 높은 데이터 확보
- 자동화 가능

**단점:**
- 복잡한 검증 로직 필요
- 여전히 100% 정확도는 불가능

---

### 방법 4: **상대적 비교 + A/B 테스트** ⭐⭐⭐ 최고 추천

실제 사고 데이터 없이도 **행동 변화**를 측정

#### 4-1. A/B 테스트 설계

```python
# 실제 Precision 대신 행동 변화를 측정

ab_test_design = {
    'Group A (Control)': {
        'users': 5000,
        'feedback': '피드백 없음',
        'measurement': '운전 이벤트 카운트'
    },
    'Group B (Treatment)': {
        'users': 5000,
        'feedback': 'Linear Model 점수 + 피드백',
        'measurement': '운전 이벤트 카운트'
    },
    'duration': '3개월',
    'metrics': [
        'rapid_accel_count',
        'sudden_stop_count',
        'sharp_turn_count',
        'user_engagement',
        'app_retention'
    ]
}

# 분석
def analyze_ab_test(group_a, group_b):
    """
    그룹 B가 그룹 A보다 이벤트가 줄었는가?
    """

    results = {}

    # 1. 급가속 감소율
    results['rapid_accel_reduction'] = (
        (group_a['rapid_accel'] - group_b['rapid_accel'])
        / group_a['rapid_accel'] * 100
    )

    # 2. 급정거 감소율
    results['sudden_stop_reduction'] = (
        (group_a['sudden_stop'] - group_b['sudden_stop'])
        / group_a['sudden_stop'] * 100
    )

    # 3. 통계적 유의성 (t-test)
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(group_a['events'], group_b['events'])
    results['p_value'] = p_value
    results['significant'] = p_value < 0.05

    # 4. 효과 크기 (Cohen's d)
    mean_diff = group_a['events'].mean() - group_b['events'].mean()
    pooled_std = np.sqrt(
        (group_a['events'].std()**2 + group_b['events'].std()**2) / 2
    )
    results['effect_size'] = mean_diff / pooled_std

    return results

# 결과 해석
"""
Group B가 Group A보다 이벤트가 20% 감소:
  → Linear Model 피드백이 효과가 있다는 증거
  → Precision이 낮아도 행동 변화는 유도 가능
  → 실용적 가치 입증

Group B와 Group A가 차이 없음:
  → Linear Model 피드백 효과 없음
  → 시스템 개선 필요
"""
```

**장점:**
- **Ground Truth 불필요** ⭐⭐⭐
- 실제 효과 측정 (행동 변화)
- 통계적 검증 가능
- 확장 가능

**단점:**
- 3개월 이상 소요
- 대규모 사용자 필요 (최소 1,000명)
- 외부 변수 통제 필요 (계절, 캠페인 등)

#### 4-2. 시계열 분석 (Before/After)

```python
def before_after_analysis(user_data):
    """
    피드백 시스템 도입 전/후 비교
    """

    # 피드백 도입 전 3개월
    before = user_data['2024-10-01':'2024-12-31']

    # 피드백 도입 후 3개월
    after = user_data['2025-01-01':'2025-03-31']

    # 비교
    comparison = {
        'rapid_accel': {
            'before': before['rapid_accel'].mean(),
            'after': after['rapid_accel'].mean(),
            'change': (after['rapid_accel'].mean() - before['rapid_accel'].mean())
                     / before['rapid_accel'].mean() * 100
        },
        'sudden_stop': {
            'before': before['sudden_stop'].mean(),
            'after': after['sudden_stop'].mean(),
            'change': (after['sudden_stop'].mean() - before['sudden_stop'].mean())
                     / before['sudden_stop'].mean() * 100
        }
    }

    # 트렌드 분석
    import statsmodels.api as sm

    # 시계열 분해
    decomposition = sm.tsa.seasonal_decompose(
        user_data['rapid_accel'],
        model='additive',
        period=30  # 월별
    )

    # 트렌드 추출
    trend = decomposition.trend

    # 피드백 도입 후 트렌드 감소?
    if trend[-1] < trend[len(trend)//2]:
        print("✅ 트렌드 감소 확인: 피드백 효과 있음")
    else:
        print("❌ 트렌드 변화 없음: 피드백 효과 없음")

    return comparison
```

**장점:**
- 단일 그룹으로 가능 (A/B 불필요)
- 장기 트렌드 파악
- 계절성 제거 가능

**단점:**
- 인과관계 불명확 (다른 요인일 수도)
- 외부 이벤트 영향 (캠페인, 뉴스 등)

---

### 방법 5: **보험사 협력 (장기 전략)**

보험사와 파트너십을 통해 실제 사고 데이터 확보

#### 5-1. 데이터 교환 계약

```python
partnership_model = {
    'partner': '보험사',
    'data_exchange': {
        'we_provide': [
            '익명화된 운전 점수',
            '위험 운전자 그룹 분류',
            '사고 예측 모델 결과'
        ],
        'they_provide': [
            '실제 사고 발생 여부 (익명)',
            '사고 심각도 (경미/중대)',
            '보험 청구 금액'
        ]
    },
    'privacy': {
        'anonymization': 'k-anonymity (k≥5)',
        'aggregation': '최소 100명 단위',
        'no_pii': '개인 식별 정보 제거'
    },
    'benefit': {
        'for_us': 'Ground Truth 확보',
        'for_them': '사고 예측 모델로 보험료 최적화'
    }
}

# 검증 프로세스
def validate_with_insurance_data(predictions, actual_accidents):
    """
    보험사 데이터로 모델 검증
    """

    # 매칭
    matched = 0
    for user_id in predictions:
        pred_risk = predictions[user_id]['risk_score']
        actual_accident = actual_accidents.get(user_id, False)

        if pred_risk > 0.7 and actual_accident:
            matched += 1  # True Positive

    # Precision 계산
    high_risk_count = sum(1 for p in predictions.values() if p['risk_score'] > 0.7)
    precision = matched / high_risk_count

    print(f"Precision: {precision:.2%}")
    print(f"검증 완료: {len(predictions)}명 사용자")

    return precision
```

**장점:**
- **실제 Ground Truth** ⭐⭐⭐
- 대규모 데이터 (수만~수십만)
- 높은 정확도

**단점:**
- 계약 협상 시간 (6개월~1년)
- 법적 검토 필요
- 개인정보 보호 이슈
- 초기 단계에는 불가능

#### 5-2. 텔레매틱스 보험 제품화

```python
# 우리 시스템을 보험 상품으로 만들기

telematics_insurance = {
    'product_name': 'Safe Driving Discount',
    'model': {
        'base_premium': '$1,000/year',
        'max_discount': '30%',
        'evaluation_period': '3 months',
        'driving_score': 'Our Linear Model'
    },
    'validation': {
        'after_1_year': '실제 사고율 vs 예측 비교',
        'precision_calculation': 'Actual claims / Predicted high-risk drivers'
    }
}

# 1년 후 검증
"""
예측:
  - 고위험 운전자 1,000명 (상위 10%)
  - 저위험 운전자 9,000명 (하위 90%)

실제 (1년 후):
  - 고위험 그룹 사고율: 5%
  - 저위험 그룹 사고율: 1%

분석:
  - 고위험 그룹이 5배 더 위험 → 모델 효과 있음
  - Precision 계산 가능
  - 비즈니스 모델 검증
"""
```

**장점:**
- 수익 모델 (비용 X, 수익 O)
- 장기 데이터 자연스럽게 수집
- 실제 사고율로 검증

**단점:**
- 1년 이상 소요
- 보험 라이센스 필요 또는 파트너십 필수
- 초기 불확실성 (모델 성능 검증 전 출시)

---

## 🎯 단계별 추천 전략

### Phase 1: 즉시 시작 (0-3개월)

```python
immediate_actions = {
    'Method 1': 'Proxy Label - 충격 감지',
    'cost': '무료 (기존 센서 활용)',
    'accuracy': '70-80%',
    'data_volume': '수천 건/월',
    'effort': '낮음'
}

immediate_actions['Method 4'] = 'A/B 테스트 설계 및 시작',
immediate_actions['cost_4'] = '무료',
immediate_actions['value_4'] = '실제 효과 측정 (Ground Truth 불필요)'
```

**실행 계획:**
1. 앱에 충격 감지 기능 추가 (1주)
2. 사용자 1,000명에게 배포 (2주)
3. A/B 테스트 시작 (3개월 진행)

### Phase 2: 중기 실행 (3-6개월)

```python
mid_term_actions = {
    'Method 2': '크라우드소싱 라벨링',
    'cost': '$2,500 (1,000 샘플)',
    'accuracy': '75-85%',
    'data_volume': '1,000 샘플',
    'effort': '중간'
}

mid_term_actions['Method 3'] = '자체 보고 + 검증 시스템 구축',
mid_term_actions['cost_3'] = '무료',
mid_term_actions['volume_3'] = '수백 건/월'
```

**실행 계획:**
1. 100개 샘플 크라우드소싱 라벨링 테스트 (1개월)
2. 품질 검증 후 1,000개로 확대 (2개월)
3. 자체 보고 시스템 개발 및 배포 (3개월)
4. A/B 테스트 결과 분석 (3개월 완료 시점)

### Phase 3: 장기 전략 (6-12개월)

```python
long_term_actions = {
    'Method 5': '보험사 협력 협상',
    'timeline': '6-12개월',
    'cost': '무료 또는 수익 공유',
    'accuracy': '95%+ (실제 사고 데이터)',
    'data_volume': '수만 건/년',
    'effort': '높음 (법률, 계약)'
}

long_term_actions['Telematics Product'] = '텔레매틱스 보험 제품화',
long_term_actions['timeline_2'] = '12개월+',
long_term_actions['value_2'] = '수익 모델 + Ground Truth 확보'
```

**실행 계획:**
1. 보험사 파트너 발굴 (3개월)
2. 계약 협상 및 법률 검토 (3개월)
3. 파일럿 프로그램 (6개월)
4. 전면 확대 (12개월+)

---

## 📊 방법 비교표

| 방법 | 비용 | 정확도 | 시간 | 확장성 | 추천도 |
|------|------|--------|------|--------|--------|
| **Proxy Label (충격)** | 무료 | 70-80% | 즉시 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **전문가 라벨링** | $12K | 90-95% | 3개월 | ⭐⭐ | ⭐⭐⭐ |
| **크라우드소싱** | $2.5K | 75-85% | 1개월 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **자체 보고 + 검증** | 무료 | 60-70% | 1개월 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **A/B 테스트** | 무료 | N/A* | 3개월 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **보험사 협력** | 무료** | 95%+ | 12개월+ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

*N/A: Precision 측정 안 함, 대신 실제 효과(행동 변화) 측정
**무료 또는 수익 공유 모델

---

## 🎯 최종 권장안

### ✅ **즉시 실행 (오늘부터):**

1. **A/B 테스트 시작** ⭐⭐⭐⭐⭐
   ```
   - Ground Truth 불필요
   - 실제 효과 측정 (행동 변화)
   - 비용: 무료
   - 기간: 3개월
   - 사용자: 최소 1,000명 (Group A: 500, Group B: 500)
   ```

2. **Proxy Label 구현** ⭐⭐⭐⭐
   ```
   - 충격 감지 (가속도계)
   - 급제동 + 비상등
   - 비용: 무료
   - 정확도: 70-80%
   - 즉시 데이터 수집 시작
   ```

### ✅ **3개월 후 실행:**

3. **크라우드소싱 라벨링** ⭐⭐⭐⭐
   ```
   - 1,000개 샘플 라벨링
   - 비용: $2,500
   - Precision 계산 가능
   - 모델 검증
   ```

4. **자체 보고 시스템** ⭐⭐⭐
   ```
   - 인센티브 기반
   - 교차 검증
   - 지속적 데이터 수집
   ```

### ✅ **6-12개월 후 실행:**

5. **보험사 협력** ⭐⭐⭐⭐⭐
   ```
   - 실제 사고 데이터 확보
   - 장기 전략
   - 수익 모델 연계
   ```

---

## 💡 핵심 통찰

### **Precision 평가는 필수가 아니다**

```
전통적 사고:
  Precision을 반드시 계산해야 한다 ❌

새로운 관점:
  실제 효과(행동 변화)를 측정하면 된다 ✅

이유:
  - 사용자는 Precision을 모름
  - 중요한 것은 "운전 습관이 개선되는가?"
  - A/B 테스트로 충분히 검증 가능
```

### **단계적 접근의 중요성**

```
Phase 1 (0-3개월):
  → A/B 테스트 + Proxy Label
  → 효과 있으면 계속, 없으면 중단

Phase 2 (3-6개월):
  → 크라우드소싱으로 Precision 추정
  → 모델 개선

Phase 3 (6-12개월):
  → 보험사 협력으로 Ground Truth 확보
  → 고정밀 시스템 구축
```

### **비즈니스 가치 중심**

```
질문: Precision이 50%인데 서비스할 수 있나?

답변:
  - A/B 테스트에서 Group B가 이벤트 20% 감소
  → 효과가 있다는 증거
  → Precision 50%여도 가치 있음

  - A/B 테스트에서 차이 없음
  → 효과가 없다는 증거
  → Precision 90%여도 가치 없음
```

---

**결론: 사고 데이터 없이도 A/B 테스트와 Proxy Label로 충분히 시작 가능. 단계적으로 정확도를 높여가면 됨.**

---

**작성일:** 2025-10-15
**다음 단계:** A/B 테스트 설계 문서 작성 + Proxy Label 구현 계획
