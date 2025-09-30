# Phase 5: 실사용자 데이터 기반 등급 조정 전략

## 🎯 **목표**

**사고 데이터 없이** 실제 사용자들의 운전 이벤트 분포를 활용하여 SAFE/MODERATE/AGGRESSIVE 등급 기준을 동적으로 조정하는 시스템 구축

---

## 📊 **핵심 개념: 상대 평가 + 절대 평가 하이브리드**

### 기본 아이디어
```
Phase 1-4: 절대 평가 (사고 데이터 기반 가중치)
  → 급정거 -3.5점, 급가속 -2.8점 등 고정

Phase 5: 하이브리드 평가
  → 절대 기준(Phase 4 가중치) 유지
  → 컷오프를 사용자 분포에 맞춰 동적 조정
```

---

## 🔧 **방법론 1: 백분위 기반 동적 컷오프 조정**

### 1.1 개념
```python
# Phase 4에서 도출된 초기 컷오프
INITIAL_CUTOFFS = {
    "aggressive": 77.0,  # ≤ 77점
    "safe": 88.0         # ≥ 88점
}

# 실사용자 데이터 수집 후
user_scores = [85.2, 92.1, 78.5, ...]  # 10,000명

# 목표 분포 (Phase 3 참고)
TARGET_DISTRIBUTION = {
    "SAFE": 0.65,        # 65%
    "MODERATE": 0.25,    # 25%
    "AGGRESSIVE": 0.10   # 10%
}

# 백분위 기반 조정
safe_cutoff = np.percentile(user_scores, 35)      # 하위 35% 제외
aggressive_cutoff = np.percentile(user_scores, 10) # 하위 10%
```

### 1.2 장점
- ✅ 사고 데이터 불필요
- ✅ 사용자 풀 변화에 자동 대응
- ✅ 통계적으로 안정적

### 1.3 단점
- ⚠️ 전체 사용자가 위험 운전하면 기준 하락
- ⚠️ 절대적 안전성 보장 어려움

### 1.4 구현 예시
```python
class DynamicCutoffAdjuster:
    def __init__(self, phase4_weights, initial_cutoffs, target_distribution):
        self.weights = phase4_weights
        self.initial_cutoffs = initial_cutoffs
        self.target_dist = target_distribution
        self.min_samples = 1000  # 최소 샘플 수
        
    def adjust_cutoffs(self, user_scores):
        """
        사용자 점수 분포에 맞춰 컷오프 조정
        """
        if len(user_scores) < self.min_samples:
            return self.initial_cutoffs  # 샘플 부족 시 초기값 사용
            
        # 백분위 기반 계산
        safe_percentile = (1 - self.target_dist["SAFE"]) * 100
        aggressive_percentile = self.target_dist["AGGRESSIVE"] * 100
        
        safe_cutoff = np.percentile(user_scores, safe_percentile)
        aggressive_cutoff = np.percentile(user_scores, aggressive_percentile)
        
        # 안전 장치: Phase 4 기준에서 크게 벗어나지 않도록
        safe_cutoff = np.clip(safe_cutoff, 
                             self.initial_cutoffs["safe"] - 5,
                             self.initial_cutoffs["safe"] + 5)
        aggressive_cutoff = np.clip(aggressive_cutoff,
                                   self.initial_cutoffs["aggressive"] - 5,
                                   self.initial_cutoffs["aggressive"] + 5)
        
        return {
            "safe": safe_cutoff,
            "aggressive": aggressive_cutoff
        }
```

---

## 🔧 **방법론 2: 이동 평균 기반 점진적 조정**

### 2.1 개념
```python
# 매주/매월 사용자 데이터로 점진적 업데이트
# 급격한 변화 방지

current_cutoffs = initial_cutoffs

for week in weeks:
    user_scores_this_week = collect_user_data(week)
    
    # 백분위 계산
    suggested_cutoffs = calculate_percentile_cutoffs(user_scores_this_week)
    
    # 이동 평균 (smoothing)
    alpha = 0.1  # 10%만 반영
    current_cutoffs["safe"] = (
        alpha * suggested_cutoffs["safe"] + 
        (1 - alpha) * current_cutoffs["safe"]
    )
    current_cutoffs["aggressive"] = (
        alpha * suggested_cutoffs["aggressive"] + 
        (1 - alpha) * current_cutoffs["aggressive"]
    )
```

### 2.2 장점
- ✅ 급격한 변화 방지
- ✅ 이상치에 강건
- ✅ 사용자 혼란 최소화

### 2.3 적용 주기
```
일간: alpha = 0.01  (1% 반영, 매우 안정적)
주간: alpha = 0.1   (10% 반영, 권장)
월간: alpha = 0.3   (30% 반영, 빠른 적응)
```

---

## 🔧 **방법론 3: 클러스터링 기반 자연 분할점 탐색**

### 3.1 개념
```python
from sklearn.cluster import KMeans

# 사용자 점수를 3개 클러스터로 분할
user_scores = np.array(user_scores).reshape(-1, 1)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(user_scores)

# 클러스터 중심
centers = sorted(kmeans.cluster_centers_.flatten())
# 예: [72.5, 82.3, 91.8]

# 자연스러운 분할점 = 중심들의 중간값
aggressive_cutoff = (centers[0] + centers[1]) / 2  # 77.4
safe_cutoff = (centers[1] + centers[2]) / 2        # 87.05
```

### 3.2 장점
- ✅ 사용자 분포의 자연스러운 경계 발견
- ✅ 데이터 기반 객관적 기준
- ✅ 사고 데이터 불필요

### 3.3 고급 버전: 가우시안 혼합 모델 (GMM)
```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(user_scores.reshape(-1, 1))

# 각 그룹의 확률 분포 기반 최적 분할점
# 더 정교한 경계 설정 가능
```

---

## 🔧 **방법론 4: Z-Score 정규화 + 절대 기준 보정**

### 4.1 개념
```python
# 사용자 점수를 표준화
mean_score = np.mean(user_scores)
std_score = np.std(user_scores)

z_scores = (user_scores - mean_score) / std_score

# Z-Score 기준으로 등급 부여
# SAFE: z > 0.5 (상위 30%)
# AGGRESSIVE: z < -1.0 (하위 16%)
# MODERATE: -1.0 ≤ z ≤ 0.5

# 원점수로 역변환
safe_cutoff = mean_score + 0.5 * std_score
aggressive_cutoff = mean_score - 1.0 * std_score
```

### 4.2 절대 기준 보정
```python
# Phase 4 기준과 비교하여 조정
phase4_safe = 88.0

if safe_cutoff < phase4_safe - 10:  # 너무 낮으면
    # "사용자 풀이 전반적으로 위험 운전"
    safe_cutoff = phase4_safe - 5  # 최대 5점만 하락 허용
```

### 4.3 장점
- ✅ 통계적으로 견고
- ✅ 사용자 풀 크기 변화에 안정적
- ✅ 절대 기준 유지 가능

---

## 🔧 **방법론 5: 다층 검증 시스템 (권장 ⭐)**

### 5.1 하이브리드 접근
```python
class MultiLayerGradingSystem:
    """
    여러 방법론을 조합한 안정적인 등급 시스템
    """
    
    def __init__(self, phase4_cutoffs, phase4_weights):
        self.base_cutoffs = phase4_cutoffs  # 절대 기준
        self.weights = phase4_weights
        
    def calculate_grade(self, user_score, user_pool_scores):
        """
        3단계 검증으로 등급 결정
        """
        # Layer 1: 절대 평가 (Phase 4 기준)
        absolute_grade = self._absolute_grading(user_score)
        
        # Layer 2: 상대 평가 (백분위)
        percentile = self._calculate_percentile(user_score, user_pool_scores)
        relative_grade = self._percentile_grading(percentile)
        
        # Layer 3: 안전 검증 (극단적 위험 운전 감지)
        safety_check = self._safety_check(user_score)
        
        # 최종 등급: 3가지 중 가장 보수적(안전한) 기준 선택
        final_grade = self._select_most_conservative(
            absolute_grade, 
            relative_grade, 
            safety_check
        )
        
        return final_grade
    
    def _absolute_grading(self, score):
        """Phase 4 기준으로 절대 평가"""
        if score >= self.base_cutoffs["safe"]:
            return "SAFE"
        elif score <= self.base_cutoffs["aggressive"]:
            return "AGGRESSIVE"
        else:
            return "MODERATE"
    
    def _percentile_grading(self, percentile):
        """백분위 기준 상대 평가"""
        if percentile >= 65:  # 상위 35%
            return "SAFE"
        elif percentile <= 10:  # 하위 10%
            return "AGGRESSIVE"
        else:
            return "MODERATE"
    
    def _safety_check(self, score):
        """극단적 위험 운전 감지"""
        # 점수가 너무 낮으면 무조건 AGGRESSIVE
        if score < 70:
            return "AGGRESSIVE"
        # 점수가 충분히 높으면 SAFE 허용
        elif score >= 90:
            return "SAFE"
        else:
            return "MODERATE"
    
    def _select_most_conservative(self, abs_grade, rel_grade, safety_grade):
        """
        가장 보수적인 등급 선택
        AGGRESSIVE > MODERATE > SAFE 순서로 우선순위
        """
        grades = [abs_grade, rel_grade, safety_grade]
        
        if "AGGRESSIVE" in grades:
            return "AGGRESSIVE"
        elif "MODERATE" in grades:
            return "MODERATE"
        else:
            return "SAFE"
```

### 5.2 동적 컷오프 업데이트
```python
class DynamicCutoffManager:
    """
    사용자 데이터 누적에 따라 컷오프 점진적 조정
    """
    
    def __init__(self, initial_cutoffs):
        self.current_cutoffs = initial_cutoffs
        self.history = []
        self.min_samples_for_update = 10000
        
    def update(self, new_user_scores):
        """
        새 사용자 데이터로 컷오프 업데이트
        """
        if len(new_user_scores) < self.min_samples_for_update:
            return self.current_cutoffs  # 샘플 부족
        
        # 방법 1: 백분위 기반 제안
        suggested_percentile = self._calculate_percentile_cutoffs(new_user_scores)
        
        # 방법 2: 클러스터링 기반 제안
        suggested_clustering = self._calculate_clustering_cutoffs(new_user_scores)
        
        # 방법 3: Z-Score 기반 제안
        suggested_zscore = self._calculate_zscore_cutoffs(new_user_scores)
        
        # 3가지 방법의 중간값 (앙상블)
        safe_cutoff = np.median([
            suggested_percentile["safe"],
            suggested_clustering["safe"],
            suggested_zscore["safe"]
        ])
        
        aggressive_cutoff = np.median([
            suggested_percentile["aggressive"],
            suggested_clustering["aggressive"],
            suggested_zscore["aggressive"]
        ])
        
        # 이동 평균으로 부드럽게 조정
        alpha = 0.1
        self.current_cutoffs["safe"] = (
            alpha * safe_cutoff + 
            (1 - alpha) * self.current_cutoffs["safe"]
        )
        self.current_cutoffs["aggressive"] = (
            alpha * aggressive_cutoff + 
            (1 - alpha) * self.current_cutoffs["aggressive"]
        )
        
        # 안전 장치: 초기 기준에서 ±10점 이내로 제한
        self.current_cutoffs = self._apply_safety_bounds(
            self.current_cutoffs, 
            initial_cutoffs,
            max_deviation=10
        )
        
        # 히스토리 저장
        self.history.append({
            "timestamp": datetime.now(),
            "cutoffs": self.current_cutoffs.copy(),
            "sample_size": len(new_user_scores),
            "mean_score": np.mean(new_user_scores),
            "std_score": np.std(new_user_scores)
        })
        
        return self.current_cutoffs
```

---

## 📊 **실전 구현 로드맵**

### Phase 5-A: 초기 배포 (1-3개월)
```python
strategy = "Phase 4 고정 기준 사용"

cutoffs = {
    "safe": 88.0,        # Phase 4-C 결과
    "aggressive": 77.0
}

# 사용자 데이터 수집만 진행
# 등급은 고정 기준으로 부여
```

**목표**: 10,000명 이상 데이터 확보

---

### Phase 5-B: 검증 및 조정 (3-6개월)
```python
# 수집된 데이터 분석
user_scores = load_user_data()  # 10,000+ 명

# 현재 분포 확인
current_distribution = {
    "SAFE": 0.45,        # 실제 45% (목표 65%와 차이)
    "MODERATE": 0.35,
    "AGGRESSIVE": 0.20   # 실제 20% (목표 10%와 차이)
}

# 원인 분석
print("사용자들이 전반적으로 Phase 4 기준보다 위험 운전")

# 조정 방안
# 방안 1: 컷오프 하향 조정 (safe: 88 → 83, aggressive: 77 → 72)
# 방안 2: 가중치 완화 (각 이벤트 -1점씩 감소)
# 방안 3: 하이브리드 (컷오프 조금 + 가중치 조금)
```

**방법**: 
1. A/B 테스트로 3가지 방안 비교
2. 사용자 피드백 수집
3. 최적 방안 선택

---

### Phase 5-C: 동적 시스템 가동 (6개월~)
```python
# 자동 조정 시스템 가동
adjuster = DynamicCutoffManager(initial_cutoffs=phase4c_cutoffs)

# 매주 업데이트
def weekly_update():
    new_users_this_week = get_users_from_last_week()
    user_scores = [calculate_score(u) for u in new_users_this_week]
    
    updated_cutoffs = adjuster.update(user_scores)
    
    # 변화 로깅
    log_cutoff_change(updated_cutoffs)
    
    # 대시보드 업데이트
    update_monitoring_dashboard(updated_cutoffs, user_scores)
```

**목표**: 
- 분포 안정화 (SAFE 60-70%, AGGRESSIVE 5-15%)
- 자동 조정으로 운영 효율화

---

## 🎯 **실제 사례: 분포 기반 조정 시나리오**

### 시나리오 1: 사용자 풀이 안전 운전
```python
# 수집된 데이터
user_scores = [91, 89, 88, 92, 87, ...]
mean = 89.5, std = 3.2

# Phase 4 고정 기준 사용 시
SAFE: 85% (너무 많음!)
AGGRESSIVE: 2% (너무 적음)

# 조정 필요
safe_cutoff: 88 → 91  (상향)
aggressive_cutoff: 77 → 80 (상향)

# 조정 후 분포
SAFE: 65% ✅
AGGRESSIVE: 10% ✅
```

### 시나리오 2: 사용자 풀이 위험 운전
```python
# 수집된 데이터
user_scores = [75, 72, 78, 71, 80, ...]
mean = 75.2, std = 4.5

# Phase 4 고정 기준 사용 시
SAFE: 8% (너무 적음!)
AGGRESSIVE: 52% (너무 많음!)

# 조정 방안
# 방안 A: 컷오프 하향 (비권장)
#   → 위험 운전을 SAFE로 인정하는 꼴

# 방안 B: 사용자 교육 강화 (권장)
#   → 위험 운전 사용자에게 알림/코칭

# 방안 C: 하이브리드
#   → 컷오프 소폭 하향 (-3점) + 교육 병행
```

---

## ⚠️ **주의사항 및 위험 관리**

### 1. Grade Inflation 방지
```python
# 나쁜 예: 사용자 만족을 위해 기준 계속 낮춤
# "모두가 SAFE면 SAFE의 의미가 없다"

# 해결책: 절대 기준 하한선 설정
ABSOLUTE_MINIMUM = {
    "safe": 80.0,        # Phase 4 기준 - 8점까지만 하락 허용
    "aggressive": 70.0   # Phase 4 기준 - 7점까지만 하락 허용
}
```

### 2. 실제 사고 데이터와의 검증
```python
# 정기적으로 실제 사고 데이터 수집 (보험사 협력 등)
# 조정된 등급과 실제 사고율의 상관관계 확인

if correlation(adjusted_grades, actual_accidents) < 0.15:
    # 상관관계가 약해지면 경고
    alert("등급 시스템이 실제 위험도를 제대로 반영하지 못함!")
    # Phase 4 기준으로 롤백 고려
```

### 3. 투명성 확보
```python
# 사용자에게 등급 기준 변경 공지
notification = f"""
안전운전 등급 기준이 업데이트되었습니다.

SAFE 등급 기준: 88점 → 85점
이유: 전체 사용자의 평균 운전 점수 향상

현재 귀하의 등급: MODERATE (83점)
SAFE 등급까지: 2점 (급정거 1회 감소 시 달성 가능)
"""
```

---

## 📊 **모니터링 대시보드**

### 필수 지표
```python
monitoring_metrics = {
    # 분포 지표
    "safe_ratio": 0.65,
    "moderate_ratio": 0.25,
    "aggressive_ratio": 0.10,
    
    # 컷오프 변화
    "safe_cutoff_history": [88.0, 87.8, 87.5, ...],
    "aggressive_cutoff_history": [77.0, 76.9, 76.8, ...],
    
    # 사용자 점수 분포
    "mean_score": 84.5,
    "median_score": 85.2,
    "std_score": 6.3,
    
    # 시계열 추이
    "weekly_mean_trend": [83.1, 83.8, 84.2, 84.5],
    
    # 실제 사고율 (가능한 경우)
    "safe_accident_rate": 0.05,      # 5% (목표 <10%)
    "aggressive_accident_rate": 0.35  # 35% (목표 >30%)
}
```

---

## 🚀 **최종 권장 방안**

### 단기 (0-6개월): 고정 기준 + 데이터 수집
```python
# Phase 4-C 결과를 고정 기준으로 사용
CUTOFFS = {"safe": 88.0, "aggressive": 77.0}
WEIGHTS = phase4c_weights  # 급정거 -3.5, 급가속 -2.8 등

# 모든 사용자 이벤트 수집 및 저장
# 목표: 10,000명 이상
```

### 중기 (6-12개월): 검증 및 조정
```python
# 수집된 데이터로 백분위/클러스터링 분석
# A/B 테스트로 조정 방안 검증
# 최적 컷오프 도출
```

### 장기 (12개월~): 동적 시스템
```python
# 다층 검증 시스템 가동
grader = MultiLayerGradingSystem(phase4c_cutoffs, phase4c_weights)

# 매주/매월 컷오프 자동 조정
adjuster = DynamicCutoffManager(initial_cutoffs)

# 실시간 모니터링 및 피드백
```

---

## 📚 **참고 논문 및 사례**

### 학계
- "Dynamic Credit Scoring" (금융권 신용등급 동적 조정)
- "Percentile-based Grading Systems" (교육 분야)
- "Adaptive Risk Assessment" (보험업)

### 산업
- **Uber Driver Rating**: 지역별 평균 기준 조정
- **Credit Card Scoring**: 시간에 따른 기준 변경
- **Insurance Telematics**: UBI (Usage-Based Insurance) 동적 요율

---

## ✅ **체크리스트**

### 데이터 수집
- [ ] 사용자별 이벤트 카운트 저장
- [ ] 야간/주간 분리 저장
- [ ] 주행 거리/시간 기록
- [ ] 최소 10,000명 데이터 확보

### 분석 준비
- [ ] 점수 계산 파이프라인 구축
- [ ] 분포 분석 도구 준비
- [ ] 백분위/클러스터링 스크립트
- [ ] 모니터링 대시보드

### 조정 실행
- [ ] A/B 테스트 설계
- [ ] 롤백 계획 수립
- [ ] 사용자 공지 준비
- [ ] 실제 사고 데이터 확보 경로

---

**Phase 5는 Phase 4의 연구를 실제 서비스로 전환하는 핵심 단계입니다!** 🎯

*문서 작성일: 2025-09-30*
