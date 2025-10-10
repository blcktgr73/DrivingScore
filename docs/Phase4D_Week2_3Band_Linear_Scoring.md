# Phase 4-D Week 2: 3-Band Linear Scoring (Scenario B)

## 목적과 범위
- 목적: Week 2(앙상블 기준) 운영을 가정하되, 사용자 노출용 점수는 선형(Linear) 체계를 유지하며 3개 밴드(SAFE/MODERATE/AGGRESSIVE)로 구분하는 방법을 정의합니다.
- 범위: Scenario B(과속 제외) 전용. 이벤트는 급가속(rapid_accel), 급정거(sudden_stop), 급회전(sharp_turn) 3개만 사용합니다.

## 전제
- 운영 점수는 설명 가능성과 안정성이 중요하므로 선형 가중치 체계를 유지합니다.
- 모델(Week 2 앙상블, T≈0.60)은 백엔드 위험 플래그/심사 보조 등 성능 중심 용도에 사용하며, 점수와 분리 운영합니다.

## 선형 점수 공식
- 기본점수: 100점
- 페널티: 이벤트 카운트 × 주/야 가중치 합산
- 최종점수: `score = clip(100 + Σ penalties, 0, 100)`

가중치(Scenario B, Phase 4-C 확정값):
- 급가속(rapid_accel): 주간 -2.58, 야간 -3.67
- 급정거(sudden_stop): 주간 -3.07, 야간 -4.70
- 급회전(sharp_turn): 주간 -1.86, 야간 -2.43

적용 규칙:
- 주/야 이벤트 카운트를 분리 집계하여 각각의 가중치를 곱합니다.
- 여러 주행(세션/트립)을 합산할 때는 이벤트 총합을 사용하고, 점수는 마지막에 0~100으로 클리핑합니다.

## 3-Band 구분 규칙
고정 컷오프(Phase 4-C 기준)를 기본으로 채택합니다.
- SAFE: `score ≥ 77`
- MODERATE: `72 ≤ score ≤ 76`
- AGGRESSIVE: `score ≤ 71`

권고 대안(선택): 분포 변화가 크면 사용자 분포를 고려한 완만한 재보정 적용
- 목표 분포 예시: SAFE ~60%, MODERATE ~35%, AGGRESSIVE ~5%
- 방법: 최근 N주 사용자 점수의 백분위(예: 60%, 20%)를 참조해 SAFE/AGGRESSIVE 컷오프를 제안하되,
  - 스무딩: `new = 0.2×suggested + 0.8×current`
  - 클램프: 주간 ±1점, 월간 ±3점 이내로만 변경
  - 안전범위: SAFE 70~90, AGGRESSIVE 60~80 외로 벗어나지 않게 제한

## 참조 구현(의사코드)
```python
# inputs
rapid_day, rapid_night = counts['rapid_accel_day'], counts['rapid_accel_night']
sudden_day, sudden_night = counts['sudden_stop_day'], counts['sudden_stop_night']
sharp_day, sharp_night = counts['sharp_turn_day'], counts['sharp_turn_night']

W = {
  'rapid':  {'day': -2.58, 'night': -3.67},
  'sudden': {'day': -3.07, 'night': -4.70},
  'sharp':  {'day': -1.86, 'night': -2.43},
}

penalty = (
  rapid_day  * W['rapid']['day']  + rapid_night  * W['rapid']['night']  +
  sudden_day * W['sudden']['day'] + sudden_night * W['sudden']['night'] +
  sharp_day  * W['sharp']['day']  + sharp_night  * W['sharp']['night']
)
score = max(0, min(100, 100 + penalty))

if score >= 77:
  band = 'SAFE'
elif score >= 72:  # 72~76
  band = 'MODERATE'
else:
  band = 'AGGRESSIVE'
```

## 예시
- 입력(주간 급가속 2회, 야간 급정거 1회, 야간 급회전 1회):
  - 페널티 = 2×(-2.58) + 1×(-4.70) + 1×(-2.43) = -12.29
  - 점수 = 100 - 12.29 = 87.71 → SAFE

## 검증/모니터링 가이드
- 분포 추적: 사용자 점수 분포(평균/백분위)와 밴드 비율을 주간/월간으로 모니터링
- 민원/피드백: 오탐·과벌점 사례를 표본 조사하여 가중치/컷오프 재검토 근거로 축적
- 재보정 조건: 분포 변동이 임계(예: SAFE ±10%p 초과) 또는 정책 변경 시에만 적용

## 모델 연계(Week 2, 참조)
- 위험탐지(백엔드)는 Week 2 앙상블(Scenario B, T≈0.60) 사용
- 사용자 노출 점수는 본 문서의 선형 체계를 유지(설명성/일관성 확보)
- 필요 시 모델 신호로 밴드 경계 미세 조정 가능(±1점 이내, 월 단위)

## 요약
- 선형 점수는 간결하고 설명 가능하며, 3밴드 기준은 Phase 4-C 컷오프(77/72/71)를 기본으로 유지합니다.
- 운영 안정성을 위해 컷오프 변경은 스무딩·클램프·안전범위를 통해 점진적으로 수행합니다.
- 모델(Week 2 앙상블)과의 역할 분담으로 성능과 설명성을 동시에 확보합니다.

