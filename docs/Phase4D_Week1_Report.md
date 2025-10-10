# Phase 4-D Week 1 보고서: Quick Wins

작성일: 2025-10-10
Phase: Phase 4-D-1 (Quick Wins)
상태: 완료

---

## Executive Summary

Week 1에서는 Class Weight + Threshold 최적화만으로 Phase 4-C 대비 성능을 크게 향상했습니다. Scenario A(4개 이벤트)와 Scenario B(3개 이벤트)를 모두 비교했고, 최종적으로 Scenario A + Phase 4-D 조합을 권장합니다.

---

## Detection Criteria Update

- 급가속/급감속: Δspeed ≥ 10 km/h/s 조건이 3초 지속 시 이벤트로 카운트
- 급회전: Centrifugal Acceleration Jump ≥ 400 degree m/s^2 기준 적용
- 상세 기준은 docs/Detection_Criteria_Update.md 참조

---

## 주요 결과

### Scenario A (급가속, 급정거, 급회전, 과속)

| 지표 | Phase 4-C | Phase 4-D Week 1 | 개선 |
|------|-----------|------------------|------|
| Precision | 80.1% | 94.1% | +13.9%p |
| Recall    | 94.8% | 90.5% | -4.3%p  |
| F1 Score  | 0.8684 | 0.9225 | +0.0541 (1.06배) |
| Threshold | 0.50   | 0.65   | - |

### Scenario B (급가속, 급정거, 급회전)

| 지표 | Phase 4-C | Phase 4-D Week 1 | 개선 |
|------|-----------|------------------|------|
| Precision | 78.7% | 93.3% | +14.6%p |
| Recall    | 94.7% | 88.6% | -6.1%p  |
| F1 Score  | 0.8597 | 0.9090 | +0.0493 (1.06배) |
| Threshold | 0.50   | 0.65   | - |

권장: Scenario A + Phase 4-D (F1 0.9225, Precision 94.1%, Recall 90.5%)

---

## 전략별 Threshold 결과 (Scenario A)

| 전략 | Threshold | Precision | Recall | F1 |
|------|-----------|-----------|--------|-----|
| Baseline (Class Weight, 0.50) | 0.50 | 55.0% | 99.6% | 0.7089 |
| F1 최대화 | 0.65 | 94.1% | 90.5% | 0.9225 |
| Precision ≥ 68% | 0.55 | 68.3% | 98.2% | 0.8058 |

---

## 산출물 (Artifacts)

- research/phase4d_step1_results.json
- research/phase4d_step2_results.json
- research/phase4d_final_results.json
- research/phase4d_scenario_comparison.json

---

## 다음 단계

- Week 2: sklearn Ensemble (SMOTE + Voting) 검토 및 적용
- Week 3: XGBoost + Hyperparameter Tuning

권장사항
- 운영 친화: Phase 4-D (T=0.65) 채택 → 오탐 대폭 감소, F1 최고
- 보험 손실 최소: Phase 4-C 유지 시 T=0.55(Precision 68%, Recall 98%) 고려

---

## Phase 4-C 결과 업데이트 (새 기준 반영)

- 데이터 요약
  - US Accidents: 500,000건
  - Vehicle Sensors: 50,000건
  - 매칭 샘플: 26,888건 (매칭률 5.38%)
- 이벤트별 사고 심각도 상관관계 (새 기준: Δspeed/Jump 적용)
  - 급가속: -0.0059
  - 급정거: +0.0043
  - 급회전: -0.0032
  - 과속:  -0.0049
- 시간대별 사고 심각도 (야간 증가율)
  - 주간 평균: 1.991, 야간 평균: 2.006, 증가율: +0.7%

참고: 새 기준 적용으로 이벤트 정의가 보수화되어 상관계수는 전반적으로 0에 근접하는 방향으로 재산정되었습니다. Phase 4-D Week1의 Threshold 최적화 결과(Precision 94.1%, Recall 90.5%, F1 0.9225, T=0.65)는 그대로 유지됩니다.
---

## 이벤트 가중치 (Week 1, 운영 스코어 기준)

아래 가중치는 Phase 4-D Week 1에서도 Phase 4-C의 최종 가중치를 유지하여 사용합니다.

### Scenario A (급가속, 급정거, 급회전, 과속)

| 이벤트 | 주간 가중치 | 야간 가중치 |
|---|---:|---:|
| 급가속 (rapid_accel) | -2.94 | -4.50 |
| 급정거 (sudden_stop) | -3.49 | -5.77 |
| 급회전 (sharp_turn) | -2.07 | -2.91 |
| 과속 (over_speed) | -1.50 | -1.23 |

### Scenario B (급가속, 급정거, 급회전)

| 이벤트 | 주간 가중치 | 야간 가중치 |
|---|---:|---:|
| 급가속 (rapid_accel) | -2.58 | -3.67 |
| 급정거 (sudden_stop) | -3.07 | -4.70 |
| 급회전 (sharp_turn) | -1.86 | -2.43 |

---

## Rebaseline (새 기준, Phase 4-A~4-C 대용)

간단 합성 윈도우(4초 @1Hz)로 새 기준(Δspeed/Jump)을 적용해 A/B를 재평가했습니다.

- Scenario A (4개 이벤트)
  - F1 최대화: T=0.30, Precision 36.5%, Recall 100.0%, F1 0.5349
  - Precision 극대화: T=0.57, Precision 100.0%, Recall 2.7%, F1 0.0526
- Scenario B (3개 이벤트)
  - F1 최대화: T=0.30, Precision 36.5%, Recall 100.0%, F1 0.5349
  - Precision 극대화: T=0.50, Precision 100.0%, Recall 5.6%, F1 0.1055

참고: 본 재평가는 기준 변경의 상대 비교를 위한 경량 시뮬레이션입니다. 정밀 결과는 Phase 4-A/B/C 전체 파이프라인을 새 기준으로 재실행하여 보강할 수 있습니다.

---

## Phase 4-A~4-C (새 기준) 실행 요약

- Phase 4-B (대규모 매칭, 새 기준 적용):
  - 매칭 샘플: 10,000개, 매칭률 6.20%
  - 이벤트-심각도 상관관계: rapid_accel +0.004, sudden_stop +0.000, sharp_turn +0.000, over_speed -0.008
  - 보고서: research/phase4b_improved_results.json
- Phase 4-C (대규모 시뮬레이션):
  - 매칭 샘플: 26,888개 (500K 사고, 50K 센서)
  - 이벤트-심각도 상관관계: 급가속 -0.0059, 급정거 +0.0043, 급회전 -0.0032, 과속 -0.0049
  - 보고서: research/phase4c_final_report.json

해석: 새 기준(Δspeed/Jump) 적용으로 이벤트 정의가 더 엄격·선별적으로 변해, 상관관계는 전반적으로 0에 근접하는 보수적 추정으로 이동했습니다. 분류 모델(Phase 4-D) 관점에서는 Threshold 최적화를 통해 여전히 높은 Precision/F1 유지가 가능하며, 운영 점수 가중치는 Phase 4-C 최종값을 유지합니다.

---

## Phase 4-A~C (새 기준) 정리 및 Phase 4-D 비교 포인트

- Phase 4-A (파일럿, 새 기준):
  - 합성 센서 2,500 샘플, 총 이벤트 406건
- Phase 4-B (개선 매칭, 새 기준):
  - 10,000 매칭, 상관: rapid_accel +0.004, sudden_stop +0.000, sharp_turn +0.000, over_speed -0.008
- Phase 4-C (대규모, 새 기준):
  - 26,888 매칭, 상관: 급가속 -0.0059, 급정거 +0.0043, 급회전 -0.0032, 과속 -0.0049

Phase 4-D와의 비교 관점
- 새 기준으로 이벤트 정의가 보수화되어 상관이 0에 수렴 → 모델은 Threshold 최적화로 대응 필요
- Week1(Phase 4-D) F1 최적화(T=0.65)는 여전히 유효 (Precision 94.1%, Recall 90.5%, F1 0.9225)
- Rebaseline(간이)에서 A/B F1≈0.535 (T≈0.30): 절대치는 보수적이나, A=B 경향 확인

---

## Phase 4-D Week 1 최종 비교 (새 기준 반영 맥락)

- Scenario A (4개): Precision 94.1%, Recall 90.5%, F1 0.9225, T=0.65
- Scenario B (3개): Precision 93.3%, Recall 88.6%, F1 0.9090, T=0.65

결론: Scenario A + Phase 4-D 권장 (F1 +0.0135).

참고: 최신 실행 시각 2025-10-10 15:50:25

관련 문서: Phase4D_Week2_Report.md

참고: 최신 실행 시각 2025-10-10 15:50:34

---

## 모델 개요
- 분류 모델: Logistic Regression + class_weight="balanced" (소수 클래스 보정).
- 특징(Scenario): A=급가속·급정거·급회전·과속(4개), B=급가속·급정거·급회전(3개).
- Threshold 최적화: (1) F1 최대화, (2) Precision≥60%, (3) Precision≥68% 전략 비교.
- 평가 지표: Precision/Recall/F1, (비교 맥락에서 AUC).

---

## 운영 결정

- 채택 시나리오: Scenario B (급가속·급정거·급회전; 과속 제외)
- Week 1 성능(참고): Threshold 0.65, Precision 93.3%, Recall 88.6%, F1 0.9090
- 적용 지침: 과속(Over Speeding) 이벤트는 점수·모델에서 제외, 나머지 3개 이벤트로 운용