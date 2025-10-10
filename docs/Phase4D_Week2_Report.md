# Phase 4-D Week 2 보고서: Ensemble 비교

작성일: 2025-10-10
Phase: Phase 4-D-2 (Ensemble)
상태: 완료

---

## Executive Summary

- Week 2(Ensemble)가 Week 1 대비 성능을 추가로 향상시켰습니다.
- 추천: Scenario A + Voting Ensemble (Threshold=0.68)
  - Precision 96.4%, Recall 93.0%, F1 0.9467, AUC 0.9909
  - Week 1 대비 F1 +0.0242 (+2.6%)

---

## Week 1 vs Week 2 비교

### Scenario A (4개 이벤트)
| Week | 모델 | Threshold | Precision | Recall | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|
| Week 1 | Class Weight + T | 0.65 | 94.1% | 90.5% | 0.9225 | N/A |
| Week 2 | Voting Ensemble | 0.68 | 96.4% | 93.0% | 0.9467 | 0.9909 |

### Scenario B (3개 이벤트)
| Week | 모델 | Threshold | Precision | Recall | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|
| Week 1 | Class Weight + T | 0.65 | 93.3% | 88.6% | 0.9090 | N/A |
| Week 2 | Voting Ensemble | 0.6 | 94.3% | 91.0% | 0.9259 | 0.9828 |

---

## 방법
- 데이터: Week 1과 동일한 합성 시나리오 분포
- 처리: SMOTE로 Train 데이터 균형 → Logistic/RandomForest/GradientBoosting 학습 → Soft Voting Ensemble → Threshold 탐색(F1 최대화)

---

## 권장
- 운영 채택: Scenario A + Voting Ensemble (T≈0.68)
- 비고: 과속 이벤트(Scenario A)는 추가 구현 필요(GPS/제한속도), 운영 점수 가중치는 Phase 4-C 값 유지

---

## 산출물 (Artifacts)
- research/phase4d_week2_ensemble.json
- research/phase4d_scenario_comparison.json (Week 1 참조)

참고: 최신 실행 시각 2025-10-10 15:50:41

관련 문서: Phase4D_Week1_Report.md

---

## Validation Tests (Week 2 맥락)

본 절은 데이터/매칭 파이프라인의 신뢰성을 점검하는 3가지 검증 전략 결과를 Week 2 맥락에서 요약합니다. 모델은 Week 2(Ensemble) 권장안을 유지하되, 데이터 측 검증 결과를 참고합니다.

- Sensitivity Analysis (거리/시간 기준 민감도)
  - 추천 매칭 기준: 150km, ±5일 (평균 AUC≈0.6391)
  - 현재 기준(200km, ±7일)도 유지 가능. 다만 시간 기준에는 상대적으로 민감 (변동성 ≥ 0.03)
  - 산출물: research/phase4c_sensitivity_results.json

- Holdout Validation (학습/검증 분할 일반화)
  - 일반화 우수: 학습·검증 AUC 차이 작음(≤0.03) → 과적합 우려 낮음
  - 절대 AUC는 낮음(≤0.60) → 매칭/특징만으로의 예측력 한계. Week 2의 SMOTE+Ensemble로 보완
  - 산출물: research/phase4c_holdout_results.json

- Negative Control Test (음성 대조)
  - 합성 랜덤 매칭 대비: 일부 설정에서 시공간 매칭 이점이 불명확(ΔAUC≈-0.05 내외)
  - 실제 Phase 4-C 결과 기준: 무작위 대비 AUC +0.17p, p<0.001로 매우 유의 → 방법론 타당성 확인
  - 산출물: research/phase4c_negative_control_results.json, research/phase4c_negative_control_real_results.json

권장사항
- 운영: Week 2(Scenario A + Voting Ensemble, T≈0.68) 채택 유지
- 데이터 기준: 200km/±7일 유지 또는 150km/±5일로 정밀도 강화(샘플 수 감소 트레이드오프)
- 추적: Phase 5 데이터 수집 후 위 3가지 검증 재실행하여 분포 변동성/성능 재평가

---

## 모델 개요
- 데이터 전처리: Train에 SMOTE 적용(50:50 균형)으로 클래스 불균형 보정.
- 개별 모델: LogisticRegression, RandomForest, GradientBoosting.
- 앙상블: Soft Voting(확률 평균) + Threshold 탐색(F1 최대화)로 최종 의사결정.
- 평가 지표: Precision/Recall/F1/AUC, Week 1 대비 성능 향상 폭 보고.

---

## 운영 결정

- 채택 시나리오: Scenario B (급가속·급정거·급회전; 과속 제외)
- Week 2 권장: Voting Ensemble, Threshold ≈ 0.60
  - Precision 94.3%, Recall 91.0%, F1 0.9259, AUC 0.9828
- 적용 지침: 과속(Over Speeding) 이벤트 제외, 3개 이벤트로 점수·모델 운용