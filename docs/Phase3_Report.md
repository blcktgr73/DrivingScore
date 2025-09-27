# Phase 3 실데이터 검증 보고서 (2025-09-27)

## 1. 데이터 개요
- **출처**: Kaggle [Driver Behavior Analysis](https://www.kaggle.com/datasets/outofskills/driving-behavior)
- **센서 피처**: 가속도(AccX/AccY/AccZ), 자이로(GyroX/GyroY/GyroZ), Class(NORMAL/SLOW/AGGRESSIVE)
- **집계 단위**: 8틱(약 8샘플) 윈도우로 묶어 455개 구간 생성 (train/test 합산)
- **양성 비율**: AGGRESSIVE 다수인 윈도우가 28.6%
- **야간 비중**: Timestamp 기반 더미 플래그로 night ratio ≈ 0.50 (균형 유지)
- **참고 파일**: `research/phase3_real_data_analysis.py`, `research/phase3_results.json`

## 2. 전처리 방법
1. Timestamp 순 정렬 후 `window_id = (timestamp - min) // 8` 로 구간화
2. `AccX > 1.2` → 급가속, `AccX < -1.2` → 급정거, `|GyroZ| > 1.0` → 급회전 이벤트 카운트
3. `sqrt(AccX² + AccY²)` 상위 8%를 overspeeding 으로 간주 (Scenario A 전용)
4. Timestamp 를 5틱 단위로 짝/홀 나눠 주간/야간 근사 (`night_ratio`)
5. 환경 지표: `mean_accel_mag`, `gyro_abs_mean`, `accel_std`
6. 각 윈도우의 최빈 Class 가 AGGRESSIVE 이면 label=1, 그 외 0

## 3. 시나리오별 결과
### 3.1 이벤트 가중치 및 컷오프
| 구분 | Scenario A (과속 포함) | Scenario B (과속 제외) |
| --- | --- | --- |
| 주간 가중치 | 급가속 -1.87 · 급정거 -1.88 · 급회전 -2.15 · 과속 -2.90 | 급가속 -1.49 · 급정거 -1.31 · 급회전 -3.80 |
| 야간 가중치 | 급가속 -1.55 · 급정거 -2.42 · 급회전 -2.45 · 과속 -5.98 | 급가속 -0.69 · 급정거 -1.63 · 급회전 -6.98 |
| Aggressive 컷오프 | ≤ 77.0 | ≤ 80.5 |
| Safe 컷오프 | ≥ 88.0 | ≥ 88.5 |
| SAFE 비중 | 64.84% | 86.81% |
| SAFE 사고율 | 14.58% | 23.04% |
| AGGRESSIVE 비중/사고율 | 24.62% / 64.29% | 4.40% / 80.00% |

### 3.2 모델 성능 (테스트 세트)
| 모델 | Scenario A AUC / F1 | Scenario B AUC / F1 |
| --- | --- | --- |
| Logistic Regression | 0.743 / 0.564 | 0.727 / 0.488 |
| XGBoost | 0.728 / 0.533 | 0.686 / 0.452 |
| LightGBM | 0.680 / 0.492 | 0.638 / 0.418 |

- 과속 이벤트를 포함하면 모든 모델에서 AUC가 0.016~0.042p 개선되고 F1도 0.07~0.08p 상승
- Scenario B는 SAFE 구간이 크게 늘지만(▲21.97%p) SAFE 구간 사고율도 +8.46%p 증가

## 4. 주요 인사이트
1. **실데이터에서도 과속 이벤트 효과 존재**: Scenario A가 모든 모델 지표에서 우위 → Phase 1의 “과속 영향 적음” 결론을 부분 수정할 필요.
2. **야간 가중치의 재조정 필요**: 실데이터 기준 급회전·과속 야간 가중치가 주간 대비 2~3배까지 커져, 주행 시간 정보가 실제 포함되면 재학습 필요.
3. **환경 패널티 해석**: `mean_accel_mag` 증가, `accel_std` 증가가 환경 승수(exp 계수) 상승을 유도하여 변동성이 큰 주행을 추가 감점.
4. **SAFE 컷오프 보정 필요**: Scenario B는 88점 이상이 86%까지 늘지만 사고율 23%로 높아, SAFE 등급 신뢰성 확보를 위해 최소 컷오프 상향(>90) 검토.

## 5. 다음 단계 제안
1. US Accidents 등 외부 사건 데이터와 결합해 야간·기상 계수를 실제 값으로 대체
2. `night_ratio` 근사 대신 실제 시각/위치 정보가 담긴 센서 로그 확보
3. SAFE 등급 사고율을 15% 이하로 낮추기 위한 확률 보정(플랫닝·Platt scaling) 실험
4. Phase 2/3 비교 리포트 통합 및 score migration checklist 작성
