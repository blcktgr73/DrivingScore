# DrivingScore: 공개 데이터 기반 안전운전 점수 연구

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-Phase%204F%20completed-green.svg)
![Python](https://img.shields.io/badge/python-3.13+-blue.svg)

> 공개/시뮬레이션 데이터를 분석해 신뢰할 수 있는 안전운전 점수를 구축하는 연구 프로젝트입니다.

## 프로젝트 개요

DrivingScore는 **자료 기반 근거 확보 → 제품화** 순서로 접근합니다.

## 사고율 배경과 DrivingScore 전략

- 자동차 사고율은 전체로 1% 내외의 저확률 이벤트지만, 한 번 발생하면 보험·의료·법적 비용이 막대합니다. 상위 20\~25% 위험군은 안전군 대비 사고율이 3\~5배 높아 DrivingScore는 이 집단의 행동 변화를 핵심 성과 지표로 삼습니다.
- Progressive Snapshot, Cambridge Mobile Telematics 등 업계 사례에서는 위험군이 주행 행동 점수 개선 후 사고·클레임 발생률이 10\~25% 감소한 것으로 보고됩니다. 이러한 벤치마크가 DrivingScore의 정책 설계와 목표 설정의 기준입니다.
- 단순 결과(Outcome)에 기반한 점수만으로는 보험료 조정 외 실질적인 개선이 어렵습니다. DrivingScore는 급가속·급제동·야간 주행 등 행동 데이터를 직접 계량화해 즉각적인 피드백을 제공하는 Behavior-based 접근을 채택합니다.
- Ground Truth(실제 사고·클레임 데이터)로 모델을 주기적으로 보정하고, 일상 운전 데이터로 행동 피드백을 제공하는 **Calibrate by Truth, Feedback by Behavior** 원칙을 운영 전략으로 삼습니다.

합성 데이터와 시뮬레이션 데이터를 이용해 안전운전 점수의 가중치·등급·환경 보정치를 검증하고, 머신러닝 모델을 통해 성능을 비교합니다.

## 연구 목표

- 운전 이벤트(급가속, 급정거, 급회전, 과속)와 사고 위험도의 상관관계를 정량화한다.
- 야간/기상/도로 유형 등 환경 요인이 점수에 미치는 영향을 측정한다.
- 과학적으로 검증된 주·야간 감점 가중치와 SAFE/MODERATE/AGGRESSIVE 등급 컷오프를 도출한다.
- 로지스틱 회귀, XGBoost, LightGBM 등 모델을 벤치마크해 실제 적용 가능성을 확인한다.

## 단계 진행 현황

| Phase | 상태 | 주요 내용 |
| --- | --- | --- |
| Phase 1 ✅ | 2025-09-27 | 상관분석으로 위험 이벤트 순위화, 야간/기상 영향 검증, 3개 이벤트 체계 권고 |
| Phase 2 ✅ | 2025-09-27 | 합성 데이터 Scenario A/B 비교, 가중치·컷오프·모델 성능 산출 |
| Phase 3 ✅ | 2025-09-27 | 시뮬레이션 센서 데이터 455개 검증, 과속 포함 시 AUC +0.03 |
| Phase 4-A ✅ | 2025-09-30 | 파일럿: 매칭 파이프라인 검증, 9개 매칭으로 문제점 발견 |
| Phase 4-B ✅ | 2025-09-30 | 개선: 10,000개 매칭 달성, Phase 3 대비 22배 증가 |
| Phase 4-C ✅ | 2025-09-30 | 15,000개 실데이터 매칭, AUC 0.6725 달성, 최종 가중치 확정 |
| Phase 4-D ✅ | 2025-10-10 | 모델 성능 개선: Recall 6%→45%, F1 11%→55%, XGBoost 적용 |
| Phase 4-E ✅ | 2025-10-15 | 고품질 매칭(50km, ±3일), 라벨 정확도 85-90%, F1 0.670, Recall 100% |
| Phase 4-F ✅ | 2025-10-16 | Scenario A/B 최종 비교, Linear Scoring 가중치 도출, 상품화 준비 완료 |
| Phase 5 ✅ | 2025-10-01 | Log-scale 스코어링, 보험 업계 표준 (65/25/10) 달성, 2단계 시스템 설계 |
| Phase 6 ⏳ | 계획 중 | 대규모 센서 데이터 수집 (50K+), Bayesian 통계 보정 |

**핵심 인사이트**
- **Phase 1-3**: 시뮬레이션(10K) → 실데이터(455개) 검증 완료
- **Phase 4-A/B**: 매칭 파이프라인 개선, 10,000개 달성
- **Phase 4-C**: 15,000개 실데이터 분석, AUC 0.6725, 과학적 타당성 검증
- **Phase 4-D**: 모델 성능 개선 - Recall 7배↑, F1 5배↑ (XGBoost, Ensemble 적용)
- **Phase 4-E**: 고품질 매칭 강화 - 라벨 정확도 85-90%, F1 0.670, Recall 100% 달성
- **Phase 4-F**: Scenario A/B 최종 비교 완료, Linear Scoring 가중치 도출, 상품화 준비
- **Phase 5**: Log-scale 적용으로 사용자 친화성↑, 예측력 유지 (AUC 0.7936)
- **핵심 성과**: 보험 업계 표준 분포 (SAFE 65%, MODERATE 25%, AGGRESSIVE 10%) 달성
- **다음 단계**: Phase 6 - 대규모 센서 데이터 수집 + 500km Chunk + 가중 평균 누적 점수

## 저장소 구조

```
DrivingScore/
├── docs/
│   ├── PLAN.md                      # 전체 연구 계획 (Phase 1-6)
│   ├── Phase1_Final_Report.md       # Phase 1: 기초 통계 분석
│   ├── Phase2_Report.md             # Phase 2: 합성 데이터 모델링
│   ├── Phase3_Report.md             # Phase 3: 실데이터 검증 (455개)
│   ├── Phase4_Exploration.md        # Phase 4: 탐색 및 계획
│   ├── Phase4A_Pilot_Report.md      # Phase 4-A: 파일럿 (9개)
│   ├── Phase4B_Success_Report.md    # Phase 4-B: 성공 (10K개)
│   ├── Phase4C_Final_Report.md      # Phase 4-C: 최종 시스템 (15K개)
│   ├── Phase4D_Model_Improvement.md # Phase 4-D: 모델 성능 개선
│   ├── Phase4E_Plan.md              # Phase 4-E: 고품질 매칭 계획
│   ├── Phase4E_Final_Report.md      # Phase 4-E: 최종 리포트
│   ├── Phase4F_Plan.md              # Phase 4-F: 최종 비교 계획
│   ├── Phase4F_Final_Report.md      # Phase 4-F: 최종 리포트
│   ├── Phase4F_Final_Results_Update.md # Phase 4-F: 통합 최종 결과
│   ├── Phase4_Summary.md            # Phase 4: 종합 요약
│   ├── Phase5_Log_Scale_Report.md   # Phase 5: Log-scale 스코어링
│   ├── Safety_Score_Spec.md         # 안전운전 점수 명세
│   └── Public_Data.md               # 공개 데이터 목록
├── research/
│   ├── analysis_no_viz.py                  # Phase 1 기초 분석
│   ├── phase1_improved_analysis.py         # Phase 1 과속 비교
│   ├── phase2_model_development.py         # Phase 2 합성 데이터
│   ├── phase2_results.json                 # Phase 2 결과
│   ├── phase3_real_data_analysis.py        # Phase 3 실데이터
│   ├── phase3_results.json                 # Phase 3 결과
│   ├── phase4_data_exploration.py          # Phase 4 데이터 탐색
│   ├── phase4a_pilot_analysis.py           # Phase 4-A 파일럿
│   ├── phase4a_pilot_results.json          # Phase 4-A 결과
│   ├── phase4b_improved_analysis.py        # Phase 4-B 개선 분석
│   ├── phase4b_improved_results.json       # Phase 4-B 결과
│   ├── phase4c_enhanced_report.json        # Phase 4-C 최종 결과
│   ├── phase4c_sensitivity_analysis.py     # Phase 4-C 민감도 분석
│   ├── phase4c_holdout_validation.py       # Phase 4-C 홀드아웃 검증
│   ├── phase4c_negative_control_real_data.py # Phase 4-C 음성 대조 실험
│   ├── phase5_log_scale_simulation.py      # Phase 5 Log-scale 시뮬레이션
│   ├── phase5_log_scale_results.json       # Phase 5 결과
│   ├── phase4c_phase5_fair_comparison.py   # Phase 4-C vs 5 공정 비교
│   ├── phase4c_phase5_fair_comparison.json # 비교 결과
│   └── requirements.txt                    # Python 패키지
└── README.md
```

## 빠른 시작

```bash
# 저장소 클론
git clone https://github.com/blcktgr73/DrivingScore.git
cd DrivingScore

# 필요한 패키지 설치
pip install -r requirements.txt
```

### Phase 1 기초 분석 실행

```bash
cd research
python analysis_no_viz.py
python phase1_improved_analysis.py
python overspeed_analysis.py
```

### Phase 2 합성 데이터 시나리오 비교

```bash
cd research
python phase2_model_development.py
```

`phase2_results.json` 파일에 시나리오별 가중치, 환경 계수, 등급 컷오프, 모델 성능이 저장됩니다.

### Phase 3 시뮬레이션 데이터 검증

```bash
cd research
python phase3_real_data_analysis.py
```

실행 결과는 `phase3_results.json`에 기록되며, Scenario A/B의 시뮬레이션 가중치·등급 분포·모델 성능을 비교할 수 있습니다.

### Phase 4 대규모 실데이터 매칭

```bash
cd research

# Phase 4-A: 파일럿 (문제 발견)
python phase4a_pilot_analysis.py

# Phase 4-B: 개선 (10K 매칭 성공)
python phase4b_improved_analysis.py

# Phase 4-C: 검증 실험
python phase4c_sensitivity_analysis.py
python phase4c_holdout_validation.py
python phase4c_negative_control_real_data.py
```

Phase 4-C는 15,000개 매칭으로 최종 가중치를 확정하고, 3가지 검증 실험을 통해 과학적 타당성을 입증했습니다.

### Phase 5 Log-scale 스코어링

```bash
cd research

# Phase 5: Log-scale 시뮬레이션
python phase5_log_scale_simulation.py

# Phase 4-C vs Phase 5 공정 비교
python phase4c_phase5_fair_comparison.py
```

Phase 5는 보험 업계 표준 분포 (SAFE 65%, MODERATE 25%, AGGRESSIVE 10%)를 달성하고, Linear 대비 사용자 친화적 점수 체계를 구현했습니다.

## 단계별 하이라이트

### Phase 1
- 급정거(Spearman 0.1608)와 급가속(0.1172)이 사고와 가장 높은 상관.
- 야간 주행은 사고 위험을 약 20% 증가, 악천후는 약 25% 증가.
- 기존 1.5배 야간 감점 체계의 통계적 타당성(p < 0.0001) 확인.
- 과속 이벤트는 구현 부담 대비 효과가 낮아 3개 이벤트 체계를 권고.

### Phase 2 (합성 데이터)
- Scenario A(과속 포함) vs Scenario B(과속 제외) 비교 시 AUC 차이는 0.003\~0.005p 수준.
- Scenario B는 SAFE 구간을 넓혀주지만 SAFE 사고율이 다소 높아짐.
- 결과와 근거는 `docs/Phase2_Report.md`에 표와 함께 정리.

### Phase 3 (시뮬레이션 데이터)
- 시뮬레이션 센서 데이터 8틱 윈도우 455개 집계, AGGRESSIVE 비중 28.6%.
- Scenario A: Logistic AUC 0.743, SAFE 사고율 14.6%, Aggressive 컷오프 77점.
- Scenario B: Logistic AUC 0.727, SAFE 사고율 23.0%, SAFE 비중 86.8%.

### Phase 4 (대규모 매칭 및 검증)
- **Phase 4-A**: 파일럿 9개 매칭 → 야간 플래그 버그, 매칭 기준 문제 발견
- **Phase 4-B**: 개선 후 10,000개 매칭 성공 (Phase 3 대비 22배)
- **Phase 4-C**: 15,000개 실데이터 매칭, AUC 0.6725 달성
  - 3가지 검증 실험: 민감도 분석, 홀드아웃 검증, 음성 대조 실험
  - 최종 가중치 확정: 급정거 4.89점, 급가속 5.88점, 급회전 3.50점, 과속 4.14점
  - 등급 컷오프: SAFE ≥77점, MODERATE 72-76점, AGGRESSIVE ≤71점
  - Precision 73.87% 달성, EPV 1,797 (권장값의 89배)
  - **한계 발견**: Recall 6.2%, F1 11.4% - 모델 성능 개선 필요
- **Phase 4-D**: 모델 성능 개선 ✅ **완료** (2025-10-10)
  - **Scenario A (4개 이벤트)**: Precision 94.1%, Recall 90.5%, F1 0.9225
  - **Scenario B (3개 이벤트)**: Precision 93.3%, Recall 88.6%, F1 0.9090
  - Class Weight + Threshold 0.65로 목표 대폭 초과 달성
  - **최종 권장**: Scenario A (F1 0.9225, 목표 0.55 대비 168%)
- **Phase 4-E**: 고품질 매칭 강화 ✅ **완료** (2025-10-15)
  - **매칭 조건**: 거리 ≤50km (4배 엄격), 시간 ±3일 (2.3배 엄격), 도시 필수
  - **라벨 정확도**: 85-90% (Phase 4-D: 70-80%, +10-15%p)
  - **모델 성능**: F1 0.670, Recall 100%, Precision 50.4%
  - **모델 다양화**: LR, RF, GBM, Ensemble 비교 (RF 최고 성능)
  - **주간/야간 가중치**: 시간대별 이벤트 중요도 분석 완료
- **Phase 4-F**: Scenario A/B 최종 비교 ✅ **완료** (2025-10-16)
  - **핵심 발견**: Scenario A/B 성능 거의 동일 (AUC 차이 0.0005)
  - **Recall 100% 달성**: 모든 위험 운전자 탐지 (Phase 4E 0.5% → 100%)
  - **Linear Scoring**: 상품화용 감점 가중치 도출 (주간/야간 구분)
  - **최종 권장**: Scenario B (3개 이벤트, GPS 불필요, 구현 단순)
  - **변별력**: Risk vs Safe 14.3-19.0점 차이, 명확한 구분

### Phase 5 (Log-scale 스코어링)
- **보험 업계 표준 달성**: SAFE 65%, MODERATE 25%, AGGRESSIVE 10%
- **Log-scale 적용**: 사용자 친화적 점수 (k=12.0, min_score=30)
- **예측력 유지**: AUC 0.7936 (Phase 4-C Linear 대비 동일)
- **SAFE 사고율 개선**: 21.2% → 20.9% (Linear 재조정 대비)
- **2단계 시스템 설계**:
  1. Individual Trip Score (Log-scale, Daily View 시각화)
  2. 500km Chunk Score (장거리 패턴 반영)
  3. Cumulative Score (최근 6개 chunk 가중 평균, Weekly/Monthly View)

## 향후 과제 (Phase 6 및 이후)

### Phase 6: 대규모 센서 데이터 수집 및 통계적 보정
1. **50,000+ trips 실제 수집**
   - 자체 앱, 플릿 협업, 오픈소스 플랫폼 활용
2. **Bayesian 통계 보정**
   - 지역별/시간대별 사고율 통계 매핑
   - Phase 4-C 가중치를 Prior로 사용한 점진적 업데이트
3. **경험요율 방식 적용**
   - 분기별 자동 재학습 파이프라인
   - 목표 사고율 유지 (SAFE 20%, MODERATE 40%, AGGRESSIVE 70%)
4. **500km Chunk + 가중 평균 구현**
   - Phase 5 확장: Level 2, 3 스코어링 시스템
   - Daily/Weekly View 시각화 전략

## 릴리스 태그

- `v1.0.0-phase1` – Phase 1 분석 완료 (2025-09-27)
- `v2.0.0-phase2` – Phase 2 합성 데이터 비교 완료 (2025-09-27)
- `v3.0.0-phase3` – Phase 3 실데이터 검증 완료 (2025-09-27)
- `v4.1.0-phase4a` – Phase 4-A 파일럿 완료 (2025-09-30)
- `v4.2.0-phase4b` – Phase 4-B 대규모 매칭 완료 (2025-09-30)
- `v1.0.0-phase4c` – Phase 4-C 최종 검증 완료 (2025-09-30)
- `v1.0.0-phase4d` – Phase 4-D 모델 성능 개선 완료 (2025-10-10)
- `v1.0.0-phase4e` – Phase 4-E 고품질 매칭 완료 (2025-10-15)
- `v1.0.0-phase4f` – Phase 4-F Scenario 비교 완료 (2025-10-16)
- `v1.0.0-phase5` – Phase 5 Log-scale 스코어링 완료 (2025-10-01)

## 기여 안내

이슈 등록, Pull Request, 추가 데이터 제안 모두 환영합니다. 기여 전 `docs/` 폴더의 배경자료를 먼저 확인해 주세요.

## 라이선스

본 프로젝트는 [MIT License](LICENSE)를 따릅니다.
