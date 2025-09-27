# DrivingScore: 공개 데이터 기반 안전운전 점수 연구

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-Phase%203%20in%20progress-orange.svg)
![Python](https://img.shields.io/badge/python-3.13+-blue.svg)

> 공개/시뮬레이션 데이터를 분석해 신뢰할 수 있는 안전운전 점수를 구축하는 연구 프로젝트입니다.

## 프로젝트 개요

DrivingScore는 **자료 기반 근거 확보 → 제품화** 순서로 접근합니다. 다양한 공개 데이터(Kaggle 등)와 합성 데이터를 이용해 안전운전 점수의 가중치·등급·환경 보정치를 검증하고, 머신러닝 모델을 통해 성능을 비교합니다.

## 연구 목표

- 운전 이벤트(급가속, 급정거, 급회전, 과속)와 사고 위험도의 상관관계를 정량화한다.
- 야간/기상/도로 유형 등 환경 요인이 점수에 미치는 영향을 측정한다.
- 과학적으로 검증된 주·야간 감점 가중치와 SAFE/MODERATE/AGGRESSIVE 등급 컷오프를 도출한다.
- 로지스틱 회귀, XGBoost, LightGBM 등 모델을 벤치마크해 실제 적용 가능성을 확인한다.

## 단계 진행 현황

| Phase | 상태 | 주요 내용 |
| --- | --- | --- |
| Phase 1 (완료) | 2025-09-27 | 상관분석으로 위험 이벤트 순위화, 야간/기상 영향 검증, 3개 이벤트 체계(급가속·급정거·급회전) 권고 |
| Phase 2 (완료) | 2025-09-27 | 합성 데이터 기반 Scenario A(과속 포함) vs Scenario B(과속 제외) 비교, 가중치·컷오프·모델 성능 산출 |
| Phase 3 (진행 중) | 2025-09-27 | Kaggle 센서 데이터(Driver Behavior)로 실데이터 검증, 과속 포함 시 AUC ≈ +0.03, SAFE 구간 보정 과제 도출 |

**핵심 인사이트**
- 합성 데이터에서는 과속이 상대적으로 영향이 작았지만, 실데이터 검증에서 Scenario A(과속 포함)가 AUC 약 0.03p 향상과 SAFE 사고율 14.6% 달성으로 의미 있는 개선을 보여줌.
- Scenario B(과속 제외)는 구현이 단순하지만 SAFE 사고율이 높아(23%) 확률 보정 및 컷오프 재설계가 필요.
- 야간 급회전/과속 가중치가 주간 대비 2~3배 상승 → 실제 시간·기상·위치 피처 추가가 다음 단계 과제.

## 저장소 구조

```
DrivingScore/
├── docs/
│   ├── PLAN.md                  # 단계별 연구 계획 및 진행 상황
│   ├── Phase1_Final_Report.md   # Phase 1 분석 결과
│   ├── Phase2_Report.md         # Scenario A/B 합성 데이터 비교
│   ├── Phase3_Report.md         # Kaggle 실데이터 검증 요약
│   ├── Safety_Score_Spec.md     # 안전운전 점수 산식 명세(주/야간 가중치)
│   └── Public_Data.md           # 활용 예정 공개 데이터 목록
├── research/
│   ├── analysis_no_viz.py               # Phase 1 기초 분석 스크립트
│   ├── phase1_improved_analysis.py      # 과속 제외 시나리오 비교 (Phase 1 확장)
│   ├── overspeed_analysis.py            # 과속 영향 분석
│   ├── phase2_model_development.py      # Phase 2 합성 데이터 파이프라인
│   ├── phase2_results.json              # Phase 2 결과(JSON)
│   ├── phase3_real_data_analysis.py     # Phase 3 Kaggle 실데이터 파이프라인
│   └── phase3_results.json              # Phase 3 결과(JSON)
└── README.md
```

## 빠른 시작

```bash
# 저장소 클론
git clone https://github.com/blcktgr73/DrivingScore.git
cd DrivingScore

# 필요한 패키지 설치
pip install -r research/requirements.txt
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

### Phase 3 Kaggle 실데이터 검증

```bash
cd research
python phase3_real_data_analysis.py
```

실행 결과는 `phase3_results.json`에 기록되며, Scenario A/B의 실제 가중치·등급 분포·모델 성능을 비교할 수 있습니다.

## 단계별 하이라이트

### Phase 1
- 급정거(Spearman 0.1608)와 급가속(0.1172)이 사고와 가장 높은 상관.
- 야간 주행은 사고 위험을 약 20% 증가, 악천후는 약 25% 증가.
- 기존 1.5배 야간 감점 체계의 통계적 타당성(p < 0.0001) 확인.
- 과속 이벤트는 구현 부담 대비 효과가 낮아 3개 이벤트 체계를 권고.

### Phase 2 (합성 데이터)
- Scenario A(과속 포함) vs Scenario B(과속 제외) 비교 시 AUC 차이는 0.003~0.005p 수준.
- Scenario B는 SAFE 구간을 넓혀주지만 SAFE 사고율이 다소 높아짐.
- 결과와 근거는 `docs/Phase2_Report.md`에 표와 함께 정리.

### Phase 3 (실데이터)
- Kaggle 센서 데이터 8틱 윈도우 455개 집계, AGGRESSIVE 비중 28.6%.
- Scenario A: Logistic AUC 0.743, SAFE 사고율 14.6%, Aggressive 컷오프 77점.
- Scenario B: Logistic AUC 0.727, SAFE 사고율 23.0%, SAFE 비중 86.8%.
- SAFE 사고율을 20% 이하로 낮추려면 예측 확률 ≤ 0.70(약 컷오프 90점)으로 조정하면 SAFE 비중 82.2%, 사고율 19.5% 달성.

## 향후 과제

1. US Accidents / Porto Seguro 등 다른 공개 데이터와 결합해 환경 계수(기상·도로)를 실측값으로 교체.
2. `night_ratio` 근사 대신 실제 시간·위치 정보를 갖춘 로그 확보.
3. SAFE 등급 사고율을 15% 이하로 낮추기 위한 확률 보정(Platt scaling 등)과 컷오프 재설계.
4. Phase 2/3 결합 리포트와 score migration 체크리스트 작성.

## 릴리스 태그

- `v1.0.0-phase1` – Phase 1 분석 완료
- `v2.0.0-phase2` – 합성 데이터 시나리오 비교 완료
- `v3.0.0-phase3` – Kaggle 실데이터 검증 진행 중(현재)

## 기여 안내

이슈 등록, Pull Request, 추가 데이터 제안 모두 환영합니다. 기여 전 `docs/` 폴더의 배경자료를 먼저 확인해 주세요.

## 라이선스

본 프로젝트는 [MIT License](LICENSE)를 따릅니다.
