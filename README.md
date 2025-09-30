# DrivingScore: 공개 데이터 기반 안전운전 점수 연구

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-Phase%204B%20completed-green.svg)
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
| Phase 1 ✅ | 2025-09-27 | 상관분석으로 위험 이벤트 순위화, 야간/기상 영향 검증, 3개 이벤트 체계 권고 |
| Phase 2 ✅ | 2025-09-27 | 합성 데이터 Scenario A/B 비교, 가중치·컷오프·모델 성능 산출 |
| Phase 3 ✅ | 2025-09-27 | Kaggle 센서 데이터 455개 검증, 과속 포함 시 AUC +0.03 |
| Phase 4-A ✅ | 2025-09-30 | 파일럿: 매칭 파이프라인 검증, 9개 매칭으로 문제점 발견 |
| Phase 4-B ✅ | 2025-09-30 | 개선: 10,000개 매칭 달성, Phase 3 대비 22배 증가 |
| Phase 4-C ⏳ | 계획 중 | 실제 Kaggle 데이터 100K+ 분석, 최종 시스템 완성 |

**핵심 인사이트**
- **Phase 1-3**: 시뮬레이션(10K) vs 실데이터(455개) 비교로 기초 검증 완료
- **Phase 4-A**: 파일럿에서 야간 플래그 버그 및 매칭 기준 문제 발견
- **Phase 4-B**: 문제 해결로 10,000개 매칭 달성 (Phase 3 대비 22배, Phase 4-A 대비 1,111배)
- **핵심 교훈**: 합성 데이터는 파이프라인 검증용, 실제 의미는 진짜 데이터에서 나옴
- **다음 단계**: Phase 4-C에서 실제 Kaggle 7.7M 사고 + 350K+ 센서 데이터로 최종 검증

## 저장소 구조

```
DrivingScore/
├── docs/
│   ├── PLAN.md                      # 전체 연구 계획 (Phase 1-5)
│   ├── Phase1_Final_Report.md       # Phase 1: 기초 통계 분석
│   ├── Phase2_Report.md             # Phase 2: 합성 데이터 모델링
│   ├── Phase3_Report.md             # Phase 3: 실데이터 검증 (455개)
│   ├── Phase4_Exploration.md        # Phase 4: 탐색 및 계획
│   ├── Phase4A_Pilot_Report.md      # Phase 4-A: 파일럿 (9개)
│   ├── Phase4B_Success_Report.md    # Phase 4-B: 성공 (10K개)
│   ├── Phase4_Summary.md            # Phase 4: 종합 요약
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
│   └── requirements.txt                    # Python 패키지
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

### Phase 4 대규모 실데이터 매칭

```bash
cd research

# Phase 4-A: 파일럿 (문제 발견)
python phase4a_pilot_analysis.py

# Phase 4-B: 개선 (10K 매칭 성공)
python phase4b_improved_analysis.py
```

Phase 4-A는 9개 매칭으로 버그를 발견하고, Phase 4-B는 개선하여 10,000개 매칭을 달성했습니다.

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

### Phase 4 (대규모 매칭)
- **Phase 4-A**: 파일럿 9개 매칭 → 야간 플래그 버그, 매칭 기준 문제 발견
- **Phase 4-B**: 개선 후 10,000개 매칭 성공 (Phase 3 대비 22배, Phase 4-A 대비 1,111배)
- 매칭 품질: 평균 거리 136.7km, 시간 차이 3.5일 (합리적)
- 한계: 합성 데이터로 상관계수 ~0.001 (실제 Kaggle 데이터 필요)

## 향후 과제 (Phase 4-C 및 이후)

### Phase 4-C: 실제 Kaggle 데이터 분석
1. **US Accidents 실제 데이터** (7.7M 건) 다운로드 및 전처리
2. **Vehicle Sensor 실제 데이터** (350K+ 건) 확보 및 통합
3. **50,000-100,000개 실제 매칭** 달성
4. **의미있는 상관관계** (0.15-0.30) 및 가중치 도출
5. **실용적 시스템 완성** (AUC 0.80+, 실제 서비스 적용 가능)

### Phase 5: 실용화 및 확장
1. 한국 교통 데이터 적용 및 지역 특성 반영
2. 실시간 센서 데이터 수집 및 빅데이터 레이크 구축
3. 확률 보정(Platt scaling) 및 SAFE 등급 사고율 15% 이하 달성
4. 실제 서비스 배포 및 A/B 테스트

## 릴리스 태그

- `v1.0.0-phase1` – Phase 1 분석 완료 (2025-09-27)
- `v2.0.0-phase2` – Phase 2 합성 데이터 비교 완료 (2025-09-27)
- `v3.0.0-phase3` – Phase 3 실데이터 검증 완료 (2025-09-27)
- `v4.1.0-phase4a` – Phase 4-A 파일럿 완료 (2025-09-30)
- `v4.2.0-phase4b` – Phase 4-B 대규모 매칭 완료 (2025-09-30, 현재)

## 기여 안내

이슈 등록, Pull Request, 추가 데이터 제안 모두 환영합니다. 기여 전 `docs/` 폴더의 배경자료를 먼저 확인해 주세요.

## 라이선스

본 프로젝트는 [MIT License](LICENSE)를 따릅니다.
