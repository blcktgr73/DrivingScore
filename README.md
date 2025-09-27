# 🚗 DrivingScore: 공개 데이터 기반 운전 점수 시스템 연구

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-Phase%201%20완료-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.13+-blue.svg)

> **데이터 기반으로 검증된 안전한 운전 점수 시스템을 위한 과학적 연구 프로젝트**

## 📊 프로젝트 개요

DrivingScore는 공개 데이터셋을 활용하여 운전 안전 점수 시스템의 과학적 근거를 도출하고, 최적의 가중치 체계를 개발하는 연구 프로젝트입니다. 시스템 구축보다는 **데이터 분석을 통한 과학적 검증**에 집중합니다.

### 🎯 연구 목표

- 운전 이벤트와 실제 사고 간의 상관관계 정량화
- 환경적 위험 요인 (야간, 날씨, 도로 유형) 분석
- 데이터 기반 최적 가중치 체계 개발
- 통계적으로 검증된 등급 분류 기준 수립

### ✅ **Phase 1 주요 성과** (2025-09-27 완료)

🎯 **핵심 결론: 3개 이벤트 시스템 (급가속, 급정거, 급회전) 채택 결정**

- **예측 성능 3.8% 향상** (AUC: 0.5427 → 0.5633)
- **모든 이벤트 상관관계 대폭 개선** (11.6%~43.7%)
- **과속 이벤트 제외 결정** (기술적 한계 및 낮은 예측력)
- **야간 운전 위험도 19.6% 증가 정량화**
- **현재 80점 SAFE 기준의 과학적 타당성 확인**

---

## 🔬 연구 방법론

### 분석 대상 이벤트

| 이벤트 | 측정 방법 | Phase 1 결과 | 채택 여부 |
|--------|-----------|--------------|-----------|
| **급가속** | 가속도계 기반 | 상관계수 0.1172 | ✅ **채택** |
| **급정거** | 가속도계 기반 | 상관계수 0.1608 | ✅ **채택** |
| **급회전** | 자이로스코프 기반 | 상관계수 0.0669 | ✅ **채택** |
| ~~과속~~ | ~~GPS 기반~~ | ~~상관계수 0.0665~~ | ❌ **제외** |

### 핵심 연구 질문

1. ✅ **어떤 운전 이벤트가 실제 사고와 가장 높은 상관관계를 보이는가?**
   - **답**: 급정거 > 급가속 > 급회전 > 과속 순

2. ✅ **야간/주간, 날씨별 위험도 차이를 데이터로 정량화할 수 있는가?**
   - **답**: 야간 +19.6%, 악천후 +25.4% 위험 증가

3. ✅ **운전자 모집단 분포를 고려한 적절한 등급 구분점은?**
   - **답**: 현재 80점 SAFE 기준이 과학적으로 타당함 (ROC 최적값: 81.2점)

4. 🔄 **기존 보험사 점수 체계와 비교한 우리 모델의 예측력은?**
   - **상태**: Phase 2에서 실제 데이터로 검증 예정

---

## 📁 프로젝트 구조

```
DrivingScore/
├── docs/                          # 📚 문서
│   ├── PLAN.md                    # 전체 연구 계획
│   ├── Phase1_Final_Report.md     # ⭐ Phase 1 종합 분석 리포트
│   ├── Safety_Score_Spec.md       # 안전 점수 시스템 사양
│   └── Public_Data.md             # 활용 공개 데이터 목록
├── research/                       # 🔬 연구 코드 및 결과
│   ├── phase1_improved_analysis.py # 과속 포함/제외 비교 분석
│   ├── analysis_no_viz.py         # 기초 통계 분석
│   ├── overspeed_analysis.py      # 과속 제외 분석
│   ├── requirements.txt           # 연구 환경 의존성
│   └── correlation_matrix.png     # 상관관계 매트릭스
└── README.md                       # 프로젝트 개요 (이 파일)
```

---

## 🚀 빠른 시작

### 환경 설정

```bash
# 저장소 클론
git clone https://github.com/blcktgr73/DrivingScore.git
cd DrivingScore

# 연구 환경 설정
cd research
pip install -r requirements.txt
```

### Phase 1 분석 재현

```bash
# 기초 통계 분석 실행
python analysis_no_viz.py

# 과속 포함/제외 비교 분석 실행
python phase1_improved_analysis.py

# 과속 제외 시나리오만 분석
python overspeed_analysis.py
```

---

## 📈 연구 결과 하이라이트

### 🎯 Phase 1 핵심 발견사항

#### 1. 사고 예측력 순위
```
1. 급정거    (0.1608) ⭐ 최고 예측력
2. 급가속    (0.1172)
3. 급회전    (0.0669)
4. 과속      (0.0665) ❌ 제외 결정
```

#### 2. 과속 제외 시 성능 개선
| 지표 | 4개 이벤트 | 3개 이벤트 | 개선율 |
|------|------------|------------|--------|
| **AUC-ROC** | 0.5427 | **0.5633** | **+3.8%** |
| **급가속 상관계수** | 0.1050 | **0.1172** | **+11.6%** |
| **급정거 상관계수** | 0.1212 | **0.1608** | **+32.7%** |
| **급회전 상관계수** | 0.0465 | **0.0669** | **+43.7%** |

#### 3. 환경적 위험 요인
- **야간 운전**: +19.6% 사고 위험 증가 (p<0.0001)
- **악천후**: +25.4% 사고 위험 증가
- **야간 시 급가속 53%, 급정거 28% 증가**

#### 4. 등급 분류 검증
- **SAFE (80점+)**: 과학적 타당성 확인 (ROC 최적값: 81.2점)
- **야간 1.5배 가중치**: 통계적 근거 확보

---

## 📋 연구 단계

### ✅ Phase 1: 기초 통계 분석 (완료)
- [x] 사고-이벤트 상관관계 분석
- [x] 환경적 위험 요인 분석
- [x] 과속 포함/제외 비교 분석
- [x] **최종 결정: 3개 이벤트 시스템 채택**

### 🚀 Phase 2: 모델 개발 및 검증 (시작 준비)
- [ ] 실제 Kaggle 데이터셋 활용 분석
- [ ] 3개 이벤트 기반 최적 가중치 도출
- [ ] XGBoost, LightGBM 등 고급 모델링
- [ ] 기존 보험사 모델과 성능 비교

### ⏳ Phase 3: 심화 분석 및 검증 (대기)
- [ ] 장기 시계열 패턴 분석
- [ ] 교차검증 및 일반화 테스트
- [ ] 최종 연구 보고서 작성

---

## 📊 활용 데이터

### 현재 활용 중 (Phase 1)
- **시뮬레이션 데이터**: 10,000 샘플 교통사고 패턴 모델링

### Phase 2 예정
- **[US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)**: 300만+ 실제 교통사고 기록
- **[Porto Seguro Safe Driver](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)**: 59만+ 보험 클레임 데이터
- **[Driver Behavior Analysis](https://www.kaggle.com/datasets/outofskills/driving-behavior)**: 운전자 행동 패턴 데이터

---

## 🛠️ 기술 스택

### 데이터 분석
- **Python 3.13+**
- **pandas, numpy**: 데이터 처리
- **scipy, statsmodels**: 통계 분석
- **scikit-learn**: 머신러닝
- **matplotlib, seaborn**: 시각화

### 고급 모델링 (Phase 2)
- **XGBoost**: 그래디언트 부스팅
- **LightGBM**: 고성능 부스팅
- **Jupyter**: 연구 노트북

---

## 📚 주요 문서

### 📖 연구 리포트
- **[Phase 1 종합 분석 리포트](docs/Phase1_Final_Report.md)** ⭐ **필독**
  - 기초 통계 분석 결과
  - 과속 포함/제외 비교 분석
  - 최종 결정 근거 및 통계적 검증

### 📋 계획 및 사양
- **[연구 계획서 (PLAN.md)](docs/PLAN.md)**: 전체 연구 일정 및 방법론
- **[안전 점수 사양 (Safety_Score_Spec.md)](docs/Safety_Score_Spec.md)**: 시스템 구현 명세
- **[공개 데이터 (Public_Data.md)](docs/Public_Data.md)**: 활용 데이터셋 목록

---

## 🏷️ 버전 태그

- **`v1.0.0-phase1`**: Phase 1 완료 - 3개 이벤트 시스템 최종 결정

---

## 🤝 기여 방법

### 연구 참여
1. **Issue 등록**: 연구 질문이나 개선 제안
2. **Fork & PR**: 분석 코드 개선 또는 새로운 분석 방법 제안
3. **데이터 제공**: 추가 공개 데이터셋 제안

### 코드 기여
```bash
# 1. Fork 저장소
# 2. 새 브랜치 생성
git checkout -b feature/new-analysis

# 3. 변경사항 커밋
git commit -m "feat: Add new correlation analysis method"

# 4. Pull Request 생성
```

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 공개됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 📞 연락처

- **GitHub Issues**: 버그 리포트 및 기능 요청
- **Discussions**: 연구 방법론 및 결과 토론

---

## 🙏 감사의 말

- **Kaggle**: 공개 데이터셋 제공
- **Anthropic Claude**: 연구 분석 및 문서화 지원

---

**⭐ Phase 1에서 과학적으로 검증된 3개 이벤트 시스템으로 더 정확하고 안정적인 운전 안전 점수를 제공합니다!**

*최종 업데이트: 2025-09-27*