# Phase 4-C 실제 Kaggle 데이터 분석 준비 가이드

## 🎯 **목표**
실제 Kaggle 데이터 (US Accidents 7.7M + Vehicle Sensor)를 사용하여 최종 검증 및 실용적 가중치 도출

## 📋 **사전 준비 체크리스트**

### 1. Kaggle 계정 설정
```
✅ Kaggle 계정 생성: https://www.kaggle.com/
✅ 휴대폰 인증 완료 (API 사용을 위해 필수)
✅ 프로필 완성
```

### 2. Kaggle API 설정

#### Windows 환경
```powershell
# 1. Kaggle CLI 설치
pip install kaggle

# 2. API 토큰 생성
# Kaggle 웹사이트 → Account → Settings → API → Create New API Token
# kaggle.json 파일 다운로드

# 3. API 토큰 배치
# 다운로드한 kaggle.json을 다음 위치에 복사:
# %USERPROFILE%\.kaggle\kaggle.json
# 예: C:\Users\YourName\.kaggle\kaggle.json

# 4. 권한 설정 (중요!)
# Windows에서는 파일 속성 → 보안에서 다른 사용자 접근 제거
```

#### Linux/Mac 환경
```bash
# 1. Kaggle CLI 설치
pip install kaggle

# 2. API 토큰 배치
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3. 확인
kaggle datasets list
```

### 3. 필요한 Python 패키지
```bash
pip install kaggle pandas numpy scikit-learn
```

---

## 📊 **Phase 4-C 데이터 다운로드**

### Dataset 1: US Accidents
```bash
# 데이터셋 정보 확인
kaggle datasets list -s "us accidents"

# 다운로드 (약 3-5GB, 시간 소요)
kaggle datasets download -d sobhanmoosavi/us-accidents

# 압축 해제
mkdir -p data/us_accidents
unzip us-accidents.zip -d data/us_accidents/

# 예상 파일: US_Accidents_March23.csv (또는 유사)
```

**데이터셋 정보:**
- URL: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
- 규모: ~7.7M 레코드
- 기간: 2016-2023
- 크기: 3-5GB

### Dataset 2: Driver Behavior Analysis
```bash
# 다운로드 (이미 Phase 3에서 사용)
kaggle datasets download -d outofskills/driving-behavior

# 압축 해제
mkdir -p data/driver_behavior
unzip driving-behavior.zip -d data/driver_behavior/
```

### Dataset 3: Additional Vehicle Sensor Data (선택)
```bash
# 추가 센서 데이터 검색
kaggle datasets list -s "vehicle sensor"
kaggle datasets list -s "automotive telematics"
kaggle datasets list -s "driving behavior"

# 유용한 데이터셋 다운로드
# 예시:
# kaggle datasets download -d [dataset-path]
```

---

## 🔧 **데이터 전처리 계획**

### US Accidents 전처리
```python
필요한 컬럼:
- ID: 고유 식별자
- Severity: 사고 심각도 (1-4)
- Start_Time: 사고 발생 시각
- Latitude, Longitude: 위치
- Weather_Condition: 날씨
- Temperature(F): 온도
- Visibility(mi): 가시거리
- [기타 환경 변수들]

전처리 작업:
1. 결측치 처리
2. 시간 파싱 (datetime 변환)
3. 야간/주간 구분 (일출/일몰 기준)
4. 지역 코드 생성 (위도/경도 기반)
5. 샘플링 (전체 사용 or 100K-1M)
```

### Vehicle Sensor 전처리
```python
필요한 데이터:
- Timestamp: 측정 시각
- AccX, AccY, AccZ: 가속도
- GyroX, GyroY, GyroZ: 자이로
- Latitude, Longitude: 위치 (있다면)
- Class/Label: 운전 스타일

전처리 작업:
1. 윈도우 집계 (8틱 or 가변)
2. 이벤트 감지 (급가속/급정거/급회전)
3. 시간대 구분
4. 위치 정보 보완
```

---

## 🔗 **매칭 전략 (Phase 4-B 개선 버전)**

### 전략 1: 지역-시간 매칭
```python
매칭 기준:
- 거리: 200km 이내
- 시간: 사고 전후 ±7일
- 환경: 야간/주간 일치 시 가점

예상 결과:
- 100K 사고 샘플 + 10K 센서 샘플
- 예상 매칭: 50,000-80,000개
```

### 전략 2: 사고 유형별 매칭
```python
사고 유형 분류:
- Rear-end: 급정거 패턴
- Side-swipe: 급회전 패턴  
- Head-on: 과속 패턴
- Single-vehicle: 복합 패턴

매칭:
각 사고 유형에 해당하는 센서 패턴 우선 매칭
```

### 전략 3: 환경 조건 매칭
```python
환경 변수 일치:
- 날씨 조건 (맑음/비/눈)
- 도로 유형 (고속도로/시내)
- 시간대 (출퇴근/한산)

가중치 부여:
환경이 많이 일치할수록 높은 매칭 점수
```

---

## 📈 **예상 처리 시간 및 리소스**

### 시나리오 A: 로컬 PC
```
하드웨어:
- RAM: 32GB 이상
- CPU: 8코어 이상
- 저장공간: 50GB SSD

처리 시간:
- 데이터 다운로드: 30분-2시간
- 전처리: 4-8시간
- 매칭: 12-24시간
- 분석: 2-4시간
총: 2-3일

비용: $0 (전기세 제외)
리스크: 메모리 부족, 처리 시간 오래 걸림
```

### 시나리오 B: 클라우드 (권장)
```
인스턴스:
- AWS r6i.4xlarge (128GB RAM, 16 vCPU)
- 또는 Google Colab Pro+ (유사 스펙)

처리 시간:
- 데이터 준비: 1-2시간
- 전처리: 2-3시간
- 매칭: 4-6시간
- 분석: 1-2시간
총: 8-13시간 (1일 안에 완료)

비용: $300-500
리스크: 낮음, 안정적
```

---

## 🎯 **성공 기준**

### 데이터 품질
```
✅ US Accidents: 100,000개 이상 사용
✅ Vehicle Sensor: 10,000개 이상 사용
✅ 최종 매칭: 50,000개 이상
✅ 매칭 품질: 평균 거리 <150km, 시간 차이 <5일
```

### 분석 결과
```
✅ 상관계수: 0.10-0.30 (통계적으로 유의미)
✅ p-value: <0.0001
✅ AUC: 0.75-0.85
✅ 실용적 가중치: 이벤트별 차등 가중치 도출
```

### 문서화
```
✅ Phase 4-C 실행 보고서
✅ 최종 가중치 매트릭스
✅ 통계적 검증 결과
✅ 실용화 가이드
```

---

## ⚠️ **예상 문제 및 해결책**

### 문제 1: 데이터 다운로드 실패
```
원인: Kaggle API 인증 오류
해결:
1. kaggle.json 위치 확인
2. 파일 권한 확인 (600)
3. 휴대폰 인증 완료 확인
```

### 문제 2: 메모리 부족
```
원인: 7.7M 레코드 전체 로딩
해결:
1. 청크 단위 처리
2. 샘플링 (100K-1M)
3. 클라우드 사용
```

### 문제 3: 매칭률 저조
```
원인: 지역/시간 불일치
해결:
1. 매칭 기준 추가 완화
2. 더 많은 센서 데이터 확보
3. 매칭 알고리즘 개선
```

### 문제 4: 처리 시간 초과
```
원인: 대용량 데이터 처리
해결:
1. 병렬 처리
2. 샘플링
3. 클라우드 사용
```

---

## 📝 **실행 체크리스트**

### 사전 준비
- [ ] Kaggle 계정 생성 및 인증
- [ ] Kaggle API 설정 완료
- [ ] 필요한 패키지 설치
- [ ] 저장 공간 확보 (최소 50GB)

### 데이터 확보
- [ ] US Accidents 다운로드
- [ ] Driver Behavior 다운로드
- [ ] 추가 센서 데이터 확보 (선택)
- [ ] 데이터 무결성 확인

### 분석 실행
- [ ] 전처리 스크립트 실행
- [ ] 매칭 파이프라인 실행
- [ ] 상관관계 분석
- [ ] 통계적 검증
- [ ] 보고서 작성

---

## 🚀 **다음 단계**

### 즉시 실행
1. Kaggle 계정 설정
2. API 키 발급
3. 데이터 다운로드

### 1주차
1. 데이터 전처리
2. 매칭 파이프라인 실행
3. 초기 결과 확인

### 2주차
1. 상세 분석
2. 통계적 검증
3. 가중치 도출

### 3주차
1. 최종 보고서 작성
2. 실용화 방안 수립
3. Phase 4 완료

---

## 📚 **참고 자료**

### Kaggle 데이터셋
- [US Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- [Driver Behavior](https://www.kaggle.com/datasets/outofskills/driving-behavior)

### 문서
- [Kaggle API 공식 문서](https://github.com/Kaggle/kaggle-api)
- Phase 4-A 파일럿 보고서
- Phase 4-B 성공 보고서

### 코드
- `research/phase4b_improved_analysis.py` - 참고 코드
- `research/phase4c_real_data_analysis.py` - 실행 예정

---

*문서 작성일: 2025-09-30*  
*Phase 4-C 시작 예정: Kaggle 데이터 확보 후*  
*예상 완료: 2-3주*
