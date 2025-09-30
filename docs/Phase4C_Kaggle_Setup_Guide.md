# Phase 4-C Kaggle 데이터 다운로드 실전 가이드

## 🎯 목적
실제 Kaggle 데이터를 다운로드하고 Phase 4-C를 실행하기 위한 단계별 가이드

---

## 📋 단계 1: Kaggle 계정 및 API 설정

### 1.1 Kaggle 계정 생성
1. https://www.kaggle.com/ 접속
2. 우측 상단 **Register** 클릭
3. 이메일 또는 Google 계정으로 가입
4. 이메일 인증 완료

### 1.2 전화번호 인증 (필수!)
```
⚠️ API 사용을 위해서는 전화번호 인증이 필수입니다!

1. 로그인 후 프로필 클릭
2. Settings → Phone Verification
3. 전화번호 입력 및 인증 코드 확인
4. 인증 완료 확인
```

### 1.3 API 토큰 발급
1. Kaggle 웹사이트 로그인
2. 프로필 클릭 → **Account** 선택
3. **API** 섹션으로 스크롤
4. **Create New API Token** 클릭
5. `kaggle.json` 파일 자동 다운로드

### 1.4 API 토큰 배치 (Windows)

#### PowerShell에서 실행:
```powershell
# 1. .kaggle 디렉토리 생성
New-Item -ItemType Directory -Force -Path $env:USERPROFILE\.kaggle

# 2. 다운로드한 kaggle.json 파일을 복사
# (다운로드 폴더에 있다고 가정)
Copy-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"

# 3. 파일 위치 확인
Get-Item $env:USERPROFILE\.kaggle\kaggle.json

# 4. 권한 설정 (다른 사용자의 접근 제거)
$acl = Get-Acl "$env:USERPROFILE\.kaggle\kaggle.json"
$acl.SetAccessRuleProtection($true, $false)
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule($env:USERNAME, "FullControl", "Allow")
$acl.SetAccessRule($rule)
Set-Acl "$env:USERPROFILE\.kaggle\kaggle.json" $acl
```

---

## 📦 단계 2: Kaggle CLI 설치

### Windows PowerShell:
```powershell
# Python 및 pip 업데이트
python -m pip install --upgrade pip

# Kaggle CLI 설치
pip install kaggle

# 설치 확인
kaggle --version
```

예상 출력:
```
Kaggle API 1.6.x
```

### 환경 변수 설정 (선택):
```powershell
# PowerShell 프로필에 추가 (선택)
notepad $PROFILE

# 다음 줄 추가:
$env:KAGGLE_CONFIG_DIR="$env:USERPROFILE\.kaggle"
```

---

## 📊 단계 3: 데이터 다운로드

### 3.1 US Accidents 데이터셋

#### 데이터셋 정보 확인:
```powershell
kaggle datasets list -s "us accidents"
```

#### 다운로드 (약 3-5GB):
```powershell
# 프로젝트 디렉토리로 이동
cd D:\AIPrj\DrivingScore

# data 디렉토리 생성
New-Item -ItemType Directory -Force -Path data\us_accidents

# 데이터 다운로드
kaggle datasets download -d sobhanmoosavi/us-accidents -p data\us_accidents

# 압축 해제
Expand-Archive -Path data\us_accidents\us-accidents.zip -DestinationPath data\us_accidents -Force

# 파일 확인
Get-ChildItem data\us_accidents
```

예상 파일:
- `US_Accidents_March23.csv` (또는 유사한 이름)
- 크기: 3-5GB

### 3.2 Driver Behavior 데이터셋

#### 다운로드:
```powershell
# 디렉토리 생성
New-Item -ItemType Directory -Force -Path data\driver_behavior

# 데이터 다운로드
kaggle datasets download -d outofskills/driving-behavior -p data\driver_behavior

# 압축 해제
Expand-Archive -Path data\driver_behavior\driving-behavior.zip -DestinationPath data\driver_behavior -Force

# 파일 확인
Get-ChildItem data\driver_behavior
```

### 3.3 추가 센서 데이터 (선택)

#### 검색:
```powershell
kaggle datasets list -s "vehicle sensor"
kaggle datasets list -s "automotive telematics"
```

#### 유용한 데이터셋 예시:
```powershell
# 예시 1: UAH-DriveSet
kaggle datasets download -d outofskills/uah-driveset -p data\additional_sensors

# 예시 2: 기타 운전 행동 데이터
kaggle datasets search "driving behavior" --sort-by updated
```

---

## 🔧 단계 4: 필요한 Python 패키지 설치

### 4.1 requirements.txt 업데이트

`research/requirements.txt`에 추가:
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### 4.2 패키지 설치:
```powershell
cd D:\AIPrj\DrivingScore
pip install -r research\requirements.txt
```

---

## 🚀 단계 5: Phase 4-C 실행

### 5.1 데이터 확인:
```powershell
python research\phase4c_real_data_analysis.py
```

처음 실행 시 데이터 존재 여부를 확인합니다.

### 5.2 전체 분석 실행:

데이터가 준비되면 스크립트가 자동으로:
1. US Accidents 로딩 (샘플: 100,000개)
2. Vehicle Sensor 로딩
3. 매칭 실행 (목표: 50,000-100,000개)
4. 상관관계 분석
5. 가중치 도출
6. 최종 보고서 생성

예상 실행 시간: **4-8시간**

---

## 📈 단계 6: 결과 확인

### 6.1 생성될 파일:
```
research/phase4c_real_results.json       # 전체 결과
research/phase4c_matched_data.csv        # 매칭된 데이터
research/phase4c_weights.json            # 최종 가중치
docs/Phase4C_Final_Report.md             # 최종 보고서
```

### 6.2 주요 지표:
```
매칭 샘플 수: 50,000-100,000개
상관계수:
  - 급정거: 0.20-0.30
  - 급가속: 0.15-0.25
  - 급회전: 0.10-0.20
  - 과속: 0.05-0.15
  
AUC: 0.75-0.85
p-value: <0.0001
```

---

## ⚠️ 문제 해결

### 문제 1: "401 Unauthorized"
```
원인: API 토큰이 없거나 잘못 설정됨

해결:
1. kaggle.json 파일 위치 확인:
   Get-Item $env:USERPROFILE\.kaggle\kaggle.json
   
2. 파일 내용 확인 (JSON 형식):
   Get-Content $env:USERPROFILE\.kaggle\kaggle.json
   
3. 새 토큰 발급 후 다시 배치
```

### 문제 2: "403 Forbidden"
```
원인: 전화번호 인증 미완료

해결:
1. Kaggle 웹사이트 로그인
2. Settings → Phone Verification
3. 전화번호 인증 완료
```

### 문제 3: 다운로드 속도 느림
```
해결:
1. 안정적인 네트워크 사용
2. 여러 번 나눠서 다운로드
3. --unzip 옵션 사용 피하기 (수동 압축 해제)
```

### 문제 4: 메모리 부족
```
원인: 대용량 데이터 로딩

해결:
1. phase4c_real_data_analysis.py에서 sample_size 조정
   load_us_accidents(sample_size=50000)  # 10만 → 5만
   
2. 청크 단위 처리 사용
3. 클라우드 환경 사용 (Google Colab Pro+)
```

### 문제 5: pandas 설치 오류
```
해결:
pip install --upgrade pip setuptools wheel
pip install pandas --no-cache-dir
```

---

## 🌐 클라우드 대안 (권장)

로컬 PC의 메모리가 부족한 경우:

### Google Colab Pro+ 사용:
```python
# Colab 노트북에서:

# 1. Kaggle API 설정
from google.colab import files
files.upload()  # kaggle.json 업로드

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 2. 데이터 다운로드
!kaggle datasets download -d sobhanmoosavi/us-accidents
!unzip us-accidents.zip

# 3. 분석 실행
# (phase4c_real_data_analysis.py 코드 복사)
```

**비용**: $49.99/월  
**메모리**: 52GB RAM  
**처리 시간**: 4-6시간

---

## 📊 예상 비용 및 시간

### 로컬 PC (무료):
```
요구사항:
- RAM: 32GB+
- 저장공간: 50GB SSD
- 처리 시간: 8-24시간

장점: 무료
단점: 시간 오래 걸림, 메모리 부족 위험
```

### Google Colab Pro+ ($50):
```
스펙:
- RAM: 52GB
- GPU: 선택 가능
- 처리 시간: 4-6시간

장점: 안정적, 빠름
단점: 월 $50 비용
```

### AWS/Azure (추정 $300-500):
```
인스턴스: r6i.4xlarge (128GB RAM)
처리 시간: 2-4시간
비용: $2-3/시간 × 10-20시간

장점: 최고 성능, 완전 제어
단점: 설정 복잡, 비용 높음
```

---

## ✅ 최종 체크리스트

### 사전 준비:
- [ ] Kaggle 계정 생성
- [ ] 전화번호 인증 완료
- [ ] API 토큰 발급 및 배치
- [ ] Kaggle CLI 설치 확인

### 데이터 다운로드:
- [ ] US Accidents 다운로드 완료
- [ ] Driver Behavior 다운로드 완료
- [ ] 파일 무결성 확인

### 환경 설정:
- [ ] pandas, numpy, scikit-learn 설치
- [ ] 메모리 32GB+ 확보 (또는 클라우드)
- [ ] 저장공간 50GB+ 확보

### 실행:
- [ ] phase4c_real_data_analysis.py 실행
- [ ] 에러 없이 완료
- [ ] 결과 파일 생성 확인

---

## 🚀 시작하기

```powershell
# 1. Kaggle API 설정
kaggle --version

# 2. 데이터 다운로드
cd D:\AIPrj\DrivingScore
kaggle datasets download -d sobhanmoosavi/us-accidents -p data\us_accidents
kaggle datasets download -d outofskills/driving-behavior -p data\driver_behavior

# 3. 압축 해제
Expand-Archive -Path data\us_accidents\us-accidents.zip -DestinationPath data\us_accidents -Force
Expand-Archive -Path data\driver_behavior\driving-behavior.zip -DestinationPath data\driver_behavior -Force

# 4. 패키지 설치
pip install pandas numpy scikit-learn scipy

# 5. 실행!
python research\phase4c_real_data_analysis.py
```

---

## 📞 지원

문제가 발생하면:
1. `docs/Phase4C_Preparation_Guide.md` 참고
2. Kaggle API 공식 문서: https://github.com/Kaggle/kaggle-api
3. 프로젝트 Issues 등록

**Phase 4-C 실행을 시작하세요!** 🎉
