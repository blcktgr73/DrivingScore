# Safety Score Specification (안전성 점수 사양)

## 1. 목적
안전성(Safety) 지표는 운전 중 위험 이벤트를 기반으로 사용자의 안전 운전 수준을 가시화한다. 본 문서는 **점수 계산 로직**과 **향후 개선 계획**을 정의한다.

---

## 2. 현재 구현 요약
- **데이터 소스**: `DrivingAnalysisRecord`
  - 주요 필드
    - `evnt: Map<Event, Int>` → 일반 시간 이벤트
      - `Event.RAPID_ACCELERATION` (급가속)
      - `Event.SUDDEN_STOP` (급정거)
      - `Event.OVER_SPEEDING` (과속)
      - `Event.SHARP_TURN` (급회전)
    - `nightEvnt: Map<Event, Int>` → 야간 시간 이벤트 (신규)
      - `Event.RAPID_ACCELERATION` (야간 급가속)
      - `Event.SUDDEN_STOP` (야간 급정거)
      - `Event.OVER_SPEEDING` (야간 과속)
      - `Event.SHARP_TURN` (야간 급회전)
    - `sc: Int` → 각 트립의 기본 점수(0~100)
    - `timestamp: Long` → 트립 시작/종료 시간 (야간 판별용)
- **집계 범위**
  - `DailyViewDataMapper`가 금일(또는 최신 날짜)의 트립 목록을 수집
  - 트립 단위 이벤트 카운트를 일 단위로 합산

### 2.1 점수 계산 흐름 
1. 트립 목록을 순회하며 이벤트 카운트 누적
   - 일반 이벤트: `evnt` 맵에서 카운트 수집
   - 야간 이벤트: `nightEvnt` 맵에서 카운트 수집
2. `baseScore = trips.map { it.sc }.average().roundToInt()`
3. **분리된 이벤트별 가중치 적용**
   - 일반 이벤트: `dayPenalty = dayEventCount * 2`
   - 야간 이벤트: `nightPenalty = nightEventCount * 3`
4. 최종 점수: `safetyScore = (baseScore - dayPenalty - nightPenalty).coerceIn(0, 100)`
5. 밴드 계산: `SafetyMetric.calculateBand(score)`
   - 80 이상 → `SAFE`
   - 60 이상 → `MODERATE`
   - 그 외 → `AGGRESSIVE`

### 2.2 UI 반영
- 개별 이벤트 카운트 노출
- **이벤트 카운트 표시**:
  - 일반 이벤트: `evnt` 맵 기반 카운트 표시
  - 야간 이벤트: `nightEvnt` 맵 기반 카운트 표시 (별도 구분)
- **밴드별 표시**:
  - `SAFE` (80-100점): 안전한 운전 스타일
  - `MODERATE` (60-79점): 보통 수준의 운전 스타일
  - `AGGRESSIVE` (0-59점): 공격적인 운전 스타일
- 텍스트/문구는 간단한 수치 위주로 표기 (자연어 서술 미도입)

---

## 3. 분리된 이벤트 기반 가중치 시스템

### 3.1 데이터 구조 변경
- **일반 이벤트**: `evnt: Map<Event, Int>` - 주간 시간대 발생 이벤트
- **야간 이벤트**: `nightEvnt: Map<Event, Int>` - 야간 시간대 발생 이벤트
- **야간 시간 정의**: 일몰 ~ 일출 구간 (위치 기반 천문학적 계산)

### 3.2 이벤트별 가중치 체계
| 이벤트 저장 위치 | 이벤트 유형 | 감점 | 적용 시간 |
|-----------------|-----------|------|-----------|
| `evnt` | 급가속/급정거/과속/급회전 | -2점 | 일반 시간 |
| `nightEvnt` | 급가속/급정거/과속/급회전 | -3점 | 야간 시간 (일몰~일출) |

### 3.3 계산 로직 개선
```kotlin
fun calculateTotalPenalty(evnt: Map<Event, Int>, nightEvnt: Map<Event, Int>): Int {
    val dayPenalty = evnt.values.sum() * 2      // 일반 이벤트 x2
    val nightPenalty = nightEvnt.values.sum() * 3  // 야간 이벤트 x3
    return dayPenalty + nightPenalty
}

// 기존 함수는 더 이상 필요 없음 (이벤트가 이미 분리되어 저장됨)
```

### 3.4 이벤트 분류 및 저장 로직
```kotlin
fun categorizeAndStoreEvent(event: Event, timestamp: Long, gpsLocation: Location) {
    val isNightTime = isNightDriving(timestamp, gpsLocation)

    if (isNightTime) {
        nightEvnt[event] = (nightEvnt[event] ?: 0) + 1
    } else {
        evnt[event] = (evnt[event] ?: 0) + 1
    }
}
```

---

## 4. 기술 부채 및 한계
| 항목        | 현황     | 제한 사항                                           |
| --------- | ------ | ----------------------------------------------- |
| 휴대폰 사용 지표 | 미반영    | `DrivingAnalysisRecord`에 직접 포함돼 있지 않음. 별도 로그 필요 |
| ~~야간 운전 비중~~ | ✅ **완료** | `evnt`/`nightEvnt` 분리된 데이터 구조로 구현 완료              |
| ~~이벤트 가중치~~ | ✅ **완료** | 분리된 이벤트 저장으로 차등 가중치 자동 적용                     |
| 누적 구간 관리  | 미구현    | 500km 구간 기준 점수화 로직 없음                           |
| 데이터 마이그레이션 | **신규 과제** | 기존 단일 `evnt` → `evnt`/`nightEvnt` 분리 마이그레이션 필요     |

---

## 5. 향후 개선 계획

### 5.1 데이터 확장
1. **휴대폰 사용 시간**
   - 운전 중 포그라운드 앱/센서 로그에서 총 사용 분 추출
   - `DrivingAnalysisRecord`와 매핑 가능한 별도 테이블 또는 확장 필드 정의

### 5.2 점수 공식 확대 (분리된 이벤트 기반)
- `evnt`/`nightEvnt` 분리 구조를 활용한 새로운 공식
  ```kotlin
  safetyScore = baseScore
    - calculateTotalPenalty(evnt, nightEvnt)
    - phoneUsageMinutes * w5
  ```

---

## 6. 문서 히스토리
| 버전 | 일자 | 작성자 | 주요 변경 |
|-------|------|--------|------------|
| 0.1 | 2025-09-27 | AI Assistant | 초기 작성 (현재 로직 계획) |
