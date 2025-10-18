"""
Phase 4G Step 1: Kaggle 실제 사고 데이터 기반 Combined Dataset 생성

주요 특징:
- 20,000개 trip 생성 (Risk 25%, Safe 75%)
- Kaggle 실제 사고 데이터 매칭 (50km, ±3일, 도시 필수 일치)
- MDPI k-means 연구 기반 이벤트 생성
- 급회전 가상 생성 (급정거의 30-50%)
- Risk:Safe 사고 비율 4:1
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
import random
import sys
import io

# UTF-8 출력 설정 (Windows 한글 문제 해결)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Seed 고정 (재현성)
random.seed(42)
np.random.seed(42)

print("=" * 80)
print("Phase 4G Step 1: 데이터 생성 시작")
print("=" * 80)

# ====================================================================================
# 1. Kaggle 사고 데이터 로드
# ====================================================================================
print("\n[1/6] Kaggle 사고 데이터 로드 중...")

import os
kaggle_file = os.path.join('..', 'data', 'us_accidents', 'US_Accidents_March23.csv')
if not os.path.exists(kaggle_file):
    kaggle_file = os.path.join('data', 'us_accidents', 'US_Accidents_March23.csv')

# 필요한 컬럼만 로드 (메모리 절약)
use_cols = [
    'ID', 'Severity', 'Start_Time', 'Start_Lat', 'Start_Lng',
    'City', 'State', 'Weather_Condition', 'Temperature(F)'
]

# 20만 개만 로드 (처리 속도 향상, 20K 데이터 생성에 충분)
df_accidents = pd.read_csv(kaggle_file, usecols=use_cols, nrows=200000)

# 날짜 파싱
df_accidents['Start_Time'] = pd.to_datetime(df_accidents['Start_Time'], errors='coerce')

# 결측치 제거
df_accidents = df_accidents.dropna(subset=['Start_Time', 'Start_Lat', 'Start_Lng', 'City'])

print(f"✅ 로드 완료: {len(df_accidents):,}개 사고 데이터")
print(f"   도시 수: {df_accidents['City'].nunique()}")
print(f"   날짜 범위: {df_accidents['Start_Time'].min()} ~ {df_accidents['Start_Time'].max()}")

# 상위 도시 추출 (매칭률 향상)
top_cities = df_accidents['City'].value_counts().head(50).index.tolist()
print(f"   상위 50개 도시 사용: {', '.join(top_cities[:10])}...")

# ====================================================================================
# 2. MDPI 기반 이벤트 생성 함수
# ====================================================================================
print("\n[2/6] MDPI 기반 이벤트 생성 함수 정의...")

# MDPI 연구 통계 (docs/MDPI_Harsh_Driving_Events_Study.md)
MDPI_STATS = {
    'harsh_accel_mean': 11.95,     # /100km
    'harsh_brake_mean': 16.39,     # /100km
    'harsh_accel_std': 27.86,
    'harsh_brake_std': 29.76,
    'dangerous_accel_threshold': 48.82,  # K-means Dangerous 그룹
    'dangerous_brake_threshold': 45.40,
}

def generate_mdpi_events(driver_type, distance_km, is_night):
    """
    MDPI k-means 기반 급가속/급정거 생성

    Args:
        driver_type: 'RISK' or 'SAFE'
        distance_km: 주행 거리 (km)
        is_night: 야간 여부

    Returns:
        dict: {'rapid_accel': int, 'sudden_stop': int}
    """
    # 야간 배율
    night_multiplier = 1.5 if is_night else 1.0

    # 그룹별 기본값
    if driver_type == 'RISK':
        # Dangerous 그룹 (K-means 위험 임계값의 85%)
        accel_base = MDPI_STATS['dangerous_accel_threshold'] * 0.85  # 41.50
        brake_base = MDPI_STATS['dangerous_brake_threshold'] * 0.85  # 38.59
    else:  # SAFE
        # Non-Dangerous 그룹 (평균값의 65%)
        accel_base = MDPI_STATS['harsh_accel_mean'] * 0.65  # 7.77
        brake_base = MDPI_STATS['harsh_brake_mean'] * 0.65  # 10.65

    # 야간 조정
    accel_base *= night_multiplier
    brake_base *= night_multiplier

    # 정규분포로 변동성 추가
    accel_per_100km = max(0, np.random.normal(
        accel_base,
        MDPI_STATS['harsh_accel_std']
    ))
    brake_per_100km = max(0, np.random.normal(
        brake_base,
        MDPI_STATS['harsh_brake_std']
    ))

    # 거리 스케일링
    distance_factor = distance_km / 100.0

    return {
        'rapid_accel': int(accel_per_100km * distance_factor),
        'sudden_stop': int(brake_per_100km * distance_factor)
    }

def generate_sharp_turn(sudden_stop_count):
    """
    급회전 생성 (가상)

    근거: 급정거와 급회전은 상관관계가 있다고 가정
    급회전은 급정거보다 덜 빈번함 (30-50% 비율)

    Args:
        sudden_stop_count: 급정거 횟수

    Returns:
        int: 급회전 횟수
    """
    ratio = random.uniform(0.3, 0.5)
    return int(sudden_stop_count * ratio)

def generate_speeding(driver_type, distance_km):
    """
    과속 생성 (Phase 4F 기준 유지)

    Args:
        driver_type: 'RISK' or 'SAFE'
        distance_km: 주행 거리

    Returns:
        int: 과속 횟수
    """
    if driver_type == 'RISK':
        if random.random() < 0.40:  # 40% 확률
            return random.randint(5, 15)
    else:  # SAFE
        if random.random() < 0.08:  # 8% 확률
            return random.randint(1, 3)

    return 0

# ====================================================================================
# 3. Haversine 거리 계산 함수
# ====================================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    두 GPS 좌표 간 거리 계산 (km)
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    km = 6371 * c  # 지구 반지름 6371 km
    return km

# ====================================================================================
# 4. 사고 매칭 함수
# ====================================================================================
def match_accident(trip_data, df_accidents, used_accident_ids, max_distance_km=50, max_days=3):
    """
    Kaggle 사고 데이터와 매칭

    매칭 기준:
    - 도시: 필수 일치 (대소문자 무시)
    - 시간: ±3일 이내
    - 거리: 50km 이내

    Args:
        trip_data: dict (trip 정보)
        df_accidents: DataFrame (Kaggle 사고 데이터)
        used_accident_ids: set (이미 사용된 사고 ID)
        max_distance_km: 최대 거리 (km)
        max_days: 최대 시간 차이 (일)

    Returns:
        dict or None: 매칭된 사고 정보
    """
    city = trip_data['city']
    trip_date = trip_data['datetime']
    trip_lat = trip_data['start_lat']
    trip_lon = trip_data['start_lng']

    # 도시 필터링 (필수 조건, 대소문자 무시)
    city_accidents = df_accidents[df_accidents['City'].str.lower() == city.lower()].copy()

    if len(city_accidents) == 0:
        return None

    # 시간 필터링 (±3일)
    time_min = trip_date - timedelta(days=max_days)
    time_max = trip_date + timedelta(days=max_days)
    city_accidents = city_accidents[
        (city_accidents['Start_Time'] >= time_min) &
        (city_accidents['Start_Time'] <= time_max)
    ]

    if len(city_accidents) == 0:
        return None

    # 거리 계산
    city_accidents['distance_km'] = city_accidents.apply(
        lambda row: haversine_distance(
            trip_lat, trip_lon,
            row['Start_Lat'], row['Start_Lng']
        ),
        axis=1
    )

    # 거리 필터링 (50km 이내)
    city_accidents = city_accidents[city_accidents['distance_km'] <= max_distance_km]

    if len(city_accidents) == 0:
        return None

    # 오버샘플링 방지: 이미 사용된 사고 제외
    city_accidents = city_accidents[~city_accidents['ID'].isin(used_accident_ids)]

    if len(city_accidents) == 0:
        return None

    # 가장 가까운 사고 선택
    nearest = city_accidents.nsmallest(1, 'distance_km').iloc[0]

    return {
        'accident_id': nearest['ID'],
        'severity': int(nearest['Severity']),
        'weather': nearest['Weather_Condition'] if pd.notna(nearest['Weather_Condition']) else 'Clear',
        'temperature': nearest['Temperature(F)'] if pd.notna(nearest['Temperature(F)']) else 70.0,
        'distance_km': nearest['distance_km'],
        'time_diff_days': abs((nearest['Start_Time'] - trip_date).days)
    }

# ====================================================================================
# 5. 20K Combined Dataset 생성
# ====================================================================================
print("\n[3/6] 20,000개 Combined Dataset 생성 중...")

TARGET_TOTAL = 20000
RISK_RATIO = 0.25
SAFE_RATIO = 0.75

risk_count = int(TARGET_TOTAL * RISK_RATIO)  # 5,000
safe_count = int(TARGET_TOTAL * SAFE_RATIO)  # 15,000

print(f"   Risk 그룹: {risk_count:,}개 (25%)")
print(f"   Safe 그룹: {safe_count:,}개 (75%)")

# Risk:Safe 사고 비율 4:1
# Risk 16%, Safe 4% → 전체 약 7%
RISK_ACCIDENT_PROB = 0.16
SAFE_ACCIDENT_PROB = 0.04

print(f"   Risk 사고율: {RISK_ACCIDENT_PROB*100:.1f}%")
print(f"   Safe 사고율: {SAFE_ACCIDENT_PROB*100:.1f}%")
print(f"   비율: {RISK_ACCIDENT_PROB/SAFE_ACCIDENT_PROB:.1f}:1")

combined_data = []
used_accident_ids = set()

# 시작 날짜 (Kaggle 데이터 범위 내 - 실제 데이터 확인 필요)
# Kaggle 데이터: 2016-02-08 ~ 2017-01-26
start_date = datetime(2016, 3, 1)
date_range_days = 300  # 10개월

# 거리 분포 (정규분포, 평균 50km)
distances = np.random.gamma(shape=2, scale=25, size=TARGET_TOTAL)
distances = np.clip(distances, 5, 200)  # 5-200km

for i in range(TARGET_TOTAL):
    # 운전자 타입
    if i < risk_count:
        driver_type = 'RISK'
        accident_prob = RISK_ACCIDENT_PROB
    else:
        driver_type = 'SAFE'
        accident_prob = SAFE_ACCIDENT_PROB

    # 거리
    distance_km = round(distances[i], 1)

    # 시간대 (주간 60%, 야간 40%)
    is_night = random.random() < 0.40
    time_of_day = 'Night' if is_night else 'Day'

    # 날짜 생성
    days_offset = random.randint(0, date_range_days)
    trip_date = start_date + timedelta(days=days_offset)

    # 시간 생성
    if is_night:
        hour = random.choice(list(range(18, 24)) + list(range(0, 6)))
    else:
        hour = random.randint(6, 17)

    trip_datetime = trip_date.replace(hour=hour, minute=random.randint(0, 59))

    # 도시 선택 (상위 50개 도시)
    city = random.choice(top_cities)

    # 시작 GPS (도시별 대략적 좌표, 실제로는 도시 중심 ±0.5도)
    city_data = df_accidents[df_accidents['City'] == city].iloc[0]
    start_lat = city_data['Start_Lat'] + random.uniform(-0.5, 0.5)
    start_lng = city_data['Start_Lng'] + random.uniform(-0.5, 0.5)

    # 이벤트 생성
    events = generate_mdpi_events(driver_type, distance_km, is_night)
    rapid_accel = events['rapid_accel']
    sudden_stop = events['sudden_stop']
    sharp_turn = generate_sharp_turn(sudden_stop)
    speeding = generate_speeding(driver_type, distance_km)

    # Trip 데이터
    trip = {
        'trip_id': f"T{i+1:05d}",
        'driver_type': driver_type,
        'city': city,
        'datetime': trip_datetime,
        'start_lat': start_lat,
        'start_lng': start_lng,
        'distance_km': distance_km,
        'time_of_day': time_of_day,
        'rapid_accel': rapid_accel,
        'sudden_stop': sudden_stop,
        'sharp_turn': sharp_turn,
        'speeding': speeding,
    }

    # 사고 매칭 시도
    accident = None

    # 확률에 따라 사고 매칭 시도
    if random.random() < accident_prob:
        accident = match_accident(trip, df_accidents, used_accident_ids)

        if accident:
            used_accident_ids.add(accident['accident_id'])

    # 결과 저장
    if accident:
        trip['has_accident'] = 1
        trip['accident_id'] = accident['accident_id']
        trip['severity'] = accident['severity']
        trip['weather'] = accident['weather']
        trip['temperature'] = accident['temperature']
        trip['match_distance_km'] = accident['distance_km']
        trip['match_time_diff_days'] = accident['time_diff_days']
    else:
        trip['has_accident'] = 0
        trip['accident_id'] = None
        trip['severity'] = None
        trip['weather'] = 'Clear'
        trip['temperature'] = 70.0
        trip['match_distance_km'] = None
        trip['match_time_diff_days'] = None

    combined_data.append(trip)

    if (i + 1) % 5000 == 0:
        print(f"   진행: {i+1:,}/{TARGET_TOTAL:,} ({(i+1)/TARGET_TOTAL*100:.1f}%)")

print(f"✅ 생성 완료: {len(combined_data):,}개")

# ====================================================================================
# 6. 통계 확인
# ====================================================================================
print("\n[4/6] 통계 확인 중...")

df = pd.DataFrame(combined_data)

# 그룹별 통계
risk_data = df[df['driver_type'] == 'RISK']
safe_data = df[df['driver_type'] == 'SAFE']

risk_accident_count = risk_data['has_accident'].sum()
safe_accident_count = safe_data['has_accident'].sum()

risk_accident_rate = risk_accident_count / len(risk_data) * 100
safe_accident_rate = safe_accident_count / len(safe_data) * 100

total_accident_count = df['has_accident'].sum()
total_accident_rate = total_accident_count / len(df) * 100

print(f"\n📊 사고 통계:")
print(f"   Risk 그룹: {risk_accident_count}/{len(risk_data)} ({risk_accident_rate:.2f}%)")
print(f"   Safe 그룹: {safe_accident_count}/{len(safe_data)} ({safe_accident_rate:.2f}%)")
print(f"   전체: {total_accident_count}/{len(df)} ({total_accident_rate:.2f}%)")
if safe_accident_rate > 0:
    print(f"   Risk/Safe 비율: {risk_accident_rate/safe_accident_rate:.2f}:1 (목표 4:1)")
else:
    print(f"   Risk/Safe 비율: N/A (Safe 사고 없음)")

# 오버샘플링 확인
matched_accidents = df[df['has_accident'] == 1]['accident_id'].dropna()
unique_accidents = matched_accidents.nunique()
total_matched = len(matched_accidents)

print(f"\n🔄 오버샘플링 확인:")
print(f"   매칭된 사고 수: {total_matched}")
print(f"   고유 사고 ID: {unique_accidents}")
if total_matched > 0:
    print(f"   중복률: {(total_matched - unique_accidents) / total_matched * 100:.2f}%")
else:
    print(f"   중복률: N/A (매칭 없음)")

# 이벤트 통계
print(f"\n🚗 이벤트 통계 (평균):")
print(f"   Risk 그룹:")
print(f"      급가속: {risk_data['rapid_accel'].mean():.2f}회")
print(f"      급정거: {risk_data['sudden_stop'].mean():.2f}회")
print(f"      급회전: {risk_data['sharp_turn'].mean():.2f}회")
print(f"      과속: {risk_data['speeding'].mean():.2f}회")
print(f"   Safe 그룹:")
print(f"      급가속: {safe_data['rapid_accel'].mean():.2f}회")
print(f"      급정거: {safe_data['sudden_stop'].mean():.2f}회")
print(f"      급회전: {safe_data['sharp_turn'].mean():.2f}회")
print(f"      과속: {safe_data['speeding'].mean():.2f}회")

# MDPI 기대값과 비교 (100km당)
risk_accel_per_100km = risk_data['rapid_accel'].sum() / risk_data['distance_km'].sum() * 100
risk_brake_per_100km = risk_data['sudden_stop'].sum() / risk_data['distance_km'].sum() * 100
safe_accel_per_100km = safe_data['rapid_accel'].sum() / safe_data['distance_km'].sum() * 100
safe_brake_per_100km = safe_data['sudden_stop'].sum() / safe_data['distance_km'].sum() * 100

print(f"\n📈 100km당 이벤트 (MDPI 비교):")
print(f"   Risk 급가속: {risk_accel_per_100km:.2f} (목표 41.5)")
print(f"   Risk 급정거: {risk_brake_per_100km:.2f} (목표 38.6)")
print(f"   Safe 급가속: {safe_accel_per_100km:.2f} (목표 7.8)")
print(f"   Safe 급정거: {safe_brake_per_100km:.2f} (목표 10.7)")

# ====================================================================================
# 7. JSON 저장
# ====================================================================================
print("\n[5/6] JSON 파일 저장 중...")

output_file = 'phase4g_combined_20k.json'

# datetime을 문자열로 변환
for trip in combined_data:
    trip['datetime'] = trip['datetime'].strftime('%Y-%m-%d %H:%M:%S')

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, indent=2, ensure_ascii=False)

print(f"✅ 저장 완료: {output_file}")

# ====================================================================================
# 8. 샘플 출력
# ====================================================================================
print("\n[6/6] 샘플 데이터 출력...")

print("\n📋 Risk + 사고 샘플:")
risk_accident = df[(df['driver_type'] == 'RISK') & (df['has_accident'] == 1)]
if len(risk_accident) > 0:
    sample = risk_accident.iloc[0]
    print(f"ID: {sample['trip_id']}")
    print(f"사고여부: O")
    print(f"이벤트: 급가속 {sample['rapid_accel']}회, 급정거 {sample['sudden_stop']}회, " +
          f"급회전 {sample['sharp_turn']}회, 과속 {sample['speeding']}회")
    print(f"시간대: {sample['time_of_day']}")
    print(f"도시: {sample['city']}")
    print(f"날씨: {sample['weather']}")
    print(f"거리: {sample['distance_km']:.1f}km")
    print(f"심각도: {int(sample['severity'])} (심각)")

print("\n📋 Safe + 사고 샘플:")
safe_accident = df[(df['driver_type'] == 'SAFE') & (df['has_accident'] == 1)]
if len(safe_accident) > 0:
    sample = safe_accident.iloc[0]
    print(f"ID: {sample['trip_id']}")
    print(f"사고여부: O")
    print(f"이벤트: 급가속 {sample['rapid_accel']}회, 급정거 {sample['sudden_stop']}회, " +
          f"급회전 {sample['sharp_turn']}회, 과속 {sample['speeding']}회")
    print(f"시간대: {sample['time_of_day']}")
    print(f"도시: {sample['city']}")
    print(f"날씨: {sample['weather']}")
    print(f"거리: {sample['distance_km']:.1f}km")
    print(f"심각도: {int(sample['severity'])} (심각)")

print("\n" + "=" * 80)
print("Phase 4G Step 1 완료!")
print("=" * 80)
print(f"\n다음 단계: python phase4g_step2_data_report.py")
