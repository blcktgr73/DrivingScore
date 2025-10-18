#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4F - Sample Data Report 업데이트 v2
매칭정보를 포함한 새로운 양식으로 리포트 생성

작성일: 2025-10-17
"""

import json
import sys

# UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 100)
print("Phase 4F - Sample Data 매칭정보 추출 및 리포트 업데이트")
print("=" * 100)
print()

# 1. 기존 샘플 데이터 로드
with open('phase4f_data_samples.json', 'r', encoding='utf-8') as f:
    samples_data = json.load(f)

# 2. Combined 데이터 로드
with open('phase4f_combined_20k.json', 'r', encoding='utf-8') as f:
    combined_data = json.load(f)

# 3. sensor_id로 매칭정보 매핑 생성
sensor_to_info = {}
for item in combined_data['data']:
    if 'metadata' in item and 'sensor_id' in item['metadata']:
        sensor_id = item['metadata']['sensor_id']
        sensor_to_info[sensor_id] = {
            'label': item['label'],
            'source': item['source'],
            'features': item['features'],
            'metadata': item['metadata']
        }

print(f"총 {len(sensor_to_info)}개의 센서 매칭정보 생성")
print()

# 4. 새로운 양식으로 리포트 생성
def format_sample(sample_id, sample_data, matching_info):
    """샘플 데이터를 새로운 양식으로 포맷"""

    # 기본 정보
    accident = sample_data['accident']
    features = sample_data['features']
    scoring = sample_data['scoring']

    # 사고 여부
    accident_mark = "O" if accident else "X"

    # 이벤트 정보
    events = []
    if features['rapid_accel'] > 0:
        events.append(f"급가속 {features['rapid_accel']}회")
    if features['sudden_stop'] > 0:
        events.append(f"급정거 {features['sudden_stop']}회")
    if features['sharp_turn'] > 0:
        events.append(f"급회전 {features['sharp_turn']}회")
    if features['over_speed'] > 0:
        events.append(f"과속 {features['over_speed']}회")
    event_str = ", ".join(events) if events else "없음"

    # 시간대
    time_period = "야간" if features['is_night'] == 1 else "주간"

    # 매칭 정보
    if matching_info and 'metadata' in matching_info:
        metadata = matching_info['metadata']
        city = metadata.get('city', 'N/A')
        weather = metadata.get('weather', 'N/A')
        grade = scoring['grade']
        distance_km = metadata.get('distance_km', 'N/A')
        time_diff_hours = metadata.get('time_diff_hours', 'N/A')
        severity = metadata.get('severity', 'N/A')

        matching_str = f"{city} | {weather} | {grade}"
        detail_str = f"거리: {distance_km}km | 시간차: {time_diff_hours} 시간 | 심각도: {severity}"
    else:
        matching_str = "매칭정보 없음"
        detail_str = "N/A"

    # 포맷된 문자열 생성
    formatted = f"""ID {sample_id}
사고여부: {accident_mark}
이벤트: {event_str}
시간대: {time_period}
매칭정보: {matching_str}
{detail_str}"""

    return formatted

# 5. Risk 샘플 처리
print("Risk Samples (사고 발생):")
print("-" * 100)
for i, sample in enumerate(samples_data['risk_samples'], 1):
    sensor_id = sample['id']
    matching_info = sensor_to_info.get(sensor_id)

    formatted = format_sample(sensor_id, sample, matching_info)
    print(f"\n{i}. {formatted}\n")

print()
print("=" * 100)
print()

# 6. Safe 샘플 처리
print("Safe Samples (사고 미발생):")
print("-" * 100)
for i, sample in enumerate(samples_data['safe_samples'], 1):
    sensor_id = sample['id']
    matching_info = sensor_to_info.get(sensor_id)

    formatted = format_sample(sensor_id, sample, matching_info)
    print(f"\n{i}. {formatted}\n")

print()
print("=" * 100)
print("✅ 샘플 데이터 리포트 업데이트 완료")
print("=" * 100)
