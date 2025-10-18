#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4F - Sample Data Report 업데이트
매칭정보를 포함한 새로운 양식으로 리포트 생성

작성일: 2025-10-17
"""

import json
import sys

# UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 1. 기존 샘플 데이터 로드
with open('phase4f_data_samples.json', 'r', encoding='utf-8') as f:
    samples_data = json.load(f)

# 2. Combined 데이터 로드
with open('phase4f_combined_20k.json', 'r', encoding='utf-8') as f:
    combined_data = json.load(f)

# 3. sensor_id로 매칭정보 찾기
def find_matching_info(sensor_id, combined_data):
    """sensor_id에 해당하는 매칭정보 찾기"""
    for item in combined_data['data']:
        if 'metadata' in item and 'sensor_id' in item['metadata']:
            if item['metadata']['sensor_id'] == sensor_id:
                return item
        # 매칭된 경우 match_id 확인
        elif 'metadata' in item and 'match_id' in item['metadata']:
            # match_id에서 sensor_id 추출 필요 - 원본 매칭 데이터 확인
            pass
    return None

# 4. 모든 샘플에 대해 매칭정보 추출
enhanced_samples = {
    "risk_samples": [],
    "safe_samples": []
}

print("=" * 100)
print("Phase 4F - Sample Data 매칭정보 추출")
print("=" * 100)
print()

# Risk 샘플 처리
print("Risk Samples:")
for sample in samples_data['risk_samples']:
    sensor_id = sample['id']

    # Combined 데이터에서 찾기
    matched_item = None
    for item in combined_data['data']:
        if 'metadata' in item:
            # sensor_id 직접 매칭 (사고 없음)
            if 'sensor_id' in item['metadata'] and item['metadata']['sensor_id'] == sensor_id:
                matched_item = item
                break
            # match_id로 매칭 필요 - extraction_results에서 확인

    enhanced_sample = sample.copy()
    enhanced_sample['matching_info'] = matched_item['metadata'] if matched_item else None
    enhanced_samples['risk_samples'].append(enhanced_sample)

    print(f"  {sensor_id}: {matched_item is not None}")

print()
print("Safe Samples:")
# Safe 샘플 처리
for sample in samples_data['safe_samples']:
    sensor_id = sample['id']

    # Combined 데이터에서 찾기
    matched_item = None
    for item in combined_data['data']:
        if 'metadata' in item:
            if 'sensor_id' in item['metadata'] and item['metadata']['sensor_id'] == sensor_id:
                matched_item = item
                break

    enhanced_sample = sample.copy()
    enhanced_sample['matching_info'] = matched_item['metadata'] if matched_item else None
    enhanced_samples['safe_samples'].append(enhanced_sample)

    print(f"  {sensor_id}: {matched_item is not None}")

# 5. 매칭 데이터 로드 (사고 발생 케이스)
print()
print("매칭 데이터에서 사고 정보 찾는 중...")

# extraction_results가 아니라 combined에서 직접 찾기
# source가 "risk_accident" 또는 "safe_accident"인 항목 찾기

for item in combined_data['data']:
    if item['source'] in ['risk_accident', 'safe_accident']:
        if 'match_id' in item['metadata']:
            # 매칭 정보 출력
            print(f"  Found accident match: {item['metadata'].get('match_id', 'N/A')}")

# 6. 원본 매칭 데이터를 새로 로드해야 함
# step1 스크립트가 생성한 matched_data가 필요
# 하지만 이 정보는 저장되지 않았으므로, combined 데이터만으로 작업

print()
print("=" * 100)
print("매칭정보 추출 완료")
print("=" * 100)
print()
print("주의: Combined 데이터에는 sensor_id만 저장되어 있습니다.")
print("사고 발생 샘플의 경우 match_id는 있지만 원본 sensor_id 연결이 필요합니다.")
print()
print("해결방법: step1 스크립트를 수정하여 매칭 데이터에 sensor_id를 포함시켜야 합니다.")
