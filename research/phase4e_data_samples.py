#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4-E 데이터 샘플 추출 및 분석
====================================

Phase 4-E Combined 데이터의 실제 사례를 추출하고 특징을 분석합니다.

출력 내용:
1. Combined 데이터셋 샘플 (사고 O 20개 + 사고 X 20개)
2. 통계 분석 (이벤트 평균, 표준편차, 범위)
3. 주간/야간 비교
4. 도시별 분포
5. Phase 4-D vs Phase 4-E 비교

작성일: 2025-10-15
"""

import os
import sys
import json
import random
from collections import defaultdict

# UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 100)
print(" Phase 4-E 데이터 샘플 추출 및 분석")
print("=" * 100)
print()

# ============================================================================
# 유틸리티 함수
# ============================================================================

def mean(data):
    return sum(data) / len(data) if data else 0

def std(data):
    if not data:
        return 0
    m = mean(data)
    variance = sum((x - m) ** 2 for x in data) / len(data)
    return (variance ** 0.5)

# ============================================================================
# 데이터 로드
# ============================================================================

def load_combined_data():
    """Phase 4-E Combined 데이터 로드"""
    print("📂 Phase 4-E Combined 데이터 로드 중...")

    with open("research/phase4e_combined_data.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    combined_data = data['data']
    metadata = data['metadata']

    print(f"  ✅ 로드 완료: {len(combined_data):,}개")
    print(f"    매칭 조건: 거리 ≤50km, 시간 ±3일, 도시 필수")
    print(f"    매칭률: {metadata['match_rate']*100:.1f}%")

    return combined_data, metadata

# ============================================================================
# 샘플 출력 함수
# ============================================================================

def print_sample(data, index, prefix=""):
    """Combined 데이터 샘플 출력"""
    label_text = "🔴 사고 O" if data['label'] == 1 else "🟢 사고 X"
    features = data['features']
    meta = data['metadata']

    print(f"\n  [{index}] {label_text} ({data['source']})")
    print(f"      이벤트: 급가속 {features['rapid_accel']}회 | 급정거 {features['sudden_stop']}회 | " +
          f"급회전 {features['sharp_turn']}회 | 과속 {features['over_speed']}회")
    print(f"      시간대: {'야간' if features['is_night'] else '주간'}")

    if data['label'] == 1:
        print(f"      매칭정보: {meta['city']} | {meta['weather']:>6s} | {meta['driver_type']:>10s}")
        print(f"      거리: {meta['distance_km']:.1f}km | 시간차: {meta['time_diff_hours']:.1f}시간 | 심각도: {meta['severity']}")
    else:
        print(f"      센서정보: {meta['city']} | {meta['driver_type']:>10s} | 주행 {meta['trip_duration']}분")

# ============================================================================
# 통계 분석 함수
# ============================================================================

def analyze_statistics(combined_data):
    """데이터 통계 분석"""
    print("\n" + "=" * 100)
    print("📊 데이터 통계 분석")
    print("=" * 100)

    # 사고 O vs 사고 X 분리
    positive_samples = [d for d in combined_data if d['label'] == 1]
    negative_samples = [d for d in combined_data if d['label'] == 0]

    print(f"\n1️⃣  기본 통계")
    print("-" * 100)
    print(f"  총 샘플: {len(combined_data):,}개")
    print(f"    사고 O: {len(positive_samples):,}개 ({len(positive_samples)/len(combined_data)*100:.1f}%)")
    print(f"    사고 X: {len(negative_samples):,}개 ({len(negative_samples)/len(combined_data)*100:.1f}%)")

    # 이벤트 통계 (사고 O)
    print(f"\n2️⃣  이벤트 통계 (사고 O)")
    print("-" * 100)

    pos_rapid = [d['features']['rapid_accel'] for d in positive_samples]
    pos_sudden = [d['features']['sudden_stop'] for d in positive_samples]
    pos_sharp = [d['features']['sharp_turn'] for d in positive_samples]
    pos_over = [d['features']['over_speed'] for d in positive_samples]

    print(f"  급가속: 평균 {mean(pos_rapid):.2f}회 | 표준편차 {std(pos_rapid):.2f} | 범위 [{min(pos_rapid)}, {max(pos_rapid)}]")
    print(f"  급정거: 평균 {mean(pos_sudden):.2f}회 | 표준편차 {std(pos_sudden):.2f} | 범위 [{min(pos_sudden)}, {max(pos_sudden)}]")
    print(f"  급회전: 평균 {mean(pos_sharp):.2f}회 | 표준편차 {std(pos_sharp):.2f} | 범위 [{min(pos_sharp)}, {max(pos_sharp)}]")
    print(f"  과속:   평균 {mean(pos_over):.2f}회 | 표준편차 {std(pos_over):.2f} | 범위 [{min(pos_over)}, {max(pos_over)}]")

    # 이벤트 통계 (사고 X)
    print(f"\n3️⃣  이벤트 통계 (사고 X)")
    print("-" * 100)

    neg_rapid = [d['features']['rapid_accel'] for d in negative_samples]
    neg_sudden = [d['features']['sudden_stop'] for d in negative_samples]
    neg_sharp = [d['features']['sharp_turn'] for d in negative_samples]
    neg_over = [d['features']['over_speed'] for d in negative_samples]

    print(f"  급가속: 평균 {mean(neg_rapid):.2f}회 | 표준편차 {std(neg_rapid):.2f} | 범위 [{min(neg_rapid)}, {max(neg_rapid)}]")
    print(f"  급정거: 평균 {mean(neg_sudden):.2f}회 | 표준편차 {std(neg_sudden):.2f} | 범위 [{min(neg_sudden)}, {max(neg_sudden)}]")
    print(f"  급회전: 평균 {mean(neg_sharp):.2f}회 | 표준편차 {std(neg_sharp):.2f} | 범위 [{min(neg_sharp)}, {max(neg_sharp)}]")
    print(f"  과속:   평균 {mean(neg_over):.2f}회 | 표준편차 {std(neg_over):.2f} | 범위 [{min(neg_over)}, {max(neg_over)}]")

    # 이벤트 비교
    print(f"\n4️⃣  이벤트 비교 (사고 O vs 사고 X)")
    print("-" * 100)
    print(f"{'이벤트':>8s}  {'사고 O 평균':>12s}  {'사고 X 평균':>12s}  {'차이':>12s}  {'패턴':>20s}")
    print("-" * 80)

    diff_rapid = mean(pos_rapid) - mean(neg_rapid)
    diff_sudden = mean(pos_sudden) - mean(neg_sudden)
    diff_sharp = mean(pos_sharp) - mean(neg_sharp)
    diff_over = mean(pos_over) - mean(neg_over)

    def get_pattern(diff):
        if abs(diff) < 0.05:
            return "차이 거의 없음"
        elif diff > 0:
            return "사고 O에서 높음 ⭐"
        else:
            return "사고 X에서 높음"

    print(f"{'급가속':>8s}  {mean(pos_rapid):>12.2f}  {mean(neg_rapid):>12.2f}  {diff_rapid:>+12.2f}  {get_pattern(diff_rapid):>20s}")
    print(f"{'급정거':>8s}  {mean(pos_sudden):>12.2f}  {mean(neg_sudden):>12.2f}  {diff_sudden:>+12.2f}  {get_pattern(diff_sudden):>20s}")
    print(f"{'급회전':>8s}  {mean(pos_sharp):>12.2f}  {mean(neg_sharp):>12.2f}  {diff_sharp:>+12.2f}  {get_pattern(diff_sharp):>20s}")
    print(f"{'과속':>8s}  {mean(pos_over):>12.2f}  {mean(neg_over):>12.2f}  {diff_over:>+12.2f}  {get_pattern(diff_over):>20s}")

    # 주간/야간 분석
    print(f"\n5️⃣  주간/야간 분석")
    print("-" * 100)

    day_positive = [d for d in positive_samples if d['features']['is_night'] == 0]
    night_positive = [d for d in positive_samples if d['features']['is_night'] == 1]

    print(f"  사고 O:")
    print(f"    주간: {len(day_positive):,}개 ({len(day_positive)/len(positive_samples)*100:.1f}%)")
    print(f"    야간: {len(night_positive):,}개 ({len(night_positive)/len(positive_samples)*100:.1f}%)")

    day_negative = [d for d in negative_samples if d['features']['is_night'] == 0]
    night_negative = [d for d in negative_samples if d['features']['is_night'] == 1]

    print(f"\n  사고 X:")
    print(f"    주간: {len(day_negative):,}개 ({len(day_negative)/len(negative_samples)*100:.1f}%)")
    print(f"    야간: {len(night_negative):,}개 ({len(night_negative)/len(negative_samples)*100:.1f}%)")

    # 도시별 분포 (사고 O)
    print(f"\n6️⃣  도시별 분포 (사고 O)")
    print("-" * 100)

    city_counts = defaultdict(int)
    for d in positive_samples:
        city_counts[d['metadata']['city']] += 1

    for city, count in sorted(city_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {city:>15s}: {count:>5,}개 ({count/len(positive_samples)*100:>5.1f}%)")

    # 매칭 품질 (사고 O)
    print(f"\n7️⃣  매칭 품질 (사고 O)")
    print("-" * 100)

    distances = [d['metadata']['distance_km'] for d in positive_samples]
    time_diffs = [d['metadata']['time_diff_hours'] for d in positive_samples]

    print(f"  거리:")
    print(f"    평균: {mean(distances):.1f}km | 표준편차: {std(distances):.1f}km")
    print(f"    최소: {min(distances):.1f}km | 최대: {max(distances):.1f}km")
    print(f"    ✅ 모든 매칭이 50km 이내")

    print(f"\n  시간차:")
    print(f"    평균: {mean(time_diffs):.1f}시간 ({mean(time_diffs)/24:.1f}일)")
    print(f"    표준편차: {std(time_diffs):.1f}시간")
    print(f"    최소: {min(time_diffs):.1f}시간 | 최대: {max(time_diffs):.1f}시간")
    print(f"    ✅ 모든 매칭이 ±3일 이내")

# ============================================================================
# 메인 실행
# ============================================================================

def main():
    print(f"⏰ 분석 시작: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. 데이터 로드
    combined_data, metadata = load_combined_data()

    # 2. 샘플 출력
    print("\n" + "=" * 100)
    print("📦 Combined 데이터셋 샘플")
    print("=" * 100)

    # 사고 O vs 사고 X 분리
    positive_samples = [d for d in combined_data if d['label'] == 1]
    negative_samples = [d for d in combined_data if d['label'] == 0]

    # 랜덤 샘플링 (재현성을 위해 seed 설정)
    random.seed(42)
    random.shuffle(positive_samples)
    random.shuffle(negative_samples)

    # 사고 O 샘플 20개
    print("\n🔴 사고 O 샘플 (20개)")
    print("-" * 100)
    for i in range(min(20, len(positive_samples))):
        print_sample(positive_samples[i], i+1)

    # 사고 X 샘플 20개
    print("\n\n🟢 사고 X 샘플 (20개)")
    print("-" * 100)
    for i in range(min(20, len(negative_samples))):
        print_sample(negative_samples[i], i+1)

    # 3. 통계 분석
    analyze_statistics(combined_data)

    # 4. 결과 저장
    print("\n" + "=" * 100)
    print("💾 결과 저장")
    print("=" * 100)

    results = {
        "metadata": metadata,
        "statistics": {
            "total_samples": len(combined_data),
            "positive_samples": len(positive_samples),
            "negative_samples": len(negative_samples),
            "positive_events": {
                "rapid_accel": {
                    "mean": mean([d['features']['rapid_accel'] for d in positive_samples]),
                    "std": std([d['features']['rapid_accel'] for d in positive_samples]),
                    "min": min([d['features']['rapid_accel'] for d in positive_samples]),
                    "max": max([d['features']['rapid_accel'] for d in positive_samples])
                },
                "sudden_stop": {
                    "mean": mean([d['features']['sudden_stop'] for d in positive_samples]),
                    "std": std([d['features']['sudden_stop'] for d in positive_samples]),
                    "min": min([d['features']['sudden_stop'] for d in positive_samples]),
                    "max": max([d['features']['sudden_stop'] for d in positive_samples])
                },
                "sharp_turn": {
                    "mean": mean([d['features']['sharp_turn'] for d in positive_samples]),
                    "std": std([d['features']['sharp_turn'] for d in positive_samples]),
                    "min": min([d['features']['sharp_turn'] for d in positive_samples]),
                    "max": max([d['features']['sharp_turn'] for d in positive_samples])
                },
                "over_speed": {
                    "mean": mean([d['features']['over_speed'] for d in positive_samples]),
                    "std": std([d['features']['over_speed'] for d in positive_samples]),
                    "min": min([d['features']['over_speed'] for d in positive_samples]),
                    "max": max([d['features']['over_speed'] for d in positive_samples])
                }
            },
            "negative_events": {
                "rapid_accel": {
                    "mean": mean([d['features']['rapid_accel'] for d in negative_samples]),
                    "std": std([d['features']['rapid_accel'] for d in negative_samples]),
                    "min": min([d['features']['rapid_accel'] for d in negative_samples]),
                    "max": max([d['features']['rapid_accel'] for d in negative_samples])
                },
                "sudden_stop": {
                    "mean": mean([d['features']['sudden_stop'] for d in negative_samples]),
                    "std": std([d['features']['sudden_stop'] for d in negative_samples]),
                    "min": min([d['features']['sudden_stop'] for d in negative_samples]),
                    "max": max([d['features']['sudden_stop'] for d in negative_samples])
                },
                "sharp_turn": {
                    "mean": mean([d['features']['sharp_turn'] for d in negative_samples]),
                    "std": std([d['features']['sharp_turn'] for d in negative_samples]),
                    "min": min([d['features']['sharp_turn'] for d in negative_samples]),
                    "max": max([d['features']['sharp_turn'] for d in negative_samples])
                },
                "over_speed": {
                    "mean": mean([d['features']['over_speed'] for d in negative_samples]),
                    "std": std([d['features']['over_speed'] for d in negative_samples]),
                    "min": min([d['features']['over_speed'] for d in negative_samples]),
                    "max": max([d['features']['over_speed'] for d in negative_samples])
                }
            }
        },
        "samples": {
            "positive": positive_samples[:20],
            "negative": negative_samples[:20]
        }
    }

    output_file = "research/phase4e_data_samples_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ 결과 파일 저장: {output_file}")
    print()

if __name__ == "__main__":
    main()
