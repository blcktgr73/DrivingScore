#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4D vs 4F 교차 검증 분석
==============================

Phase 4D의 이상적인 모델을 Phase 4F의 실제 데이터에 적용하여
다음을 분석합니다:

1. Phase 4D 모델로 Phase 4F 데이터 예측
2. 운전 점수 계산 및 분류 (SAFE/MODERATE/AGGRESSIVE)
3. 점수 그룹별 실제 사고율 분석
4. 합성 vs 실제 데이터 성능 차이 분석

핵심 질문:
- 이상적인 환경(Phase 4D)에서 학습한 모델이 실제 데이터(Phase 4F)에서도 작동하는가?
- 운전 점수가 실제 사고율을 잘 예측하는가?
- SAFE/MODERATE/AGGRESSIVE 분류가 현실과 일치하는가?

작성일: 2025-10-16
"""

import json
import sys
import math
from datetime import datetime
from collections import defaultdict, Counter

# UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 100)
print(" Phase 4D vs 4F 교차 검증 분석")
print("=" * 100)
print()

# ============================================================================
# 데이터 로드
# ============================================================================

print("[데이터 로드] Phase 4D 모델 및 Phase 4F 데이터 로드 중...")

# Phase 4D 모델 (이상적인 합성 데이터로 학습)
with open('phase4d_final_results.json', 'r', encoding='utf-8') as f:
    phase4d_results = json.load(f)

# Phase 4F 실제 데이터
with open('phase4f_combined_20k.json', 'r', encoding='utf-8') as f:
    phase4f_data = json.load(f)

print(f"  Phase 4D 모델: {phase4d_results['phase4d_best']['method']}")
print(f"  Phase 4F 데이터: {len(phase4f_data['data']):,}개 샘플")
print()

# ============================================================================
# Phase 4D 모델 구현
# ============================================================================

print("[모델 준비] Phase 4D 스타일 운전 점수 계산 시스템 구축 중...")

# Phase 4D의 가중치 (Phase 4C 기반 - 실제 데이터로 검증된 가중치)
# Phase 4C Final Report 참조
WEIGHTS_PHASE4D = {
    'rapid_accel': 2.5,   # 급가속
    'sudden_stop': 3.0,   # 급정거 (가장 위험)
    'sharp_turn': 2.0,    # 급회전
    'over_speed': 1.5     # 과속
}

NIGHT_MULTIPLIER = 1.5  # 야간 가중치

# 점수 범위 (Phase 5 Log-scale 기반)
SCORE_RANGES = {
    'SAFE': (65, 100),        # 65점 이상
    'MODERATE': (25, 64),     # 25-64점
    'AGGRESSIVE': (0, 24)     # 0-24점
}

def calculate_driving_score_phase4d(sample):
    """
    Phase 4D 스타일 운전 점수 계산

    점수 = 100 - (이벤트 감점)
    감점 = Σ(이벤트 횟수 × 가중치) × 야간 배율
    """
    features = sample['features']

    # 기본 감점 계산
    deduction = (
        features['rapid_accel'] * WEIGHTS_PHASE4D['rapid_accel'] +
        features['sudden_stop'] * WEIGHTS_PHASE4D['sudden_stop'] +
        features['sharp_turn'] * WEIGHTS_PHASE4D['sharp_turn'] +
        features['over_speed'] * WEIGHTS_PHASE4D['over_speed']
    )

    # 야간 가중치 적용
    if features['is_night'] == 1:
        deduction *= NIGHT_MULTIPLIER

    # 점수 계산 (0-100 범위로 제한)
    score = max(0, min(100, 100 - deduction))

    return score

def classify_driver(score):
    """운전자 분류"""
    if score >= SCORE_RANGES['SAFE'][0]:
        return 'SAFE'
    elif score >= SCORE_RANGES['MODERATE'][0]:
        return 'MODERATE'
    else:
        return 'AGGRESSIVE'

print("  Phase 4D 가중치:")
for key, val in WEIGHTS_PHASE4D.items():
    print(f"    {key}: {val}")
print(f"  야간 배율: {NIGHT_MULTIPLIER}x")
print()

# ============================================================================
# Phase 4F 데이터에 Phase 4D 모델 적용
# ============================================================================

print("=" * 100)
print("분석 1: Phase 4D 모델로 Phase 4F 데이터 점수 계산")
print("=" * 100)
print()

results = []

for sample in phase4f_data['data']:
    score = calculate_driving_score_phase4d(sample)
    classification = classify_driver(score)

    result = {
        'score': score,
        'classification': classification,
        'actual_accident': sample['label'],
        'risk_group': sample['risk_group'],
        'source': sample['source'],
        'features': sample['features']
    }
    results.append(result)

print(f"[완료] {len(results):,}개 샘플 점수 계산 완료")
print()

# ============================================================================
# 점수 분포 분석
# ============================================================================

print("=" * 100)
print("분석 2: 운전 점수 분포 분석")
print("=" * 100)
print()

scores = [r['score'] for r in results]

print(f"점수 통계:")
print(f"  평균: {sum(scores)/len(scores):.2f}점")
print(f"  최소: {min(scores):.2f}점")
print(f"  최대: {max(scores):.2f}점")
print(f"  중앙값: {sorted(scores)[len(scores)//2]:.2f}점")
print()

# 분류별 분포
classification_counts = Counter([r['classification'] for r in results])

print(f"분류별 분포:")
for cls in ['SAFE', 'MODERATE', 'AGGRESSIVE']:
    count = classification_counts[cls]
    pct = count / len(results) * 100
    print(f"  {cls:12s}: {count:5,}개 ({pct:5.1f}%)")
print()

# ============================================================================
# 핵심 분석: 점수 그룹별 실제 사고율
# ============================================================================

print("=" * 100)
print("분석 3: 점수 그룹별 실제 사고율 (핵심)")
print("=" * 100)
print()

# 그룹별 집계
group_stats = {
    'SAFE': {'total': 0, 'accidents': 0},
    'MODERATE': {'total': 0, 'accidents': 0},
    'AGGRESSIVE': {'total': 0, 'accidents': 0}
}

for r in results:
    cls = r['classification']
    group_stats[cls]['total'] += 1
    if r['actual_accident'] == 1:
        group_stats[cls]['accidents'] += 1

print("점수 그룹별 실제 사고 발생률:")
print()
print("| 그룹 | 점수 범위 | 샘플 수 | 사고 발생 | 사고율 |")
print("|------|-----------|---------|-----------|--------|")

accident_rates = {}
for cls in ['SAFE', 'MODERATE', 'AGGRESSIVE']:
    total = group_stats[cls]['total']
    accidents = group_stats[cls]['accidents']
    rate = (accidents / total * 100) if total > 0 else 0
    accident_rates[cls] = rate

    score_range = f"{SCORE_RANGES[cls][0]}-{SCORE_RANGES[cls][1]}"
    print(f"| {cls:12s} | {score_range:9s} | {total:7,}개 | {accidents:9,}개 | **{rate:5.1f}%** |")

print()

# 사고율 비율 분석
if accident_rates['SAFE'] > 0:
    moderate_ratio = accident_rates['MODERATE'] / accident_rates['SAFE']
    aggressive_ratio = accident_rates['AGGRESSIVE'] / accident_rates['SAFE']

    print(f"사고율 배수 (SAFE 기준):")
    print(f"  MODERATE / SAFE = {moderate_ratio:.2f}배")
    print(f"  AGGRESSIVE / SAFE = {aggressive_ratio:.2f}배")
    print()

# ============================================================================
# Risk Group vs 점수 분류 비교
# ============================================================================

print("=" * 100)
print("분석 4: Risk Group (Phase 4F 라벨) vs 점수 분류 (Phase 4D 모델)")
print("=" * 100)
print()

# Risk Group 기준 사고율
risk_stats = {
    0: {'total': 0, 'accidents': 0},  # Safe group
    1: {'total': 0, 'accidents': 0}   # Risk group
}

for r in results:
    rg = r['risk_group']
    risk_stats[rg]['total'] += 1
    if r['actual_accident'] == 1:
        risk_stats[rg]['accidents'] += 1

risk_rate = (risk_stats[1]['accidents'] / risk_stats[1]['total'] * 100) if risk_stats[1]['total'] > 0 else 0
safe_rate = (risk_stats[0]['accidents'] / risk_stats[0]['total'] * 100) if risk_stats[0]['total'] > 0 else 0

print("Phase 4F Risk Group (실제 라벨) 기준:")
print(f"  Risk Group 사고율: {risk_rate:.1f}%")
print(f"  Safe Group 사고율: {safe_rate:.1f}%")
print(f"  비율: {risk_rate/safe_rate if safe_rate > 0 else 0:.2f}:1")
print()

print("Phase 4D 점수 분류 기준:")
print(f"  AGGRESSIVE 사고율: {accident_rates['AGGRESSIVE']:.1f}%")
print(f"  MODERATE 사고율: {accident_rates['MODERATE']:.1f}%")
print(f"  SAFE 사고율: {accident_rates['SAFE']:.1f}%")
print()

# 교차 분석: Risk Group과 점수 분류의 일치도
print("교차 분석: Risk Group vs 점수 분류")
print()

cross_analysis = defaultdict(lambda: defaultdict(int))

for r in results:
    rg_label = 'Risk' if r['risk_group'] == 1 else 'Safe'
    score_label = r['classification']
    cross_analysis[rg_label][score_label] += 1

print("| Risk Group \\ 점수 분류 | SAFE | MODERATE | AGGRESSIVE |")
print("|-------------------------|------|----------|------------|")

for rg in ['Safe', 'Risk']:
    safe_cnt = cross_analysis[rg]['SAFE']
    mod_cnt = cross_analysis[rg]['MODERATE']
    agg_cnt = cross_analysis[rg]['AGGRESSIVE']
    total = safe_cnt + mod_cnt + agg_cnt

    print(f"| {rg:23s} | {safe_cnt:4,} ({safe_cnt/total*100:4.1f}%) | "
          f"{mod_cnt:4,} ({mod_cnt/total*100:4.1f}%) | "
          f"{agg_cnt:4,} ({agg_cnt/total*100:4.1f}%) |")

print()

# ============================================================================
# 점수 구간별 세밀한 분석
# ============================================================================

print("=" * 100)
print("분석 5: 점수 구간별 세밀한 사고율 분석")
print("=" * 100)
print()

# 10점 단위 구간
score_bins = list(range(0, 101, 10))
bin_stats = defaultdict(lambda: {'total': 0, 'accidents': 0})

for r in results:
    score = r['score']
    # 어느 구간에 속하는지 찾기
    bin_idx = min(int(score // 10) * 10, 90)  # 90-100은 90 bin에
    bin_stats[bin_idx]['total'] += 1
    if r['actual_accident'] == 1:
        bin_stats[bin_idx]['accidents'] += 1

print("10점 구간별 사고율:")
print()
print("| 점수 구간 | 샘플 수 | 사고 발생 | 사고율 |")
print("|-----------|---------|-----------|--------|")

for bin_start in sorted(bin_stats.keys()):
    bin_end = bin_start + 9 if bin_start < 100 else 100
    total = bin_stats[bin_start]['total']
    accidents = bin_stats[bin_start]['accidents']
    rate = (accidents / total * 100) if total > 0 else 0

    print(f"| {bin_start:3d}-{bin_end:3d}점 | {total:7,}개 | {accidents:9,}개 | {rate:6.1f}% |")

print()

# ============================================================================
# 합성 vs 실제 데이터 성능 비교
# ============================================================================

print("=" * 100)
print("분석 6: 합성(Phase 4D) vs 실제(Phase 4F) 데이터 성능 비교")
print("=" * 100)
print()

print("Phase 4D (합성 데이터):")
print(f"  학습 데이터: 합성 생성 데이터")
print(f"  라벨 정확도: 100% (설계상)")
print(f"  Precision: {phase4d_results['phase4d_best']['metrics']['precision']:.3f}")
print(f"  Recall: {phase4d_results['phase4d_best']['metrics']['recall']:.3f}")
print(f"  F1-Score: {phase4d_results['phase4d_best']['metrics']['f1']:.3f}")
print()

print("Phase 4F (실제 데이터):")
print(f"  학습 데이터: Kaggle US Accidents")
print(f"  라벨 정확도: 85-90% (추정)")
print(f"  실제 사고율 (전체): {sum(r['actual_accident'] for r in results) / len(results) * 100:.1f}%")
print()

print("Phase 4D 모델을 Phase 4F 데이터에 적용한 결과:")
print(f"  SAFE 그룹 사고율: {accident_rates['SAFE']:.1f}%")
print(f"  AGGRESSIVE 그룹 사고율: {accident_rates['AGGRESSIVE']:.1f}%")
if accident_rates['SAFE'] > 0:
    print(f"  위험도 차이: {accident_rates['AGGRESSIVE'] / accident_rates['SAFE']:.1f}배")
print()

# 성능 격차 분석
print("성능 격차 분석:")
print()

# Phase 4D는 Precision 94%, Recall 90% 달성
# Phase 4F 실제 데이터에서는?
# AGGRESSIVE로 분류된 사람 중 실제 사고율로 Precision 추정
# 실제 사고 중 AGGRESSIVE로 분류된 비율로 Recall 추정

total_accidents = sum(1 for r in results if r['actual_accident'] == 1)
aggressive_samples = [r for r in results if r['classification'] == 'AGGRESSIVE']
aggressive_accidents = sum(1 for r in aggressive_samples if r['actual_accident'] == 1)

if len(aggressive_samples) > 0:
    pseudo_precision = aggressive_accidents / len(aggressive_samples)
    print(f"  Pseudo-Precision (AGGRESSIVE 중 실제 사고): {pseudo_precision*100:.1f}%")
    print(f"    vs Phase 4D: {phase4d_results['phase4d_best']['metrics']['precision']*100:.1f}%")
    print(f"    차이: {(phase4d_results['phase4d_best']['metrics']['precision'] - pseudo_precision)*100:.1f}%p")

if total_accidents > 0:
    pseudo_recall = aggressive_accidents / total_accidents
    print(f"  Pseudo-Recall (사고 중 AGGRESSIVE 분류): {pseudo_recall*100:.1f}%")
    print(f"    vs Phase 4D: {phase4d_results['phase4d_best']['metrics']['recall']*100:.1f}%")
    print(f"    차이: {(phase4d_results['phase4d_best']['metrics']['recall'] - pseudo_recall)*100:.1f}%p")

print()

# ============================================================================
# 결과 저장
# ============================================================================

print("=" * 100)
print("결과 저장")
print("=" * 100)
print()

analysis_results = {
    "metadata": {
        "date": datetime.now().isoformat(),
        "phase4d_model": phase4d_results['phase4d_best']['method'],
        "phase4f_samples": len(results),
        "analysis_type": "Cross-validation: Phase 4D model on Phase 4F data"
    },
    "score_distribution": {
        "mean": sum(scores)/len(scores),
        "min": min(scores),
        "max": max(scores),
        "median": sorted(scores)[len(scores)//2]
    },
    "classification_distribution": {
        cls: {"count": classification_counts[cls],
              "percentage": classification_counts[cls]/len(results)*100}
        for cls in ['SAFE', 'MODERATE', 'AGGRESSIVE']
    },
    "accident_rates_by_classification": {
        cls: {
            "total": group_stats[cls]['total'],
            "accidents": group_stats[cls]['accidents'],
            "rate": (group_stats[cls]['accidents'] / group_stats[cls]['total'] * 100) if group_stats[cls]['total'] > 0 else 0
        }
        for cls in ['SAFE', 'MODERATE', 'AGGRESSIVE']
    },
    "risk_group_comparison": {
        "risk_group_accident_rate": risk_rate,
        "safe_group_accident_rate": safe_rate,
        "ratio": risk_rate/safe_rate if safe_rate > 0 else 0
    },
    "cross_analysis": dict(cross_analysis),
    "detailed_scores": results[:100]  # 처음 100개만 저장
}

output_file = "phase4d_4f_cross_validation_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(analysis_results, f, indent=2, ensure_ascii=False)

print(f"  [완료] 결과 저장: {output_file}")
print()

print("=" * 100)
print("[완료] Phase 4D vs 4F 교차 검증 분석 완료")
print("=" * 100)
print()
print("다음 단계: python phase4d_4f_cross_validation_report.py")
