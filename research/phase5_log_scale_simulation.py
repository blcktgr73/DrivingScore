"""
Phase 5: User-Friendly Log-Scale Scoring Simulation

목적:
- Phase 4-C 검증된 가중치 유지하면서 사용자 친화적 점수 변환
- Log-scale 적용으로 단거리 trip 점수 급락 문제 해결
- 목표 등급 분포 달성: SAFE 65%, MODERATE 25%, AGGRESSIVE 10%

근거:
- Progressive Snapshot (2024): 텔레매틱스 참여자의 80%가 안전 운전자
- Pareto 원칙 (80/20): 사고의 80%는 고위험 운전자 20%에서 발생
- 보수적 조정: SAFE 65% (Progressive 80% 대비 엄격)
"""

import json
import math
import numpy as np
from datetime import datetime


def load_phase4c_data():
    """Phase 4-C enhanced report 데이터 로드"""
    with open('phase4c_enhanced_report.json', 'r') as f:
        data = json.load(f)
    return data


def calculate_linear_score(events, weights, time_of_day='day'):
    """
    Phase 4-C Linear 방식 점수 계산

    Args:
        events: dict with keys 'rapid_accel', 'sudden_stop', 'sharp_turn', 'over_speed'
        weights: Phase 4-C validated weights (0.0x scale)
        time_of_day: 'day' or 'night'

    Returns:
        float: 100점 기준 점수
    """
    penalty = 0
    for event_type, count in events.items():
        if event_type in weights:
            # Phase 4-C 가중치를 100배로 스케일링 (0.0588 -> 5.88점)
            weight = weights[event_type] * 100
            # 야간 가중치 1.5배 적용
            if time_of_day == 'night':
                weight *= 1.5
            penalty += count * weight

    return max(0, 100 - penalty)


def calculate_log_scale_score(events, weights, time_of_day='day', k=12.0, min_score=30):
    """
    Phase 5 Log-scale 방식 점수 계산

    Args:
        events: dict with keys 'rapid_accel', 'sudden_stop', 'sharp_turn', 'over_speed'
        weights: Phase 4-C validated weights (0.0x scale)
        time_of_day: 'day' or 'night'
        k: log-scale 조정 상수 (클수록 엄격)
        min_score: 최저 점수 하한선

    Returns:
        float: 100점 기준 점수
    """
    # Step 1: Phase 4-C 방식으로 가중 합계 계산
    weighted_sum = 0
    for event_type, count in events.items():
        if event_type in weights:
            # Phase 4-C 가중치를 100배로 스케일링
            weight = weights[event_type] * 100
            # 야간 가중치 1.5배 적용 (Phase 4-C와 동일)
            if time_of_day == 'night':
                weight *= 1.5
            weighted_sum += count * weight

    # Step 2: Log-scale 변환
    penalty = k * math.log(1 + weighted_sum)

    # Step 3: 최종 점수 (하한선 적용)
    score = max(min_score, 100 - penalty)

    return score


def simulate_trip_data(phase4c_data, n_trips=15000):
    """
    Phase 4-C 통계를 기반으로 trip 데이터 시뮬레이션

    Args:
        phase4c_data: Phase 4-C enhanced report
        n_trips: 생성할 trip 수

    Returns:
        list of dicts: 각 trip의 이벤트 데이터
    """
    # Scenario A 가중치 및 상관관계 사용
    weights = phase4c_data['scenario_a']['weights']
    correlations = phase4c_data['scenario_a']['correlations']
    accident_rate = phase4c_data['data_summary']['accident_rate_pct'] / 100

    trips = []
    np.random.seed(42)

    for i in range(n_trips):
        # 야간 운전 확률 (Phase 1 결과: 야간 시 사고 19.6% 증가)
        is_night = np.random.random() < 0.3  # 야간 30%
        time_of_day = 'night' if is_night else 'day'

        # 사고 여부 (야간 시 확률 증가)
        night_multiplier = 1.196 if is_night else 1.0
        has_accident = np.random.random() < (accident_rate * night_multiplier)

        # 이벤트 발생 (사고 시 더 많은 이벤트)
        # 상관관계를 고려한 평균 이벤트 수
        base_events = 5 if has_accident else 2

        # Poisson 분포로 이벤트 생성
        events = {
            'rapid_accel': max(0, int(np.random.poisson(base_events * correlations['rapid_accel']))),
            'sudden_stop': max(0, int(np.random.poisson(base_events * correlations['sudden_stop']))),
            'sharp_turn': max(0, int(np.random.poisson(base_events * correlations['sharp_turn']))),
            'over_speed': max(0, int(np.random.poisson(base_events * correlations['over_speed'])))
        }

        trips.append({
            'trip_id': i,
            'events': events,
            'time_of_day': time_of_day,
            'has_accident': has_accident
        })

    return trips


def classify_grade_linear(score):
    """Phase 4-C Linear 방식 등급 분류"""
    if score >= 77:
        return 'SAFE'
    elif score >= 72:
        return 'MODERATE'
    else:
        return 'AGGRESSIVE'


def find_optimal_cutoffs(scores, labels, target_distribution):
    """
    목표 분포에 맞는 최적 컷오프 찾기

    Args:
        scores: list of scores
        labels: list of accident labels (1/0)
        target_distribution: dict with keys 'SAFE', 'MODERATE', 'AGGRESSIVE'

    Returns:
        dict: {'safe_cutoff': float, 'aggressive_cutoff': float}
    """
    scores_array = np.array(scores)

    # 백분위수 계산
    safe_percentile = (1 - target_distribution['SAFE']) * 100
    aggressive_percentile = target_distribution['AGGRESSIVE'] * 100

    safe_cutoff = np.percentile(scores_array, safe_percentile)
    aggressive_cutoff = np.percentile(scores_array, aggressive_percentile)

    return {
        'safe_cutoff': round(safe_cutoff, 2),
        'aggressive_cutoff': round(aggressive_cutoff, 2)
    }


def classify_grade_with_cutoffs(score, safe_cutoff, aggressive_cutoff):
    """컷오프 기반 등급 분류"""
    if score >= safe_cutoff:
        return 'SAFE'
    elif score <= aggressive_cutoff:
        return 'AGGRESSIVE'
    else:
        return 'MODERATE'


def calculate_metrics(trips, scores, cutoffs, method_name):
    """
    모델 성능 지표 계산

    Args:
        trips: trip 데이터
        scores: 계산된 점수 리스트
        cutoffs: 등급 컷오프
        method_name: 'linear' or 'log_scale'

    Returns:
        dict: 성능 지표
    """
    grades = []
    for score in scores:
        if method_name == 'linear':
            grade = classify_grade_linear(score)
        else:
            grade = classify_grade_with_cutoffs(score, cutoffs['safe_cutoff'], cutoffs['aggressive_cutoff'])
        grades.append(grade)

    # 등급별 사고율 계산
    grade_accident_rates = {}
    grade_counts = {}

    for grade in ['SAFE', 'MODERATE', 'AGGRESSIVE']:
        grade_mask = [g == grade for g in grades]
        grade_trips = [t for t, m in zip(trips, grade_mask) if m]

        if len(grade_trips) > 0:
            accident_count = sum(1 for t in grade_trips if t['has_accident'])
            grade_accident_rates[grade] = accident_count / len(grade_trips)
            grade_counts[grade] = len(grade_trips)
        else:
            grade_accident_rates[grade] = 0.0
            grade_counts[grade] = 0

    # 등급 분포
    total = len(grades)
    grade_distribution = {
        grade: count / total for grade, count in grade_counts.items()
    }

    # AUC 계산 (간이 버전)
    # 점수를 사고 예측 확률로 사용
    labels = [1 if t['has_accident'] else 0 for t in trips]
    normalized_scores = [(100 - s) / 100 for s in scores]  # 낮을수록 위험

    # Simple AUC: 사고/비사고 점수 분리도
    accident_scores = [s for s, l in zip(normalized_scores, labels) if l == 1]
    no_accident_scores = [s for s, l in zip(normalized_scores, labels) if l == 0]

    if len(accident_scores) > 0 and len(no_accident_scores) > 0:
        # Mann-Whitney U test 기반 AUC 추정
        comparisons = 0
        correct = 0
        for acc_s in accident_scores[:500]:  # 샘플링으로 계산 속도 향상
            for no_acc_s in no_accident_scores[:500]:
                comparisons += 1
                if acc_s > no_acc_s:
                    correct += 1
                elif acc_s == no_acc_s:
                    correct += 0.5
        auc = correct / comparisons if comparisons > 0 else 0.5
    else:
        auc = 0.5

    return {
        'method': method_name,
        'cutoffs': cutoffs if method_name != 'linear' else {'safe_cutoff': 77, 'aggressive_cutoff': 71},
        'grade_distribution': grade_distribution,
        'grade_accident_rates': grade_accident_rates,
        'grade_counts': grade_counts,
        'auc': round(auc, 4),
        'score_stats': {
            'mean': round(np.mean(scores), 2),
            'std': round(np.std(scores), 2),
            'min': round(np.min(scores), 2),
            'max': round(np.max(scores), 2),
            'median': round(np.median(scores), 2)
        }
    }


def compare_scoring_methods(trips, weights):
    """Linear vs Log-scale 방식 비교"""
    print("\n" + "="*70)
    print("Phase 5: Log-Scale Scoring Simulation")
    print("="*70)

    # 1. Linear 방식 (Phase 4-C)
    print("\n[1/3] Calculating Linear scores (Phase 4-C)...")
    linear_scores = []
    for trip in trips:
        score = calculate_linear_score(trip['events'], weights, trip['time_of_day'])
        linear_scores.append(score)

    linear_cutoffs = {'safe_cutoff': 77, 'aggressive_cutoff': 71}
    linear_metrics = calculate_metrics(trips, linear_scores, linear_cutoffs, 'linear')

    print(f"  Linear Score Distribution:")
    print(f"    SAFE: {linear_metrics['grade_distribution']['SAFE']*100:.1f}%")
    print(f"    MODERATE: {linear_metrics['grade_distribution']['MODERATE']*100:.1f}%")
    print(f"    AGGRESSIVE: {linear_metrics['grade_distribution']['AGGRESSIVE']*100:.1f}%")

    # 2. Log-scale 방식 (Phase 5)
    print("\n[2/3] Calculating Log-scale scores (Phase 5)...")
    log_scores = []
    for trip in trips:
        score = calculate_log_scale_score(trip['events'], weights, trip['time_of_day'], k=12.0, min_score=30)
        log_scores.append(score)

    # 목표 분포에 맞는 컷오프 찾기
    target_distribution = {'SAFE': 0.65, 'MODERATE': 0.25, 'AGGRESSIVE': 0.10}
    log_cutoffs = find_optimal_cutoffs(log_scores, [t['has_accident'] for t in trips], target_distribution)
    log_metrics = calculate_metrics(trips, log_scores, log_cutoffs, 'log_scale')

    print(f"  Log-scale Score Distribution:")
    print(f"    SAFE: {log_metrics['grade_distribution']['SAFE']*100:.1f}%")
    print(f"    MODERATE: {log_metrics['grade_distribution']['MODERATE']*100:.1f}%")
    print(f"    AGGRESSIVE: {log_metrics['grade_distribution']['AGGRESSIVE']*100:.1f}%")

    # 3. 비교 분석
    print("\n[3/3] Analyzing differences...")

    return {
        'linear': linear_metrics,
        'log_scale': log_metrics,
        'comparison': {
            'distribution_change': {
                'SAFE': log_metrics['grade_distribution']['SAFE'] - linear_metrics['grade_distribution']['SAFE'],
                'MODERATE': log_metrics['grade_distribution']['MODERATE'] - linear_metrics['grade_distribution']['MODERATE'],
                'AGGRESSIVE': log_metrics['grade_distribution']['AGGRESSIVE'] - linear_metrics['grade_distribution']['AGGRESSIVE']
            },
            'auc_change': log_metrics['auc'] - linear_metrics['auc'],
            'score_range_change': {
                'mean': log_metrics['score_stats']['mean'] - linear_metrics['score_stats']['mean'],
                'std': log_metrics['score_stats']['std'] - linear_metrics['score_stats']['std']
            }
        }
    }


def main():
    print("Phase 5: User-Friendly Log-Scale Scoring System")
    print("=" * 70)
    print("\nGoals:")
    print("  1. Maintain Phase 4-C weights while improving UX")
    print("  2. Target distribution: SAFE 65%, MODERATE 25%, AGGRESSIVE 10%")
    print("  3. Maintain prediction performance (AUC ~= 0.67)")
    print("\nIndustry References:")
    print("  - Progressive Snapshot (2024): 80% safe drivers")
    print("  - Pareto Principle: 80% accidents from 20% high-risk drivers")
    print("  - Conservative adjustment: SAFE 65% (stricter than Progressive 80%)")

    # Phase 4-C 데이터 로드
    print("\n" + "="*70)
    print("Loading Phase 4-C Data...")
    print("="*70)
    phase4c_data = load_phase4c_data()

    print(f"\nPhase 4-C Statistics:")
    print(f"  Samples: {phase4c_data['data_summary']['matched_samples']:,}")
    print(f"  Accident Rate: {phase4c_data['data_summary']['accident_rate_pct']:.2f}%")
    print(f"  AUC (Scenario A): {phase4c_data['scenario_a']['metrics']['auc']:.4f}")

    # Scenario A 가중치 사용
    weights = phase4c_data['scenario_a']['weights']
    print(f"\nWeights (Scenario A):")
    for event, weight in weights.items():
        print(f"  {event}: {weight:.4f}")

    # Trip 데이터 시뮬레이션
    print("\n" + "="*70)
    print("Simulating Trip Data...")
    print("="*70)
    n_trips = 15000
    trips = simulate_trip_data(phase4c_data, n_trips)

    accident_count = sum(1 for t in trips if t['has_accident'])
    night_count = sum(1 for t in trips if t['time_of_day'] == 'night')

    print(f"\nGenerated {n_trips:,} trips:")
    print(f"  Accidents: {accident_count:,} ({accident_count/n_trips*100:.1f}%)")
    print(f"  Night trips: {night_count:,} ({night_count/n_trips*100:.1f}%)")

    # 점수 계산 및 비교
    results = compare_scoring_methods(trips, weights)

    # 결과 출력
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print("\n1. Grade Distribution Comparison:")
    print(f"{'Grade':<12} {'Linear (4-C)':<15} {'Log-scale (5)':<15} {'Change':<10}")
    print("-" * 60)
    for grade in ['SAFE', 'MODERATE', 'AGGRESSIVE']:
        linear_pct = results['linear']['grade_distribution'][grade] * 100
        log_pct = results['log_scale']['grade_distribution'][grade] * 100
        change = results['comparison']['distribution_change'][grade] * 100
        print(f"{grade:<12} {linear_pct:>6.1f}% ({results['linear']['grade_counts'][grade]:>5}) "
              f"{log_pct:>6.1f}% ({results['log_scale']['grade_counts'][grade]:>5}) "
              f"{change:>+6.1f}%p")

    print("\n2. Grade Cutoffs:")
    print(f"  Linear (Phase 4-C):")
    print(f"    SAFE: ≥77점, MODERATE: 72-76점, AGGRESSIVE: ≤71점")
    print(f"  Log-scale (Phase 5):")
    print(f"    SAFE: ≥{results['log_scale']['cutoffs']['safe_cutoff']:.1f}점")
    print(f"    AGGRESSIVE: ≤{results['log_scale']['cutoffs']['aggressive_cutoff']:.1f}점")

    print("\n3. Accident Rates by Grade:")
    print(f"{'Grade':<12} {'Linear (4-C)':<15} {'Log-scale (5)':<15}")
    print("-" * 45)
    for grade in ['SAFE', 'MODERATE', 'AGGRESSIVE']:
        linear_rate = results['linear']['grade_accident_rates'][grade] * 100
        log_rate = results['log_scale']['grade_accident_rates'][grade] * 100
        print(f"{grade:<12} {linear_rate:>6.1f}% {log_rate:>15.1f}%")

    print("\n4. Prediction Performance:")
    print(f"  Linear AUC: {results['linear']['auc']:.4f}")
    print(f"  Log-scale AUC: {results['log_scale']['auc']:.4f}")
    print(f"  Change: {results['comparison']['auc_change']:+.4f}")

    print("\n5. Score Statistics:")
    print(f"{'Metric':<12} {'Linear (4-C)':<15} {'Log-scale (5)':<15} {'Change':<10}")
    print("-" * 60)
    for metric in ['mean', 'std', 'min', 'max', 'median']:
        linear_val = results['linear']['score_stats'][metric]
        log_val = results['log_scale']['score_stats'][metric]
        change = log_val - linear_val if metric != 'std' else log_val - linear_val
        print(f"{metric.capitalize():<12} {linear_val:>6.2f} {log_val:>15.2f} {change:>+9.2f}")

    # JSON 저장
    results['metadata'] = {
        'execution_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_trips': n_trips,
        'log_scale_params': {'k': 12.0, 'min_score': 30},
        'target_distribution': {'SAFE': 0.65, 'MODERATE': 0.25, 'AGGRESSIVE': 0.10},
        'industry_reference': {
            'progressive_snapshot': 'About 20% experience rate increase (2024)',
            'pareto_principle': '80% of accidents from 20% high-risk drivers'
        }
    }

    output_file = 'phase5_log_scale_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to: {output_file}")
    print("\n" + "="*70)
    print("Simulation Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
