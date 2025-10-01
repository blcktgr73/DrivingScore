"""
Phase 4-C vs Phase 5 공정 비교

목적:
- Phase 4-C Linear 방식의 컷오프를 재조정하여 목표 분포 (65/25/10) 달성
- Phase 5 Log-scale과 동일한 기준으로 비교
- Linear vs Log-scale의 순수한 차이점 분석

근거:
- 보험 업계 표준: SAFE 65%, MODERATE 25%, AGGRESSIVE 10%
- 공정한 비교를 위해 동일한 등급 분포 목표 적용
"""

import json
import math
import numpy as np
from datetime import datetime


def load_phase5_results():
    """Phase 5 결과 로드"""
    with open('phase5_log_scale_results.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def load_phase4c_data():
    """Phase 4-C enhanced report 데이터 로드"""
    with open('phase4c_enhanced_report.json', 'r') as f:
        return json.load(f)


def calculate_linear_score(events, weights, time_of_day='day'):
    """Phase 4-C Linear 방식 점수 계산"""
    penalty = 0
    for event_type, count in events.items():
        if event_type in weights:
            weight = weights[event_type] * 100  # Scale to 100-point system
            if time_of_day == 'night':
                weight *= 1.5
            penalty += count * weight
    return max(0, 100 - penalty)


def simulate_trip_data(phase4c_data, n_trips=15000):
    """Phase 4-C 통계 기반 trip 시뮬레이션"""
    weights = phase4c_data['scenario_a']['weights']
    correlations = phase4c_data['scenario_a']['correlations']
    accident_rate = phase4c_data['data_summary']['accident_rate_pct'] / 100

    trips = []
    np.random.seed(42)

    for i in range(n_trips):
        is_night = np.random.random() < 0.3
        time_of_day = 'night' if is_night else 'day'
        night_multiplier = 1.196 if is_night else 1.0
        has_accident = np.random.random() < (accident_rate * night_multiplier)

        base_events = 5 if has_accident else 2

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


def find_optimal_cutoffs_for_target_distribution(scores, target_distribution):
    """목표 분포에 맞는 컷오프 계산"""
    scores_array = np.array(scores)

    safe_percentile = (1 - target_distribution['SAFE']) * 100
    aggressive_percentile = target_distribution['AGGRESSIVE'] * 100

    safe_cutoff = np.percentile(scores_array, safe_percentile)
    aggressive_cutoff = np.percentile(scores_array, aggressive_percentile)

    return {
        'safe_cutoff': round(safe_cutoff, 2),
        'aggressive_cutoff': round(aggressive_cutoff, 2)
    }


def classify_grade(score, safe_cutoff, aggressive_cutoff):
    """컷오프 기반 등급 분류"""
    if score >= safe_cutoff:
        return 'SAFE'
    elif score <= aggressive_cutoff:
        return 'AGGRESSIVE'
    else:
        return 'MODERATE'


def calculate_metrics(trips, scores, cutoffs, method_name):
    """성능 지표 계산"""
    grades = [classify_grade(s, cutoffs['safe_cutoff'], cutoffs['aggressive_cutoff']) for s in scores]

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

    total = len(grades)
    grade_distribution = {grade: count / total for grade, count in grade_counts.items()}

    # AUC 계산
    labels = [1 if t['has_accident'] else 0 for t in trips]
    normalized_scores = [(100 - s) / 100 for s in scores]

    accident_scores = [s for s, l in zip(normalized_scores, labels) if l == 1]
    no_accident_scores = [s for s, l in zip(normalized_scores, labels) if l == 0]

    if len(accident_scores) > 0 and len(no_accident_scores) > 0:
        comparisons = 0
        correct = 0
        for acc_s in accident_scores[:500]:
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
        'cutoffs': cutoffs,
        'grade_distribution': grade_distribution,
        'grade_accident_rates': grade_accident_rates,
        'grade_counts': grade_counts,
        'auc': round(auc, 4),
        'score_stats': {
            'mean': round(np.mean(scores), 2),
            'std': round(np.std(scores), 2),
            'min': round(np.min(scores), 2),
            'max': round(np.max(scores), 2),
            'median': round(np.median(scores), 2),
            'q25': round(np.percentile(scores, 25), 2),
            'q75': round(np.percentile(scores, 75), 2)
        }
    }


def main():
    print("="*70)
    print("Phase 4-C vs Phase 5: Fair Comparison with Adjusted Cutoffs")
    print("="*70)
    print("\nObjective:")
    print("  - Adjust Phase 4-C cutoffs to match target distribution (65/25/10)")
    print("  - Enable fair comparison: Linear vs Log-scale")
    print("  - Analyze pure scoring method differences")

    # Load data
    print("\n" + "="*70)
    print("Loading Data...")
    print("="*70)

    phase4c_data = load_phase4c_data()
    weights = phase4c_data['scenario_a']['weights']

    print(f"\nPhase 4-C Weights:")
    for event, weight in weights.items():
        print(f"  {event}: {weight:.4f}")

    # Simulate trips
    print("\n" + "="*70)
    print("Simulating 15,000 Trips...")
    print("="*70)

    trips = simulate_trip_data(phase4c_data, 15000)
    accident_count = sum(1 for t in trips if t['has_accident'])
    night_count = sum(1 for t in trips if t['time_of_day'] == 'night')

    print(f"\nGenerated trips:")
    print(f"  Total: 15,000")
    print(f"  Accidents: {accident_count:,} ({accident_count/15000*100:.1f}%)")
    print(f"  Night trips: {night_count:,} ({night_count/15000*100:.1f}%)")

    # Calculate Linear scores
    print("\n" + "="*70)
    print("Calculating Phase 4-C Linear Scores...")
    print("="*70)

    linear_scores = [calculate_linear_score(t['events'], weights, t['time_of_day']) for t in trips]

    # Target distribution
    target_distribution = {'SAFE': 0.65, 'MODERATE': 0.25, 'AGGRESSIVE': 0.10}

    # Find new cutoffs for Linear
    print("\nFinding optimal cutoffs for target distribution (65/25/10)...")
    linear_cutoffs_adjusted = find_optimal_cutoffs_for_target_distribution(linear_scores, target_distribution)

    print(f"\nAdjusted Linear Cutoffs:")
    print(f"  SAFE: >={linear_cutoffs_adjusted['safe_cutoff']:.2f}")
    print(f"  AGGRESSIVE: <={linear_cutoffs_adjusted['aggressive_cutoff']:.2f}")

    # Calculate metrics with adjusted cutoffs
    linear_metrics_adjusted = calculate_metrics(trips, linear_scores, linear_cutoffs_adjusted, 'linear_adjusted')

    print(f"\nAdjusted Distribution:")
    print(f"  SAFE: {linear_metrics_adjusted['grade_distribution']['SAFE']*100:.1f}%")
    print(f"  MODERATE: {linear_metrics_adjusted['grade_distribution']['MODERATE']*100:.1f}%")
    print(f"  AGGRESSIVE: {linear_metrics_adjusted['grade_distribution']['AGGRESSIVE']*100:.1f}%")

    # Load Phase 5 results for comparison
    print("\n" + "="*70)
    print("Loading Phase 5 Results...")
    print("="*70)

    phase5_results = load_phase5_results()

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    print("\n1. Grade Distribution (Both Targeting 65/25/10):")
    print(f"{'Grade':<12} {'Linear (4-C)':<18} {'Log-scale (5)':<18} {'Difference':<10}")
    print("-" * 70)
    for grade in ['SAFE', 'MODERATE', 'AGGRESSIVE']:
        linear_pct = linear_metrics_adjusted['grade_distribution'][grade] * 100
        log_pct = phase5_results['log_scale']['grade_distribution'][grade] * 100
        diff = log_pct - linear_pct

        linear_count = linear_metrics_adjusted['grade_counts'][grade]
        log_count = phase5_results['log_scale']['grade_counts'][grade]

        print(f"{grade:<12} {linear_pct:>6.1f}% ({linear_count:>5}) "
              f"{log_pct:>6.1f}% ({log_count:>5}) "
              f"{diff:>+6.1f}%p")

    print("\n2. Cutoff Comparison:")
    print(f"{'Method':<15} {'SAFE Cutoff':<15} {'AGGRESSIVE Cutoff':<20}")
    print("-" * 55)
    print(f"{'Linear (4-C)':<15} >={linear_cutoffs_adjusted['safe_cutoff']:<14.1f} "
          f"<={linear_cutoffs_adjusted['aggressive_cutoff']:<18.1f}")
    print(f"{'Log-scale (5)':<15} >={phase5_results['log_scale']['cutoffs']['safe_cutoff']:<14.1f} "
          f"<={phase5_results['log_scale']['cutoffs']['aggressive_cutoff']:<18.1f}")

    cutoff_diff_safe = phase5_results['log_scale']['cutoffs']['safe_cutoff'] - linear_cutoffs_adjusted['safe_cutoff']
    cutoff_diff_agg = phase5_results['log_scale']['cutoffs']['aggressive_cutoff'] - linear_cutoffs_adjusted['aggressive_cutoff']

    print(f"{'Difference':<15} {cutoff_diff_safe:>+14.1f} {cutoff_diff_agg:>+18.1f}")

    print("\n3. Accident Rates by Grade:")
    print(f"{'Grade':<12} {'Linear (4-C)':<18} {'Log-scale (5)':<18} {'Improvement':<10}")
    print("-" * 65)
    for grade in ['SAFE', 'MODERATE', 'AGGRESSIVE']:
        linear_rate = linear_metrics_adjusted['grade_accident_rates'][grade] * 100
        log_rate = phase5_results['log_scale']['grade_accident_rates'][grade] * 100
        improvement = log_rate - linear_rate

        print(f"{grade:<12} {linear_rate:>6.1f}% {log_rate:>18.1f}% {improvement:>+9.1f}%p")

    print("\n4. Prediction Performance:")
    print(f"  Linear AUC (adjusted): {linear_metrics_adjusted['auc']:.4f}")
    print(f"  Log-scale AUC:         {phase5_results['log_scale']['auc']:.4f}")
    print(f"  Difference:            {phase5_results['log_scale']['auc'] - linear_metrics_adjusted['auc']:+.4f}")

    print("\n5. Score Distribution Statistics:")
    print(f"{'Metric':<12} {'Linear (4-C)':<18} {'Log-scale (5)':<18} {'Difference':<10}")
    print("-" * 65)
    for metric in ['mean', 'std', 'min', 'max', 'median']:
        linear_val = linear_metrics_adjusted['score_stats'][metric]
        log_val = phase5_results['log_scale']['score_stats'][metric]
        diff = log_val - linear_val

        metric_name = metric.upper() if len(metric) <= 3 else metric.capitalize()
        print(f"{metric_name:<12} {linear_val:>6.2f} {log_val:>18.2f} {diff:>+9.2f}")

    # Add quartiles separately
    print(f"{'Q25 (25%)':<12} {linear_metrics_adjusted['score_stats']['q25']:>6.2f} "
          f"{'N/A':>18} {'N/A':>9}")
    print(f"{'Q75 (75%)':<12} {linear_metrics_adjusted['score_stats']['q75']:>6.2f} "
          f"{'N/A':>18} {'N/A':>9}")

    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    safe_accident_improvement = (phase5_results['log_scale']['grade_accident_rates']['SAFE'] -
                                 linear_metrics_adjusted['grade_accident_rates']['SAFE']) * 100

    print(f"\n1. SAFE Grade Quality:")
    print(f"   Linear:    {linear_metrics_adjusted['grade_accident_rates']['SAFE']*100:.1f}% accident rate")
    print(f"   Log-scale: {phase5_results['log_scale']['grade_accident_rates']['SAFE']*100:.1f}% accident rate")
    print(f"   Result:    {abs(safe_accident_improvement):.1f}%p {'improvement' if safe_accident_improvement < 0 else 'degradation'}")

    print(f"\n2. Cutoff Stringency:")
    print(f"   Linear SAFE cutoff:    {linear_cutoffs_adjusted['safe_cutoff']:.1f} points")
    print(f"   Log-scale SAFE cutoff: {phase5_results['log_scale']['cutoffs']['safe_cutoff']:.1f} points")
    if cutoff_diff_safe < 0:
        print(f"   Result: Log-scale is MORE stringent ({abs(cutoff_diff_safe):.1f} points higher)")
    else:
        print(f"   Result: Log-scale is LESS stringent ({cutoff_diff_safe:.1f} points lower)")

    print(f"\n3. Score Range:")
    linear_range = linear_metrics_adjusted['score_stats']['max'] - linear_metrics_adjusted['score_stats']['min']
    log_range = phase5_results['log_scale']['score_stats']['max'] - phase5_results['log_scale']['score_stats']['min']
    print(f"   Linear:    {linear_range:.1f} points ({linear_metrics_adjusted['score_stats']['min']:.1f} - {linear_metrics_adjusted['score_stats']['max']:.1f})")
    print(f"   Log-scale: {log_range:.1f} points ({phase5_results['log_scale']['score_stats']['min']:.1f} - {phase5_results['log_scale']['score_stats']['max']:.1f})")
    print(f"   Result: Log-scale has {'wider' if log_range > linear_range else 'narrower'} range ({abs(log_range - linear_range):.1f} points)")

    # Save results
    results = {
        'execution_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'target_distribution': target_distribution,
        'phase4c_linear_adjusted': {
            'cutoffs': linear_cutoffs_adjusted,
            'distribution': linear_metrics_adjusted['grade_distribution'],
            'accident_rates': linear_metrics_adjusted['grade_accident_rates'],
            'grade_counts': linear_metrics_adjusted['grade_counts'],
            'auc': linear_metrics_adjusted['auc'],
            'score_stats': linear_metrics_adjusted['score_stats']
        },
        'phase5_log_scale': {
            'cutoffs': phase5_results['log_scale']['cutoffs'],
            'distribution': phase5_results['log_scale']['grade_distribution'],
            'accident_rates': phase5_results['log_scale']['grade_accident_rates'],
            'grade_counts': phase5_results['log_scale']['grade_counts'],
            'auc': phase5_results['log_scale']['auc'],
            'score_stats': phase5_results['log_scale']['score_stats']
        },
        'comparison': {
            'safe_accident_rate_improvement': safe_accident_improvement,
            'cutoff_difference': {
                'safe': cutoff_diff_safe,
                'aggressive': cutoff_diff_agg
            },
            'score_range_difference': log_range - linear_range
        }
    }

    output_file = 'phase4c_phase5_fair_comparison.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to: {output_file}")
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
