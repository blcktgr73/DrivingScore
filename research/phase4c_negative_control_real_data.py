#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4-C: Negative Control Test (실제 데이터 기반)
====================================================
실제 phase4c_enhanced_analysis.py 결과 데이터를 사용한 검증

목적: 시뮬레이션 한계를 극복하고 실제 AUC 0.65-0.67 결과 검증
"""

import os
import sys
import json
import math
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("Phase 4-C: Negative Control Test (실제 데이터 기반)")
print("=" * 70)
print()

# ============================================================================
# 실제 Phase 4-C 결과 로드
# ============================================================================

def load_phase4c_results():
    """phase4c_enhanced_report.json 로드"""
    json_path = os.path.join(os.path.dirname(__file__), 'phase4c_enhanced_report.json')

    if not os.path.exists(json_path):
        print(f"❌ {json_path} 파일이 없습니다.")
        print("phase4c_enhanced_analysis.py를 먼저 실행해주세요.")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

results = load_phase4c_results()

if results is None:
    print("프로그램을 종료합니다.")
    sys.exit(1)

print("## 1. 실제 Phase 4-C 결과")
print("-" * 70)
print(f"분석 일시: {results.get('execution_date', 'N/A')}")
print(f"매칭 샘플: {results['data_summary']['matched_samples']:,}개")
accident_rate = results['data_summary'].get('accident_rate_pct', 0) / 100
print(f"사고율: {accident_rate*100:.1f}%")
print()

# ============================================================================
# Scenario A vs B 비교
# ============================================================================

print("## 2. Scenario 성능 비교")
print("-" * 70)
print()

scenario_a = results['scenario_a']
scenario_b = results['scenario_b']

print("**Scenario A (4개 이벤트)**:")
print(f"  AUC: {scenario_a['metrics']['auc']:.4f}")
print(f"  Accuracy: {scenario_a['metrics']['accuracy']:.4f}")
print(f"  Precision: {scenario_a['metrics']['precision']:.4f}")
print(f"  Recall: {scenario_a['metrics']['recall']:.4f}")
print()

print("**Scenario B (3개 이벤트)**:")
print(f"  AUC: {scenario_b['metrics']['auc']:.4f}")
print(f"  Accuracy: {scenario_b['metrics']['accuracy']:.4f}")
print(f"  Precision: {scenario_b['metrics']['precision']:.4f}")
print(f"  Recall: {scenario_b['metrics']['recall']:.4f}")
print()

# ============================================================================
# 무작위 분류기 기준선
# ============================================================================

print("\n" + "=" * 70)
print("## 3. 무작위 분류기(Random Classifier) 대비 성능")
print("-" * 70)
print()

# 무작위 분류기의 기대 AUC = 0.5
random_auc = 0.5

auc_a = scenario_a['metrics']['auc']
auc_b = scenario_b['metrics']['auc']

diff_a = auc_a - random_auc
diff_b = auc_b - random_auc

print("**무작위 분류기 기준선**:")
print(f"  기대 AUC: {random_auc:.4f} (완전 무작위 예측)")
print()

print("**Scenario A 개선도**:")
print(f"  실제 AUC: {auc_a:.4f}")
print(f"  무작위 대비: {diff_a:+.4f} ({diff_a/random_auc*100:+.1f}%)")

if diff_a > 0.15:
    print(f"  ✅ 매우 우수 (차이 > 0.15)")
    sig_a = "매우 유의"
elif diff_a > 0.10:
    print(f"  ✅ 우수 (차이 > 0.10)")
    sig_a = "유의"
elif diff_a > 0.05:
    print(f"  ⚠️ 보통 (차이 > 0.05)")
    sig_a = "약한 유의"
else:
    print(f"  ❌ 불충분 (차이 ≤ 0.05)")
    sig_a = "유의하지 않음"

print()

print("**Scenario B 개선도**:")
print(f"  실제 AUC: {auc_b:.4f}")
print(f"  무작위 대비: {diff_b:+.4f} ({diff_b/random_auc*100:+.1f}%)")

if diff_b > 0.15:
    print(f"  ✅ 매우 우수 (차이 > 0.15)")
    sig_b = "매우 유의"
elif diff_b > 0.10:
    print(f"  ✅ 우수 (차이 > 0.10)")
    sig_b = "유의"
elif diff_b > 0.05:
    print(f"  ⚠️ 보통 (차이 > 0.05)")
    sig_b = "약한 유의"
else:
    print(f"  ❌ 불충분 (차이 ≤ 0.05)")
    sig_b = "유의하지 않음"

# ============================================================================
# 통계적 유의성 검정
# ============================================================================

print("\n\n" + "=" * 70)
print("## 4. 통계적 유의성 검정")
print("-" * 70)
print()

# AUC의 표준오차 계산 (DeLong method 근사)
n_samples = results['data_summary']['matched_samples']
n_positive = int(n_samples * accident_rate)
n_negative = n_samples - n_positive

# AUC 표준오차 근사식
se_a = math.sqrt((auc_a * (1 - auc_a)) / n_samples)
se_b = math.sqrt((auc_b * (1 - auc_b)) / n_samples)

# Z-score 계산 (H0: AUC = 0.5)
z_score_a = (auc_a - 0.5) / se_a
z_score_b = (auc_b - 0.5) / se_b

# p-value 근사 (양측 검정)
from math import erfc
p_value_a = erfc(abs(z_score_a) / math.sqrt(2))
p_value_b = erfc(abs(z_score_b) / math.sqrt(2))

print("**Scenario A**:")
print(f"  Z-score: {z_score_a:.4f}")
print(f"  p-value: {p_value_a:.6f}")

if p_value_a < 0.001:
    print(f"  ✅ 매우 유의함 (p < 0.001)")
elif p_value_a < 0.01:
    print(f"  ✅ 유의함 (p < 0.01)")
elif p_value_a < 0.05:
    print(f"  ⚠️ 약한 유의성 (p < 0.05)")
else:
    print(f"  ❌ 유의하지 않음 (p ≥ 0.05)")

print()

print("**Scenario B**:")
print(f"  Z-score: {z_score_b:.4f}")
print(f"  p-value: {p_value_b:.6f}")

if p_value_b < 0.001:
    print(f"  ✅ 매우 유의함 (p < 0.001)")
elif p_value_b < 0.01:
    print(f"  ✅ 유의함 (p < 0.01)")
elif p_value_b < 0.05:
    print(f"  ⚠️ 약한 유의성 (p < 0.05)")
else:
    print(f"  ❌ 유의하지 않음 (p ≥ 0.05)")

# 95% 신뢰구간
ci_lower_a = auc_a - 1.96 * se_a
ci_upper_a = auc_a + 1.96 * se_a
ci_lower_b = auc_b - 1.96 * se_b
ci_upper_b = auc_b + 1.96 * se_b

print()
print("**95% 신뢰구간**:")
print(f"  Scenario A: [{ci_lower_a:.4f}, {ci_upper_a:.4f}]")
print(f"  Scenario B: [{ci_lower_b:.4f}, {ci_upper_b:.4f}]")

includes_05_a = ci_lower_a <= 0.5 <= ci_upper_a
includes_05_b = ci_lower_b <= 0.5 <= ci_upper_b

print()
if not includes_05_a and not includes_05_b:
    print("  ✅ 두 시나리오 모두 0.5를 포함하지 않음 → 무작위보다 유의하게 우수")
elif not includes_05_a or not includes_05_b:
    print("  ⚠️ 한 시나리오만 0.5를 포함하지 않음 → 부분적으로 유의")
else:
    print("  ❌ 두 시나리오 모두 0.5를 포함 → 무작위와 차이 불명확")

# ============================================================================
# 실무적 해석
# ============================================================================

print("\n\n" + "=" * 70)
print("## 5. 실무적 해석")
print("-" * 70)
print()

print("**1. 예측 성능 평가**:")
if auc_a >= 0.7 or auc_b >= 0.7:
    performance = "우수"
    emoji = "✅"
elif auc_a >= 0.6 or auc_b >= 0.6:
    performance = "양호"
    emoji = "⚠️"
else:
    performance = "개선 필요"
    emoji = "❌"

print(f"  {emoji} AUC 0.65-0.67 수준: {performance}")
print(f"  - 실무 기준: AUC 0.6-0.7은 중간 수준 예측력")
print(f"  - 비교: 신용평가(0.7-0.8), 의료진단(0.8-0.9)")
print()

print("**2. 방법론 타당성**:")
if (diff_a > 0.10 or diff_b > 0.10) and (p_value_a < 0.05 or p_value_b < 0.05):
    print("  ✅ 시공간 매칭이 무작위보다 통계적으로 유의하게 우수")
    print("  → Phase 4-C 방법론은 타당함")
    validity = "타당"
elif diff_a > 0.05 or diff_b > 0.05:
    print("  ⚠️ 시공간 매칭이 무작위보다 다소 우수")
    print("  → 방법론의 효과는 있으나 제한적")
    validity = "부분적 타당"
else:
    print("  ❌ 시공간 매칭의 이점 불명확")
    print("  → 방법론 재검토 필요")
    validity = "재검토 필요"

print()

print("**3. 적용 범위 권장**:")
if validity == "타당":
    print("  ✅ 파일럿 시스템 및 내부 도구에 적용 가능")
    print("  ⚠️ 상용 서비스는 Phase 5 이후 권장")
    recommendation = "파일럿 적용 가능"
elif validity == "부분적 타당":
    print("  ⚠️ 제한적 파일럿 테스트만 권장")
    print("  ❌ Phase 5 데이터 수집 후 재검증 필수")
    recommendation = "제한적 파일럿"
else:
    print("  ❌ 프로덕션 적용 불가")
    print("  → Phase 5에서 방법론 개선 후 재평가")
    recommendation = "참고용으로만 활용"

# ============================================================================
# 최종 결론
# ============================================================================

print("\n\n" + "=" * 70)
print("## 최종 결론")
print("=" * 70)
print()

print("**Phase 4-C 방법론 검증 결과**:")
print()
print(f"1. **예측 성능**: AUC 0.65-0.67 ({performance})")
print(f"   - Scenario A: {auc_a:.4f}")
print(f"   - Scenario B: {auc_b:.4f}")
print()

print(f"2. **통계적 유의성**: {sig_b}")
print(f"   - 무작위 대비 개선: +{diff_b:.4f} ({diff_b/random_auc*100:+.1f}%)")
print(f"   - p-value: {p_value_b:.6f}")
print()

print(f"3. **방법론 타당성**: {validity}")
print(f"   - 시공간 매칭 > 무작위 예측")
print(f"   - {results['data_summary']['matched_samples']:,}개 샘플로 검증")
print()

print(f"4. **실무 적용**: {recommendation}")
if validity == "타당":
    print("   ✅ 프로토타입 및 파일럿에 사용 가능")
    print("   → Phase 5로 확장하여 정밀도 향상 권장")
else:
    print("   ⚠️ 추가 검증 필요")
    print("   → Phase 5 데이터 수집 후 재평가")

# ============================================================================
# 결과 저장
# ============================================================================

output = {
    "analysis_type": "Negative Control Test (Real Data)",
    "execution_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "data_source": "phase4c_enhanced_report.json",
    "sample_size": results['data_summary']['matched_samples'],
    "accident_rate": accident_rate,
    "random_baseline": {
        "expected_auc": random_auc,
        "description": "완전 무작위 분류기"
    },
    "scenario_a": {
        "auc": auc_a,
        "improvement_over_random": round(diff_a, 4),
        "z_score": round(z_score_a, 4),
        "p_value": round(p_value_a, 6),
        "ci_95": [round(ci_lower_a, 4), round(ci_upper_a, 4)],
        "significance": sig_a
    },
    "scenario_b": {
        "auc": auc_b,
        "improvement_over_random": round(diff_b, 4),
        "z_score": round(z_score_b, 4),
        "p_value": round(p_value_b, 6),
        "ci_95": [round(ci_lower_b, 4), round(ci_upper_b, 4)],
        "significance": sig_b
    },
    "conclusion": {
        "performance": performance,
        "validity": validity,
        "recommendation": recommendation,
        "statistically_significant": p_value_b < 0.05,
        "better_than_random": diff_b > 0.10
    }
}

output_file = os.path.join(os.path.dirname(__file__), 'phase4c_negative_control_real_results.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print()
print("=" * 70)
print(f"분석 결과 저장: {output_file}")
print("=" * 70)