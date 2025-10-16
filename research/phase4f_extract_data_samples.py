#!/usr/bin/env python3
"""
Phase 4F - 실제 데이터 샘플 추출
사고 발생/미발생 각 4개씩 대표 샘플 추출
"""
import sys
import json
import random

# UTF-8 출력 설정 (Windows)
sys.stdout.reconfigure(encoding='utf-8')

def load_data():
    """데이터 로드"""
    print("📂 데이터 로딩 중...")
    with open('phase4f_combined_20k.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 데이터는 리스트 형태
    samples = data['data']

    # Risk/Safe 그룹 분리
    risk_samples = []
    safe_samples = []

    for i, sample in enumerate(samples):
        sample_with_id = {
            'id': f"S{i+1:05d}",
            'accident': sample['label'] == 1,  # label 1 = Risk
            'features': sample['features']
        }

        if sample['label'] == 1:
            risk_samples.append(sample_with_id)
        else:
            safe_samples.append(sample_with_id)

    print(f"✅ Risk Group: {len(risk_samples)}명")
    print(f"✅ Safe Group: {len(safe_samples)}명")

    return risk_samples, safe_samples

def analyze_sample(sample):
    """샘플 분석하여 특징 계산"""
    f = sample['features']

    # 기본 이벤트
    rapid_accel = f['rapid_accel']
    sudden_stop = f['sudden_stop']
    sharp_turn = f['sharp_turn']
    over_speed = f['over_speed']
    is_night = f['is_night']

    # 엔지니어링 특징
    total_events = rapid_accel + sudden_stop + sharp_turn + over_speed
    risky_ratio = (rapid_accel + sudden_stop) / max(total_events, 1)
    night_risky = (rapid_accel + sudden_stop) * is_night * 1.5
    emergency = min(rapid_accel, sudden_stop)
    overspeed_turn = over_speed * sharp_turn

    # Linear scoring (Scenario A)
    day_penalties = {
        'rapid_accel': 4.07,
        'sudden_stop': 1.62,
        'sharp_turn': 3.92,
        'over_speed': 5.00
    }

    multiplier = 1.5 if is_night else 1.0
    deduction = (
        rapid_accel * day_penalties['rapid_accel'] * multiplier +
        sudden_stop * day_penalties['sudden_stop'] * multiplier +
        sharp_turn * day_penalties['sharp_turn'] * multiplier +
        over_speed * day_penalties['over_speed'] * multiplier
    )

    score = max(0, min(100, 100 - deduction))

    # 등급 결정
    if score >= 80:
        grade = 'SAFE'
    elif score >= 60:
        grade = 'MODERATE'
    else:
        grade = 'AGGRESSIVE'

    return {
        'id': sample['id'],
        'accident': sample['accident'],
        'features': {
            'rapid_accel': rapid_accel,
            'sudden_stop': sudden_stop,
            'sharp_turn': sharp_turn,
            'over_speed': over_speed,
            'is_night': is_night
        },
        'engineered': {
            'total_events': total_events,
            'risky_ratio': round(risky_ratio, 3),
            'night_risky': round(night_risky, 3),
            'emergency': emergency,
            'overspeed_turn': overspeed_turn
        },
        'scoring': {
            'deduction': round(deduction, 2),
            'score': int(score),
            'grade': grade
        }
    }

def select_representative_samples(samples, count=4):
    """대표적인 샘플 선택 (다양성 확보)"""
    analyzed = [analyze_sample(s) for s in samples]

    # 점수별로 정렬
    analyzed.sort(key=lambda x: x['scoring']['score'])

    # 점수 분포를 고려하여 균등하게 선택
    step = len(analyzed) // count
    selected = []

    for i in range(count):
        idx = min(i * step, len(analyzed) - 1)
        selected.append(analyzed[idx])

    return selected

def format_sample_report(samples, group_name):
    """샘플 리포트 포맷팅"""
    report = f"\n## {group_name}\n\n"

    for i, sample in enumerate(samples, 1):
        f = sample['features']
        e = sample['engineered']
        s = sample['scoring']

        report += f"### 샘플 {i}: ID {sample['id']}\n\n"
        report += f"**사고 여부**: {'✅ 사고 발생' if sample['accident'] else '⭕ 사고 없음'}\n\n"

        # 기본 이벤트
        report += "**운전 이벤트:**\n"
        report += f"- 급가속: {f['rapid_accel']}회\n"
        report += f"- 급정거: {f['sudden_stop']}회\n"
        report += f"- 급회전: {f['sharp_turn']}회\n"
        report += f"- 과속: {f['over_speed']}회\n"
        report += f"- 야간 주행: {'예' if f['is_night'] else '아니오'}\n\n"

        # 엔지니어링 특징
        report += "**분석 지표:**\n"
        report += f"- 총 이벤트: {e['total_events']}회\n"
        report += f"- 위험 비율: {e['risky_ratio']:.1%} (급가속+급정거/전체)\n"
        report += f"- 야간 위험도: {e['night_risky']:.2f}\n"
        report += f"- 긴급 상황: {e['emergency']}회\n"
        report += f"- 과속 중 회전: {e['overspeed_turn']}회\n\n"

        # 점수 및 등급
        report += "**운전 점수:**\n"
        report += f"- 총 감점: {s['deduction']:.2f}점\n"
        report += f"- 최종 점수: {s['score']}점\n"
        report += f"- 등급: **{s['grade']}**\n\n"

        # 분석
        report += "**특징 분석:**\n"
        if e['total_events'] >= 8:
            report += f"- ⚠️ 위험 이벤트가 매우 많음 ({e['total_events']}회)\n"
        elif e['total_events'] >= 5:
            report += f"- ⚠️ 위험 이벤트가 다소 많음 ({e['total_events']}회)\n"
        else:
            report += f"- ✅ 위험 이벤트가 적음 ({e['total_events']}회)\n"

        if e['risky_ratio'] >= 0.7:
            report += f"- ⚠️ 급가속/급정거 비율이 높음 ({e['risky_ratio']:.1%})\n"

        if f['is_night'] and (f['rapid_accel'] > 0 or f['sudden_stop'] > 0):
            report += f"- ⚠️ 야간 운전 중 급가속/급정거 발생 (위험도 1.5배)\n"

        if e['overspeed_turn'] > 0:
            report += f"- ⚠️ 과속 중 급회전 {e['overspeed_turn']}회 (매우 위험)\n"

        if s['grade'] == 'AGGRESSIVE':
            report += f"- 🚨 AGGRESSIVE 등급: 즉각적인 개선 필요\n"
        elif s['grade'] == 'MODERATE':
            report += f"- ⚠️ MODERATE 등급: 주의 필요\n"
        else:
            report += f"- ✅ SAFE 등급: 안전 운전\n"

        report += "\n---\n\n"

    return report

def main():
    """메인 함수"""
    print("=" * 60)
    print("Phase 4F - 실제 데이터 샘플 추출")
    print("=" * 60)

    # 1. 데이터 로드
    risk_samples, safe_samples = load_data()

    # 2. 대표 샘플 선택
    print("\n📊 대표 샘플 선택 중...")
    risk_selected = select_representative_samples(risk_samples, count=4)
    safe_selected = select_representative_samples(safe_samples, count=4)

    print(f"✅ Risk Group에서 {len(risk_selected)}개 선택")
    print(f"✅ Safe Group에서 {len(safe_selected)}개 선택")

    # 3. 리포트 생성
    print("\n📝 리포트 생성 중...")

    report = "# Phase 4F - 실제 데이터 샘플 리포트\n\n"
    report += "**생성일**: 2025년 10월 16일\n"
    report += "**데이터셋**: phase4f_combined_20k.json (20,000명)\n"
    report += "**샘플 수**: Risk 4개 + Safe 4개 = 총 8개\n\n"

    report += "## 개요\n\n"
    report += "Phase 4F에서 사용된 실제 데이터 중 대표적인 샘플을 추출하여 분석합니다.\n"
    report += "사고 발생 그룹(Risk)과 사고 미발생 그룹(Safe)에서 각각 4개씩, "
    report += "점수 분포를 고려하여 다양한 운전 패턴을 보여주는 샘플을 선택했습니다.\n\n"

    report += "### 점수 계산 방법\n\n"
    report += "**Scenario A (4개 이벤트) Linear Scoring:**\n"
    report += "- 급가속: 1회당 4.07점 감점 (야간 6.10점)\n"
    report += "- 급정거: 1회당 1.62점 감점 (야간 2.42점)\n"
    report += "- 급회전: 1회당 3.92점 감점 (야간 5.89점)\n"
    report += "- 과속: 1회당 5.00점 감점 (야간 7.50점)\n\n"

    report += "**등급 기준:**\n"
    report += "- SAFE: 80-100점\n"
    report += "- MODERATE: 60-79점\n"
    report += "- AGGRESSIVE: 0-59점\n\n"

    # Risk Group 샘플
    report += format_sample_report(risk_selected, "사고 발생 그룹 (Risk Group)")

    # Safe Group 샘플
    report += format_sample_report(safe_selected, "사고 미발생 그룹 (Safe Group)")

    # 4. 통계 요약
    report += "\n## 샘플 통계 요약\n\n"

    risk_scores = [s['scoring']['score'] for s in risk_selected]
    safe_scores = [s['scoring']['score'] for s in safe_selected]

    risk_events = [s['engineered']['total_events'] for s in risk_selected]
    safe_events = [s['engineered']['total_events'] for s in safe_selected]

    report += "### 점수 분포\n\n"
    report += f"**Risk Group:**\n"
    report += f"- 최저 점수: {min(risk_scores)}점\n"
    report += f"- 최고 점수: {max(risk_scores)}점\n"
    report += f"- 평균 점수: {sum(risk_scores)/len(risk_scores):.1f}점\n\n"

    report += f"**Safe Group:**\n"
    report += f"- 최저 점수: {min(safe_scores)}점\n"
    report += f"- 최고 점수: {max(safe_scores)}점\n"
    report += f"- 평균 점수: {sum(safe_scores)/len(safe_scores):.1f}점\n\n"

    report += "### 이벤트 발생 빈도\n\n"
    report += f"**Risk Group:**\n"
    report += f"- 최소 이벤트: {min(risk_events)}회\n"
    report += f"- 최대 이벤트: {max(risk_events)}회\n"
    report += f"- 평균 이벤트: {sum(risk_events)/len(risk_events):.1f}회\n\n"

    report += f"**Safe Group:**\n"
    report += f"- 최소 이벤트: {min(safe_events)}회\n"
    report += f"- 최대 이벤트: {max(safe_events)}회\n"
    report += f"- 평균 이벤트: {sum(safe_events)/len(safe_events):.1f}회\n\n"

    # 5. 인사이트
    report += "## 주요 인사이트\n\n"

    risk_avg = sum(risk_scores) / len(risk_scores)
    safe_avg = sum(safe_scores) / len(safe_scores)

    report += f"1. **점수 차이**: Safe Group이 Risk Group보다 평균 {safe_avg - risk_avg:.1f}점 높음\n"
    report += f"2. **이벤트 빈도**: Risk Group이 평균 {sum(risk_events)/len(risk_events):.1f}회, "
    report += f"Safe Group이 평균 {sum(safe_events)/len(safe_events):.1f}회 발생\n"

    # 등급 분포
    risk_grades = [s['scoring']['grade'] for s in risk_selected]
    safe_grades = [s['scoring']['grade'] for s in safe_selected]

    report += f"3. **등급 분포**:\n"
    report += f"   - Risk Group: AGGRESSIVE {risk_grades.count('AGGRESSIVE')}개, "
    report += f"MODERATE {risk_grades.count('MODERATE')}개, "
    report += f"SAFE {risk_grades.count('SAFE')}개\n"
    report += f"   - Safe Group: AGGRESSIVE {safe_grades.count('AGGRESSIVE')}개, "
    report += f"MODERATE {safe_grades.count('MODERATE')}개, "
    report += f"SAFE {safe_grades.count('SAFE')}개\n"

    report += f"4. **모델 성능**: 실제 사고 여부와 점수/등급의 상관관계를 통해 모델의 변별력 확인\n"

    report += "\n---\n\n"
    report += "*본 리포트는 `phase4f_extract_data_samples.py`에 의해 자동 생성되었습니다.*\n"

    # 6. 파일 저장
    output_file = '../docs/Phase4F_Data_Sample_Report.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ 리포트 생성 완료: {output_file}")

    # 7. JSON 저장 (재사용 가능)
    json_output = {
        'risk_samples': risk_selected,
        'safe_samples': safe_selected,
        'statistics': {
            'risk_scores': {
                'min': min(risk_scores),
                'max': max(risk_scores),
                'avg': round(sum(risk_scores) / len(risk_scores), 1)
            },
            'safe_scores': {
                'min': min(safe_scores),
                'max': max(safe_scores),
                'avg': round(sum(safe_scores) / len(safe_scores), 1)
            }
        }
    }

    json_file = 'phase4f_data_samples.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON 저장 완료: {json_file}")

    print("\n" + "=" * 60)
    print("✅ 모든 작업 완료!")
    print("=" * 60)

if __name__ == '__main__':
    main()
