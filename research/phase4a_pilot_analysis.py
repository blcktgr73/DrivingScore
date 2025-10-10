#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4-A: 파일럿 분석 (새 기준 적용)
- Δspeed ≥ 10 km/h/s (3초 지속)
- Centrifugal Acceleration Jump ≥ 400 degree m/s^2
"""

import sys
import os
import json
import math
import random
from datetime import datetime, timedelta
from collections import defaultdict

# UTF-8 콘솔 출력
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 새 기준 유틸
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(SCRIPT_DIR)
try:
    from detection_criteria import (
        count_rapid_accel_events_kmh,
        count_rapid_decel_events_kmh,
        count_sharp_turn_events_jump,
    )
except Exception as e:
    print("[WARN] detection_criteria import 실패:", e)
    # 더 진행하되, 기본 기준으로 폴백
    count_rapid_accel_events_kmh = None
    count_rapid_decel_events_kmh = None
    count_sharp_turn_events_jump = None


def normal_random(mean_val, std_val):
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mean_val + z * std_val


class Phase4APilot:
    def __init__(self):
        self.vehicle_sensor_sample = []

    def expand_vehicle_sensor_data(self, target_samples=2500):
        print("=" * 60)
        print("🚗 Phase 4-A: Vehicle Sensor 샘플 생성 (새 기준)")
        print("=" * 60)

        base_patterns = {
            "NORMAL": {
                "AccX_mean": 0.5, "AccX_std": 0.3,
                "GyroZ_mean": 0.1, "GyroZ_std": 0.2,
                "speed_mean": 60.0, "speed_std": 12.0,
                "aggressive_prob": 0.10
            },
            "AGGRESSIVE": {
                "AccX_mean": 1.8, "AccX_std": 0.8,
                "GyroZ_mean": 0.8, "GyroZ_std": 0.5,
                "speed_mean": 80.0, "speed_std": 18.0,
                "aggressive_prob": 0.75
            },
            "SLOW": {
                "AccX_mean": 0.2, "AccX_std": 0.15,
                "GyroZ_mean": 0.05, "GyroZ_std": 0.1,
                "speed_mean": 45.0, "speed_std": 10.0,
                "aggressive_prob": 0.02
            }
        }

        start_time = datetime(2022, 1, 1)

        for i in range(target_samples):
            style = random.choices(list(base_patterns.keys()), weights=[0.6, 0.3, 0.1])[0]
            p = base_patterns[style]

            # 측정 시각
            random_days = random.randint(0, 730)
            random_hours = random.randint(0, 23)
            ts = start_time + timedelta(days=random_days, hours=random_hours)

            # 센서 생성
            accx = normal_random(p['AccX_mean'], p['AccX_std'])
            gyroz = normal_random(p['GyroZ_mean'], p['GyroZ_std'])
            speed = max(0.0, normal_random(p['speed_mean'], p['speed_std']))
            accy = normal_random(0, 0.3)
            accz = normal_random(9.8, 0.2)

            # 새 기준 적용 (4초 @1Hz)
            sampling_hz = 1.0
            speeds_kmh = []
            s = speed
            for _ in range(4):
                s = max(0.0, s + accx * 3.6 + random.uniform(-0.8, 0.8))
                speeds_kmh.append(s)

            if all(func is not None for func in [count_rapid_accel_events_kmh, count_rapid_decel_events_kmh, count_sharp_turn_events_jump]):
                accel_mag = math.sqrt(accx**2 + accy**2 + accz**2)
                gyro_series = [gyroz + random.uniform(-0.2, 0.2) for _ in range(4)]
                accel_series = [accel_mag + random.uniform(-0.2, 0.2) for _ in range(4)]
                rapid_accel = 1 if count_rapid_accel_events_kmh(speeds_kmh, sampling_hz) > 0 else 0
                sudden_stop = 1 if count_rapid_decel_events_kmh(speeds_kmh, sampling_hz) > 0 else 0
                sharp_turn = 1 if count_sharp_turn_events_jump(gyro_series, accel_series, sampling_hz) > 0 else 0
            else:
                # 폴백 (기존 AccX/GyroZ 임계)
                rapid_accel = 1 if accx > 1.2 else 0
                sudden_stop = 1 if accx < -1.2 else 0
                sharp_turn = 1 if abs(gyroz) > 1.0 else 0

            overspeeding = 1 if (sum(speeds_kmh)/len(speeds_kmh)) > 100 else 0

            sensor = {
                "ID": f"S{i+1:06d}",
                "Timestamp": ts,
                "Style": style,
                "AccX": accx,
                "AccY": accy,
                "AccZ": accz,
                "GyroZ": gyroz,
                "Speed": s,
                "RapidAccel": rapid_accel,
                "SuddenStop": sudden_stop,
                "SharpTurn": sharp_turn,
                "OverSpeeding": overspeeding,
                "IsAggressive": 1 if random.random() < p['aggressive_prob'] else 0,
                "Is_Night": 1 if (random_hours >= 18 or random_hours <= 6) else 0
            }
            self.vehicle_sensor_sample.append(sensor)

        print(f"생성 완료: {len(self.vehicle_sensor_sample):,}개")
        self._summary()

    def _summary(self):
        style_dist = defaultdict(int)
        for s in self.vehicle_sensor_sample:
            style_dist[s['Style']] += 1
        print("\n운전 스타일 분포:")
        for k in sorted(style_dist.keys()):
            pct = style_dist[k] / len(self.vehicle_sensor_sample) * 100
            print(f"  {k}: {style_dist[k]:,}개 ({pct:.1f}%)")

        total_events = sum(s['RapidAccel'] + s['SuddenStop'] + s['SharpTurn'] + s['OverSpeeding'] for s in self.vehicle_sensor_sample)
        print(f"\n총 이벤트 발생: {total_events:,}건")


def main():
    print("🚀 Phase 4-A (새 기준) 실행")
    p = Phase4APilot()
    p.expand_vehicle_sensor_data(target_samples=2500)
    out = os.path.join(SCRIPT_DIR, 'phase4a_pilot_results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({"samples": len(p.vehicle_sensor_sample)}, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {out}")


if __name__ == '__main__':
    main()
