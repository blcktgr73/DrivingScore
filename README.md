# DrivingScore: Open Data Driven Safety Score Research

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-Phase%202%20completed-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.13+-blue.svg)

> Research project that studies how publicly available data can build a reliable driving safety score.

## Project Overview

DrivingScore focuses on evidence-driven analysis first, productization second. We analyse public (and simulated) mobility datasets to build and validate a transparent scoring model that rewards safe behaviour and highlights risky driving patterns.

### Research Goals

- Quantify how driving events relate to accident risk.
- Measure environmental factors (night driving, weather, road type).
- Derive data-backed penalty weights for a safety score.
- Define and validate tier thresholds (SAFE / MODERATE / AGGRESSIVE).
- Benchmark classical and gradient-boosted models before deployment.

### Phase Progress

| Phase | Date | Highlights |
| --- | --- | --- |
| Phase 1 (completed) | 2025-09-27 | Identified top-risk events, confirmed night/weather impact, recommended 3-event score (rapid-acceleration, sudden-stop, sharp-turn). |
| Phase 2 (completed) | 2025-09-27 | Implemented scoring pipeline, compared Scenario A (with overspeeding) vs Scenario B (without), produced weights, thresholds, model benchmarks. |
| Phase 3 (planned) | TBA | Real-data validation (Kaggle), calibration, deployment playbook. |

Key insight so far: overspeeding provides limited predictive lift. Scenario B (3 events) yields cleaner implementation, while Scenario A (4 events) offers slightly better recall. See `docs/Phase2_Report.md` for the full comparison.

---

## Repository Layout

```
DrivingScore/
└── docs/
    ├── PLAN.md                  # End-to-end research plan (updated with Phase 2 results)
    ├── Phase1_Final_Report.md   # Phase 1 statistics and findings
    ├── Phase2_Report.md         # Phase 2 scenario comparison (with vs without overspeeding)
    ├── Safety_Score_Spec.md     # Scoring specification (day/night penalties)
    └── Public_Data.md           # Public datasets roadmap
└── research/
    ├── analysis_no_viz.py       # Phase 1 baseline analysis
    ├── overspeed_analysis.py    # Phase 1 overspeed scenario study
    ├── phase1_improved_analysis.py
    ├── phase2_model_development.py # Phase 2 pipeline (scenario A/B comparison)
    ├── phase2_results.json      # Machine-readable results from Phase 2
    └── requirements.txt        # Python dependencies
└── README.md
```

---

## Quickstart

```bash
# clone repository
git clone https://github.com/blcktgr73/DrivingScore.git
cd DrivingScore

# install dependencies
pip install -r research/requirements.txt
```

### Run Phase 1 Studies

```bash
cd research
python analysis_no_viz.py
python phase1_improved_analysis.py
python overspeed_analysis.py
```

### Run Phase 2 Scenario Comparison

```bash
cd research
python phase2_model_development.py
```

The script generates `phase2_results.json` capturing penalty weights, environmental multipliers, grade thresholds, and model metrics (Logistic Regression / XGBoost / LightGBM) for both scenarios.

---

## Phase Highlights

### Phase 1 Key Findings

- Sudden stops and rapid acceleration are the strongest accident predictors (Spearman 0.1608 and 0.1172).
- Night driving increases crash probability by ~20%; severe weather adds ~25%.
- Existing 1.5x night penalty is statistically justified (p < 0.0001).
- Recommended dropping overspeeding from the production score due to low lift and GPS complexity.

### Phase 2 Scenario Comparison

| Metric | Scenario A (with overspeeding) | Scenario B (without) | Delta (B - A) |
| --- | --- | --- | --- |
| Aggressive threshold | ≤ 62 | ≤ 72 | +10 |
| Safe threshold | ≥ 70 | ≥ 77 | +7 |
| SAFE share | 22.98% | 26.15% | +3.17 pp |
| SAFE accident rate | 38.1% | 41.6% | +3.5 pp |
| Logistic AUC | 0.8445 | 0.8416 | -0.0029 |
| LightGBM AUC | 0.8399 | 0.8364 | -0.0035 |

Overspeeding slightly improves predictive metrics but increases implementation complexity. Scenario B keeps penalties simple (3 events) while Scenario A captures more aggressive drivers.

---

## Roadmap

1. Pull selected Kaggle datasets (Porto Seguro, US Accidents, Driver Behaviour) and validate Phase 2 parameters on real data.
2. Calibrate SAFE tier accident rates (target < 40%).
3. Plan Phase 3 deliverables: cross-validation workflow, score migration guide, monitoring dashboard.

---

## Release Tags

- `v1.0.0-phase1` – Phase 1 findings complete.
- `v2.0.0-phase2` – Phase 2 scenario comparison complete (current release).

---

## Contributing

Issues, pull requests, and dataset suggestions are welcome. See the docs folder for full context before contributing.

---

## License

Distributed under the MIT License. See `LICENSE` for details.
