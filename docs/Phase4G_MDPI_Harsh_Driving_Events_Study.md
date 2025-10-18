# Modeling and Sustainability Implications of Harsh Driving Events: A Predictive Machine Learning Approach

**Authors**: Antonis Kostopoulos, Thodoris Garefalakis*, Eva Michelaraki, Christos Katrakazas, George Yannis

**Affiliation**: Department of Transportation Planning and Engineering, National Technical University of Athens, 5 Iroon Polytechniou Str., 157 73 Athens, Greece

**Correspondence**: tgarefalakis@mail.ntua.gr

**Published**: Sustainability 2024, 16, 6151
**DOI**: https://doi.org/10.3390/su16146151
**License**: CC BY 4.0

---

## Abstract

Human behavior significantly contributes to severe road injuries, underscoring a critical road safety challenge. This study addresses the complex task of predicting dangerous driving behaviors through a comprehensive analysis of **over 356,000 trips**, enhancing existing knowledge in the field and promoting sustainability and road safety.

The research uses advanced machine learning algorithms (e.g., **Random Forest, Gradient Boosting, Extreme Gradient Boosting, Multilayer Perceptron, and K-Nearest Neighbors**) to categorize driving behaviors into 'Dangerous' and 'Non-Dangerous'. Feature selection techniques are applied to enhance the understanding of influential driving behaviors, while **k-means clustering** establishes reliable safety thresholds.

### Key Findings

- **Gradient Boosting** and **Multilayer Perceptron** excel, achieving **recall rates of approximately 67% to 68%** for both harsh acceleration and braking events
- Critical thresholds identified:
  - **(a) 48.82 harsh accelerations per 100 km**
  - **(b) 45.40 harsh brakings per 100 km**

The application of machine learning algorithms, feature selection, and k-means clustering offers a promising approach for improving road safety and reducing socio-economic costs through sustainable practices.

**Keywords**: road traffic safety; naturalistic driving experiment; driving behavior analysis; driving behavior; harsh events; machine learning

---

## 1. Introduction

Road safety is a major concern for the institutions of the European Union and its national components. According to research conducted by the **World Health Organization**, approximately **1.19 million human deaths per year** are related to road crashes, making it the leading cause of fatalities for individuals aged 5–29 years.

The European Union has implemented substantial measures to eliminate road fatalities by adopting **Vision Zero**, which aims to eradicate fatal road crashes by 2050. The mid-term goal is to **reduce fatalities by 50% during the decade 2021–2030**.

### Key Statistics

- **Excessive speeding and acceleration-deceleration** relationships are key factors in approximately **30% of fatal crashes**
- A **1% increase in average speed** leads to:
  - ~**2% increase** in mild injuries
  - ~**3% increase** in serious injuries
  - ~**4% increase** in fatalities

**Harsh events** (specifically acceleration and braking) play a pivotal role as key indicators in the evaluation of driving risk, particularly when assessing the degree of driving aggressiveness.

---

## 2. Literature Review

In recent years, with the advancement of **Intelligent Transport Systems (ITS)**, research has focused on the analysis of driving behavior to make safety systems capable of predicting and improving dangerous driving behavior.

### Key Studies

**Papadimitriou et al. (2019)** proposed a methodology to quantify the correlation between dangerous driving behavior and mobile phone use:
- Binary logistic regression model with **70% accuracy**
- Large correlation between harsh events and mobile phone use

**Yang et al. (2021)** evaluated driving performance in real-time:
- Applied K-means, hierarchical clustering, and GMM
- Resulted in **4 optimal safety levels**
- SVM classification with **97.9% overall accuracy**

**Yarlagadda et al. (2021)** proposed a framework using k-means clustering:
- Identified **13.5% of accelerations** and **34.7% of braking maneuvers** as aggressive
- Established thresholds for aggressive maneuvers

---

## 3. Materials and Methods

### 3.1 Naturalistic Driving Experiment

The naturalistic driving dataset was collected and provided by **OSeven Telematics, London, UK** through a specialized smartphone application that records and collects driving data continuously without interference.

**Data Collection Specifications:**
- **Hardware sensors**: Accelerometer, gyroscope, magnetometer, GPS
- **Data fusion**: iOS and Android with 9 degrees of freedom models (Yaw, Pitch, Roll)
- **Recording frequency**: Maximum 1 Hz
- **Recording interval**: Continuous at one-second intervals

**Dataset Details:**
- **356,162 different trips** in urban road network
- **75 indexes** provided for each trip
- **23 key variables** analyzed in this study
- **Collection period**: January 2020 - December 2020 (during COVID-19 pandemic)

**Privacy Compliance:**
- GDPR compliant
- All data anonymized
- No geolocation information included (except country)

### 3.2 Definition of Driving Behavior Levels

The **k-means clustering method** was selected to find numerical thresholds for safety levels, with two clusters representing the predefined classes (Non-Dangerous and Dangerous).

#### 3.2.1 Based on Harsh Accelerations Events per 100 km

**K-means Results:**
- **Centroid 1**: 5.693 events per 100 km
- **Centroid 2**: 91.942 events per 100 km
- **Threshold**: **48.817 harsh acceleration events per 100 km**

**Distribution:**
- **Non-Dangerous**: 330,395 trips (93%)
- **Dangerous**: 25,767 trips (7%)

#### 3.2.2 Based on Harsh Braking Events per 100 km

**K-means Results:**
- **Centroid 1**: 7.975 events per 100 km
- **Centroid 2**: 82.835 events per 100 km
- **Threshold**: **45.405 harsh braking events per 100 km**

**Distribution:**
- **Non-Dangerous**: 315,986 trips (89%)
- **Dangerous**: 40,176 trips (11%)

### 3.3 Feature Selection

**Permutation importance-based feature selection** was utilized to identify the most effective independent variables.

**Key Findings:**
- **Most important features**: Total distance, driving duration
- **Least important**: Mobile usage time, risky hours distance

**Input Variables for Classification:**
- Total distance
- Driving duration
- Average driving speed
- Speeding score
- Mobile use score

**Descriptive Statistics:**

| Variable | Description | Mean | St. Dev. | Min | Max |
|----------|-------------|------|----------|-----|-----|
| Total distance | Total trip distance (km) | 11.60 | 22.31 | 0.50 | 648.68 |
| Driving duration | Total driving time (s) | 769.97 | 967.15 | 61.00 | 23,900.00 |
| Average driving speed | Average speed in motion (km/h) | 42.57 | 17.58 | 5.57 | 183.91 |
| Speeding score | Excessive speed score (%) | 76.52% | 32.92 | 10.00% | 100.00% |

### 3.4 Classification Process

Five classification models were proposed:
1. **Random Forest (RF)**
2. **Gradient Boosting (GB)**
3. **Extreme Gradient Boosting (XGBoost)**
4. **K-Nearest Neighbors (kNN)**
5. **Multilayer Perceptron (MLP)**

**Performance Metrics:**
- Accuracy = (TP + TN) / (TP + FP + FN + TN)
- Precision = TP / (TP + FP)
- Sensitivity (Recall) = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- F-1 Score = 2 × (Precision × Recall) / (Precision + Recall)
- False Alarm Rate = FP / (FP + TN)

---

## 4. Results

### 4.1 Evaluation of Classification Models for Harsh Acceleration Events

**Performance Metrics:**

| Model | Accuracy | Precision | Recall | False Alarm Rate | F-1 Score | AUC |
|-------|----------|-----------|--------|------------------|-----------|-----|
| **RF** | 70.83% | 55.16% | 66.39% | 33.61% | 52.64% | 73.98% |
| **GB** | 65.28% | 55.15% | **68.05%** | **31.95%** | 50.25% | **75.10%** |
| **XGBoost** | 66.76% | 55.09% | 67.46% | 32.54% | 50.86% | 74.26% |
| **MLP** | 68.16% | 55.26% | **67.65%** | **32.35%** | 51.63% | **74.67%** |
| **kNN** | 72.70% | 53.46% | 60.08% | 39.92% | 51.47% | 64.55% |

**Key Findings:**
- **Gradient Boosting (GB)** and **Multilayer Perceptron (MLP)** provide the best results
- **GB slightly outperforms MLP** in recall and false alarm rate
- Both models achieved satisfactory AUC scores (75.1% and 74.67%)

### 4.2 Evaluation of Classification Models for Harsh Braking Events

**Performance Metrics:**

| Model | Accuracy | Precision | Recall | False Alarm Rate | F-1 Score | AUC |
|-------|----------|-----------|--------|------------------|-----------|-----|
| **RF** | 67.78% | 57.20% | 66.48% | 33.52% | 55.09% | 73.62% |
| **GB** | 63.36% | 57.36% | **67.91%** | **32.09%** | 53.13% | **74.88%** |
| **XGBoost** | 64.53% | 57.20% | 67.30% | 32.70% | 53.60% | 74.28% |
| **MLP** | 62.96% | 57.29% | **67.80%** | **32.20%** | 52.88% | **74.69%** |
| **kNN** | 68.45% | 54.88% | 60.55% | 39.45% | 53.19% | 65.00% |

**Key Findings:**
- Similar performance to harsh acceleration analysis
- **GB and MLP** outperformed other models
- AUC scores: **74.88% (GB)** and **74.69% (MLP)**

---

## 5. Discussion

### Feature Importance

**Most Important Factors:**
- **Distance** and **total trip duration** emerged as most important
- Increased distance/duration worsens driver behavior (fatigue, impaired perception)
- **Vehicle speed** is important for sudden incidents

**Less Important:**
- Driving during dangerous time zone (00:00–05:00)
- Lower traffic and pedestrian volumes during these hours might mitigate risky behaviors

### Safety Thresholds Established

Using **k-means clustering method**:
- **(a) 48.82 harsh accelerations per 100 km**
- **(b) 45.40 harsh brakings per 100 km**

These thresholds provide:
- Systematic approach to categorizing driving behavior
- Original benchmarks for dangerous vs non-dangerous behavior
- Actionable criteria for traffic safety applications

### Model Performance

**SMOTE technique** was chosen to handle imbalanced distribution:
- Effective for large and multilevel data
- Future research should explore SVM-SMOTE and SMOTE-Tomek

**Comparison with Literature:**
- Ghandour et al. (2021): GB achieved 60% accuracy
- This study: GB achieved **63.4% accuracy**
- Significant improvement in performance metrics

### Practical Applications

**For Policymakers:**
- Develop dynamic interventions based on identified thresholds
- Real-time adaptive speed management systems
- Integration into vehicular telematics systems

**For Driver Training:**
- Personalized programs targeting risky behaviors
- Early warnings and targeted education
- Behavior-based feedback systems

---

## 6. Conclusions

This paper proposes a **comprehensive framework** for analyzing and classifying driving behavior as Dangerous or Non-Dangerous.

### Key Contributions

1. **Dataset Analysis**: Over 356,000 trips analyzed
2. **Thresholds Established**:
   - 48.82 harsh accelerations per 100 km
   - 45.40 harsh brakings per 100 km
3. **Best Models**: Gradient Boosting and Multilayer Perceptron
4. **Recall Rates**: 67-68% for both harsh event categories

### Limitations

- **Deep learning models** (CNN, RNN, LSTM) not explored due to processing constraints
- **Urban focus**: Limited generalizability to rural/highway settings
- **COVID-19 impact**: Data collected during pandemic may not represent typical conditions
- **Feature importance**: Alternative methods (Bayesian, Akaike) could provide additional insights

### Future Research Directions

1. Explore deep learning models for enhanced accuracy
2. Expand to different driving environments (rural, highway)
3. Investigate temporal dynamics under different conditions
4. Consider Gaussian Mixture Models for clustering
5. Examine weather and external factors more comprehensively

### Significance

This research offers an **innovative approach** to predicting harsh events, with the key contribution being the **distinction between dangerous and non-dangerous driving behavior based on harsh events**. The findings contribute significantly to:
- Road safety enhancement
- Personalized driver feedback
- Enhanced training programs
- Advancement in automotive industry

By adopting these techniques and identified thresholds, authorities and organizations can develop effective strategies to detect and mitigate dangerous driving behaviors, ultimately contributing to **Vision Zero** and global safety initiatives.

---

## References

[Full reference list available in original PDF - 28 references total]

---

**Data Source**: OSeven Telematics, London, UK (https://oseven.io/)
**Software**: Python with scikit-learn (version 1.0.2), XGBoost (version 1.5.2)
**License**: Open Access (CC BY 4.0)
**Funding**: No external funding received

---

*This document is a markdown conversion of the original PDF paper published in Sustainability journal (MDPI).*
