# Pre-Deployment Diagnostics Report

Generated on: 2026-07-20 15:58:29

This report summarizes the dataset checks and model verification results required prior to production deployment.

---

## 1. Class Distribution (Target: `Placement Status`)

| Status | Count | Percentage |
|---|---|---|
| **Not Placed (0)** | 45,541 | 45.54% |
| **Placed (1)** | 54,459 | 54.46% |
| **Total** | 100,000 | 100.00% |

*Interpretation: The dataset is relatively balanced with a reasonable representation of both classes.*

---

## 2. Missing Values Analysis

- **Total missing values in dataset**: 0

*Observation: Zero missing values detected across all columns. No imputation or cleanup is required for missing data.*

---

## 3. Feature Correlations (Top 10 Absolute Pearson Correlations)

| Feature 1 | Feature 2 | Pearson Correlation |
|---|---|---|
| 10th Percentage | CGPA | 0.8566 |
| Programming Score | Technical Skills | 0.8301 |
| 12th Percentage | CGPA | 0.7950 |
| Workshops | Certifications | 0.7450 |
| 12th Percentage | 10th Percentage | 0.6846 |
| Resume Score | Internships | 0.5829 |
| Soft Skills | Communication Skills | 0.5797 |
| Resume Score | Projects Completed | 0.5334 |
| Resume Score | Certifications | 0.3501 |
| Workshops | Hackathons | 0.3201 |

*Interpretation: High absolute correlations indicate strong relationships. Ensure no features trigger data leakage (e.g., 'Expected Salary' has been excluded from predictors because it is derived post-placement).*

---

## 4. Confusion Matrix (On Test Set)

| Actual \ Predicted | Predicted Not Placed (0) | Predicted Placed (1) |
|---|---|---|
| **Actual Not Placed (0)** | **TN:** 2,619 | **FP:** 6,489 |
| **Actual Placed (1)** | **FN:** 2,101 | **TP:** 8,791 |

---

## 5. Classification Metrics (On Test Set)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Not Placed** | 0.5549 | 0.2875 | 0.3788 | 9,108.0 |
| **Placed** | 0.5753 | 0.8071 | 0.6718 | 10,892.0 |
| **Accuracy** | | | 0.5705 | 20,000.0 |
| **Macro Average** | 0.5651 | 0.5473 | 0.5253 | 20,000.0 |
| **Weighted Average** | 0.5660 | 0.5705 | 0.5384 | 20,000.0 |

