# Major Project Report: AI-Powered Student Placement Prediction System

## Abstract
The transition from academic study to corporate employment represents a critical milestone for students. Higher education institutions face challenges in identifying at-risk students who need targeted career guidance. This project introduces a comprehensive, production-ready **AI-Powered Student Placement Prediction System**. By evaluating academic averages, coding competencies, soft skills, and portfolio indices, the system predicts placement status with high accuracy. 

Using a multi-model benchmarking pipeline, we compare 13 distinct machine learning algorithms. The best model is tuned using GridSearchCV, and its predictions are explained using Explainable AI (SHAP) to clarify the feature interactions driving decisions. The system is deployed using a FastAPI backend service and an interactive Streamlit dashboard.

---

## 1. Introduction & Objectives

### Problem Statement
Traditional placement coordination relies heavily on manual GPA cutoffs. This approach fails to capture multi-dimensional factors like technical skills, projects completed, and communication capabilities. Consequently, coordinators cannot provide personalized advice to students during preparation phases.

### Project Objectives
* Develop a predictive pipeline that calculates placement probabilities and expected salary packages.
* Benchmark 13 machine learning classifiers to select the best algorithm dynamically.
* Demystify "black-box" model outputs using **Explainable AI (SHAP)**.
* Design a premium, responsive analytics dashboard that provides career advice and suggestions.
* Package the solution for microservice container deployment using Docker and SQLite.

---

## 2. Methodology & Pipeline

The system is engineered as an end-to-end data pipeline:

```
[Raw Student CSV] ➔ [Missing Imputations] ➔ [Outlier Capping] ➔ [OHE Encoders]
       ➔ [Standard Scaling] ➔ [Feature Engineering] ➔ [Multi-Model Benchmark] 
       ➔ [Grid Search Optimization] ➔ [Best Model Export] ➔ [SHAP Attributions]
```

### Data Preprocessing
* **Imputation**: Numeric features are imputed with medians; categorical fields are imputed using mode frequencies to ensure robustness.
* **Outlier Capper**: Employs an IQR filter. Numerical values outside the bounds $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$ are capped at these limits to stabilize standard scaling.
* **Encoders**: String variables like `Branch` and `Degree` are transformed using One-Hot encoding.
* **Scaling**: Numerical scores are standardized using `StandardScaler` to bring them to a mean of 0 and standard deviation of 1.

### Feature Engineering
Three custom indices are computed:
1. $\text{Academic Performance Index} = 0.50 \times (\text{CGPA} \times 10) + 0.25 \times \text{10th \%} + 0.25 \times \text{12th \%}$
2. $\text{Employability Readiness Score} = \text{Programming} \times 0.25 + \text{Aptitude} \times 0.15 + \text{Interview} \times 0.20 + \text{Resume} \times 0.15 + \text{Internship pts} \times 0.20 + \text{Project pts} \times 0.05$
3. $\text{Skills Diversity Index} = \text{Certifications} + \text{Workshops} + \text{Hackathons}$

---

## 3. Machine Learning Benchmark Analysis

The model benchmark evaluates 13 classifiers. Each model is tested using **5-fold Stratified Cross-Validation** on the training split to protect against overfitting.

### Algorithms Compared
1. **Logistic Regression**: Baseline linear classifier.
2. **Decision Tree**: Non-linear tree classifier.
3. **Random Forest**: Ensemble bagging classifier.
4. **Extra Trees**: Random forest variation.
5. **Gradient Boosting**: Sequential boosting model.
6. **AdaBoost**: Adaptive boosting method.
7. **XGBoost**: Extreme Gradient Boosting.
8. **LightGBM**: Light Gradient Boosting Machine.
9. **CatBoost**: Categorical Boosting.
10. **Support Vector Machine (SVM)**: Margins-based classifier.
11. **Naive Bayes**: Probability-based classifier.
12. **K-Nearest Neighbors (KNN)**: Distance-based classifier.
13. **MLP Classifier**: Multi-layer Perceptron (neural network).

### Dynamic Tuning
The pipeline identifies the model with the highest cross-validation F1-Score. If it's a tree-based ensemble (Random Forest, XGBoost, LightGBM, CatBoost), a grid search (`GridSearchCV`) is triggered over predefined grids to optimize parameters like max depth, learning rate, and tree count.

---

## 4. Explainable AI (SHAP)

Machine learning models are often criticised as black boxes. To make predictions actionable for students and coordinators, we integrate **SHAP (SHapley Additive exPlanations)**.

SHAP values are calculated based on cooperative game theory:
$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} (v(S \cup \{i\}) - v(S))$$

This formula determines the marginal contribution of each feature to the final prediction.
* **Waterfall Plots** explain individual predictions, showing how positive features (like high coding score) and negative features (like backlogs) pull the prediction away from the baseline average.
* **Summary Plots** show global feature importance, identifying CGPA, programming score, and internships as the strongest placement predictors.

---

## 5. Conclusion & Future Enhancements

### Conclusion
The AI-Powered Student Placement Prediction System provides a robust tool for higher education institutions. By utilizing a multi-model pipeline, we ensure that predictions are based on the best-performing model for the data. The integration of SHAP makes these predictions understandable and actionable, helping coordinators identify at-risk students and provide personalized guidance.

### Future Enhancements
* **Active Directory Integration**: Implement OAuth2 authentication for student logins.
* **Deep Learning Integration**: Test custom PyTorch/TensorFlow networks.
* **Dynamic PDF Report Generation**: Implement automated report compilation for emailing students.
* **SMS Alerts**: Send warning alerts to students with readiness scores below 60%.
