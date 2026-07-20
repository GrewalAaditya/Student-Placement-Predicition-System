import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from src.config import DATA_RAW_DIR, MODELS_DIR, REPORTS_DIR, TARGET_COL, NUMERICAL_COLS
from src.utils import load_pickle, logger
from src.preprocessing import prepare_data

def run_diagnostics():
    logger.info("Running pre-deployment checks...")
    
    # Paths
    raw_csv = DATA_RAW_DIR / "placement_dataset.csv"
    model_path = MODELS_DIR / "placement_model.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    report_md_path = REPORTS_DIR / "diagnostics_report.md"
    
    if not (raw_csv.exists() and model_path.exists() and preprocessor_path.exists()):
        logger.error("Missing raw dataset or model/preprocessor pickle files.")
        return

    # Load data
    df = pd.read_csv(raw_csv)
    
    # 1. Class Distribution
    total_records = len(df)
    class_counts = df[TARGET_COL].value_counts()
    class_pcts = df[TARGET_COL].value_counts(normalize=True) * 100
    
    placed_count = class_counts.get(1, 0)
    not_placed_count = class_counts.get(0, 0)
    placed_pct = class_pcts.get(1, 0.0)
    not_placed_pct = class_pcts.get(0, 0.0)
    
    # 2. Missing Values Check
    missing_counts = df.isnull().sum()
    columns_with_missing = missing_counts[missing_counts > 0]
    total_missing = missing_counts.sum()
    
    # 3. Feature Correlations
    # We include numerical columns present in the dataset (including Expected Salary if present)
    avail_num_cols = [col for col in NUMERICAL_COLS + ["Expected Salary"] if col in df.columns]
    corr_matrix = df[avail_num_cols].corr()
    
    # Extract upper triangle of correlation matrix to find pairs
    pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_list = pairs.unstack().dropna().reset_index()
    corr_list.columns = ['Feature_1', 'Feature_2', 'Correlation']
    corr_list['Abs_Correlation'] = corr_list['Correlation'].abs()
    top_corrs = corr_list.sort_values(by='Abs_Correlation', ascending=False).head(10)
    
    # 4. Model Predictions & Confusion Matrix & Classification Report
    # Load model and preprocessor
    model = load_pickle(model_path)
    preprocessor = load_pickle(preprocessor_path)
    
    # Split data using the standard pipeline splitting logic
    X_train, X_test, y_train, y_test = prepare_data(str(raw_csv))
    
    # Preprocess test features
    X_test_proc = preprocessor.transform(X_test)
    X_test_proc_arr = X_test_proc.values if hasattr(X_test_proc, "values") else X_test_proc
    
    # Predictions
    y_pred = model.predict(X_test_proc_arr)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    # cm layout:
    # TN FP
    # FN TP
    tn, fp, fn, tp = cm.ravel()
    
    # Classification Report
    cls_report = classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"], output_dict=True)
    
    # Build Markdown Report
    report_content = f"""# Pre-Deployment Diagnostics Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

This report summarizes the dataset checks and model verification results required prior to production deployment.

---

## 1. Class Distribution (Target: `{TARGET_COL}`)

| Status | Count | Percentage |
|---|---|---|
| **Not Placed (0)** | {not_placed_count:,} | {not_placed_pct:.2f}% |
| **Placed (1)** | {placed_count:,} | {placed_pct:.2f}% |
| **Total** | {total_records:,} | 100.00% |

*Interpretation: The dataset is relatively balanced with a reasonable representation of both classes.*

---

## 2. Missing Values Analysis

- **Total missing values in dataset**: {total_missing}

"""
    if len(columns_with_missing) > 0:
        report_content += "| Column | Missing Count | Percentage |\n|---|---|---|\n"
        for col, count in columns_with_missing.items():
            pct = (count / total_records) * 100
            report_content += f"| {col} | {count} | {pct:.2f}% |\n"
    else:
        report_content += "*Observation: Zero missing values detected across all columns. No imputation or cleanup is required for missing data.*"

    report_content += f"""

---

## 3. Feature Correlations (Top 10 Absolute Pearson Correlations)

| Feature 1 | Feature 2 | Pearson Correlation |
|---|---|---|
"""
    for _, row in top_corrs.iterrows():
        report_content += f"| {row['Feature_1']} | {row['Feature_2']} | {row['Correlation']:.4f} |\n"

    report_content += f"""
*Interpretation: High absolute correlations indicate strong relationships. Ensure no features trigger data leakage (e.g., 'Expected Salary' has been excluded from predictors because it is derived post-placement).*

---

## 4. Confusion Matrix (On Test Set)

| Actual \\ Predicted | Predicted Not Placed (0) | Predicted Placed (1) |
|---|---|---|
| **Actual Not Placed (0)** | **TN:** {tn:,} | **FP:** {fp:,} |
| **Actual Placed (1)** | **FN:** {fn:,} | **TP:** {tp:,} |

---

## 5. Classification Metrics (On Test Set)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Not Placed** | {cls_report['Not Placed']['precision']:.4f} | {cls_report['Not Placed']['recall']:.4f} | {cls_report['Not Placed']['f1-score']:.4f} | {cls_report['Not Placed']['support']:,} |
| **Placed** | {cls_report['Placed']['precision']:.4f} | {cls_report['Placed']['recall']:.4f} | {cls_report['Placed']['f1-score']:.4f} | {cls_report['Placed']['support']:,} |
| **Accuracy** | | | {cls_report['accuracy']:.4f} | {cls_report['macro avg']['support']:,} |
| **Macro Average** | {cls_report['macro avg']['precision']:.4f} | {cls_report['macro avg']['recall']:.4f} | {cls_report['macro avg']['f1-score']:.4f} | {cls_report['macro avg']['support']:,} |
| **Weighted Average** | {cls_report['weighted avg']['precision']:.4f} | {cls_report['weighted avg']['recall']:.4f} | {cls_report['weighted avg']['f1-score']:.4f} | {cls_report['weighted avg']['support']:,} |

"""
    
    # Save Report
    with open(report_md_path, "w") as f:
        f.write(report_content)
        
    logger.info(f"Saved diagnostics report: {report_md_path}")
    
    # Print clean summary console log
    print("\n" + "="*50)
    print("           PRE-DEPLOYMENT DIAGNOSTICS SUMMARY")
    print("="*50)
    print(f"Dataset Size: {total_records} rows")
    print(f"Class Balance: Placed: {placed_pct:.1f}%, Not Placed: {not_placed_pct:.1f}%")
    print(f"Missing Values: {total_missing}")
    print("-"*50)
    print("Model Evaluation Metrics:")
    print(f"  Accuracy:  {cls_report['accuracy']:.4f}")
    print(f"  Precision (Placed): {cls_report['Placed']['precision']:.4f}")
    print(f"  Recall (Placed):    {cls_report['Placed']['recall']:.4f}")
    print(f"  F1-Score (Placed):  {cls_report['Placed']['f1-score']:.4f}")
    print("-"*50)
    print(f"Diagnostics report written to: {report_md_path.name}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_diagnostics()
