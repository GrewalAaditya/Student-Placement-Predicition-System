import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report
)
from sklearn.model_selection import learning_curve
from src.config import REPORTS_DIR
from src.utils import logger

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, filepath: str) -> None:
    """Save confusion matrix plot."""
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Placed', 'Placed'],
                yticklabels=['Not Placed', 'Placed'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix: {filepath}")

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, filepath: str) -> None:
    """Save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curve: {filepath}")

def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, filepath: str) -> None:
    """Save Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"Saved Precision-Recall curve: {filepath}")

def plot_feature_importance(model: Any, feature_names: list, filepath: str) -> None:
    """Save feature importance plot."""
    plt.figure(figsize=(10, 6))
    
    importances = None
    title = "Feature Importance"
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title = "Feature Importance (Absolute Coefficients)"

    if importances is not None:
        if len(importances) == len(feature_names):
            df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            df_imp = df_imp.sort_values(by="Importance", ascending=False).head(15)

            sns.barplot(x="Importance", y="Feature", data=df_imp, palette="viridis")
            plt.title(title)
            plt.xlabel("Importance Score")
            plt.ylabel("Features")
            plt.tight_layout()
            plt.savefig(filepath, dpi=300)
            plt.close()
            logger.info(f"Saved feature importance plot: {filepath}")
        else:
            logger.warning(f"Importance size mismatch: values={len(importances)}, names={len(feature_names)}")
    else:
        logger.info("Model does not support direct feature importances. Saving fallback plot.")
        plt.text(0.5, 0.5, "Feature Importance not available", ha='center', va='center')
        plt.savefig(filepath, dpi=100)
        plt.close()

def plot_learning_curve(model: Any, X: np.ndarray, y: np.ndarray, filepath: str) -> None:
    """Save training learning curve plot."""
    plt.figure(figsize=(7, 5))
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), 
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    
    plt.title("Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"Saved learning curve: {filepath}")

def generate_all_evaluation_plots(
    model: Any, 
    preprocessor: Any, 
    X_test_proc: np.ndarray, 
    y_test: np.ndarray,
    feature_names: list
) -> None:
    """Generate all test performance diagnostics."""
    logger.info("Generating evaluation plots...")
    
    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    plot_confusion_matrix(y_test, y_pred, str(REPORTS_DIR / "confusion_matrix.png"))
    plot_roc_curve(y_test, y_prob, str(REPORTS_DIR / "roc_curve.png"))
    plot_precision_recall_curve(y_test, y_prob, str(REPORTS_DIR / "precision_recall_curve.png"))
    plot_feature_importance(model, feature_names, str(REPORTS_DIR / "feature_importance.png"))
    plot_learning_curve(model, X_test_proc, y_test, str(REPORTS_DIR / "learning_curve.png"))
    
    report_text = classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"])
    with open(REPORTS_DIR / "classification_report.txt", "w") as f:
        f.write(report_text)
    logger.info("Saved classification report text.")

