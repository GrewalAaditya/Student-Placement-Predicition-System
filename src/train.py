import uuid
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple

from sklearn.model_selection import cross_validate, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    GradientBoostingClassifier, AdaBoostClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.config import (
    DATA_RAW_DIR, MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH, HYPERPARAMETER_GRIDS
)
from src.preprocessing import prepare_data, build_preprocessing_pipeline
from src.utils import logger, save_pickle
from src.database import SessionLocal, TrainingLog

def get_base_models() -> Dict[str, Any]:
    """Get list of base classifiers for evaluation."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Extra Trees": ExtraTreesClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0)
    }

def train_and_evaluate_all(run_id: str) -> Tuple[pd.DataFrame, Dict[str, Any], Any, Any, Tuple]:
    """Train base models, tune the best performing one, and save the artifacts."""
    raw_csv = DATA_RAW_DIR / "placement_dataset.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"Missing raw data: {raw_csv}")

    X_train, X_test, y_train, y_test = prepare_data(str(raw_csv))

    logger.info("Fitting preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    cat_encoder = preprocessor.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_features_ohe = list(cat_encoder.get_feature_names_out(preprocessor.named_steps['preprocessor'].transformers_[1][2]))
    engineered_numerical_cols = preprocessor.named_steps['preprocessor'].transformers_[0][2]
    all_processed_cols = engineered_numerical_cols + cat_features_ohe

    base_models = get_base_models()
    leaderboard_data = []
    trained_models = {}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    logger.info("Evaluating classifiers using cross-validation...")
    for name, model in base_models.items():
        try:
            logger.info(f"Starting cross-validation for model: {name}")
            cv_results = cross_validate(
                model, X_train_proc, y_train, 
                cv=cv, 
                scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                n_jobs=-1
            )
            logger.info(f"Finished cross-validation for model: {name}")
            
            mean_acc = np.mean(cv_results['test_accuracy'])
            mean_prec = np.mean(cv_results['test_precision'])
            mean_rec = np.mean(cv_results['test_recall'])
            mean_f1 = np.mean(cv_results['test_f1'])
            mean_auc = np.mean(cv_results['test_roc_auc'])

            leaderboard_data.append({
                "Model": name,
                "CV Accuracy": round(mean_acc, 4),
                "CV Precision": round(mean_prec, 4),
                "CV Recall": round(mean_rec, 4),
                "CV F1-Score": round(mean_f1, 4),
                "CV ROC-AUC": round(mean_auc, 4)
            })

            logger.info(f"Starting fit for model: {name} on full dataset")
            model.fit(X_train_proc, y_train)
            logger.info(f"Finished fit for model: {name}")
            trained_models[name] = model

        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")

    leaderboard = pd.DataFrame(leaderboard_data)
    leaderboard = leaderboard.sort_values(by="CV F1-Score", ascending=False).reset_index(drop=True)
    logger.info(f"Leaderboard:\n{leaderboard.to_string()}")

    best_model_name = leaderboard.iloc[0]["Model"]
    logger.info(f"Selected best model: {best_model_name}")

    best_estimator = trained_models[best_model_name]
    best_params = {}
    
    if best_model_name in HYPERPARAMETER_GRIDS:
        logger.info(f"Tuning hyperparameters for {best_model_name}...")
        grid = HYPERPARAMETER_GRIDS[best_model_name]
        
        raw_models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
            "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
            "CatBoost": CatBoostClassifier(random_state=42, verbose=0)
        }
        
        grid_search = RandomizedSearchCV(
            estimator=raw_models[best_model_name],
            param_distributions=grid,
            n_iter=5,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            random_state=42
        )
        logger.info(f"Starting RandomizedSearchCV fit for model: {best_model_name}")
        grid_search.fit(X_train_proc, y_train)
        logger.info(f"Finished RandomizedSearchCV fit for model: {best_model_name}")
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        logger.info(f"Tuning complete. Best params: {best_params}")

    logger.info(f"Evaluating best model {best_model_name} on test set...")
    y_pred = best_estimator.predict(X_test_proc)
    y_prob = best_estimator.predict_proba(X_test_proc)[:, 1] if hasattr(best_estimator, "predict_proba") else y_pred

    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred)
    test_rec = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "run_id": run_id,
        "best_model_name": best_model_name,
        "test_accuracy": round(test_acc, 4),
        "test_precision": round(test_prec, 4),
        "test_recall": round(test_rec, 4),
        "test_f1_score": round(test_f1, 4),
        "test_roc_auc": round(test_auc, 4),
        "best_params": best_params,
        "processed_features": all_processed_cols,
        "leaderboard": leaderboard_data
    }

    save_pickle(preprocessor, PREPROCESSOR_PATH)
    save_pickle(best_estimator, MODEL_PATH)
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Saved model pipeline and performance metrics.")

    try:
        db = SessionLocal()
        db_log = TrainingLog(
            run_id=run_id,
            model_name=best_model_name,
            accuracy=test_acc,
            precision=test_prec,
            recall=test_rec,
            f1_score=test_f1,
            roc_auc=test_auc,
            hyperparameters=json.dumps(best_params),
            trained_at=datetime.utcnow()
        )
        db.add(db_log)
        db.commit()
        db.close()
        logger.info("Logged run to database.")
    except Exception as db_err:
        logger.error(f"Failed to log run to database: {db_err}")

    return leaderboard, metrics, preprocessor, best_estimator, (X_train_proc, X_test_proc, y_train, y_test)

if __name__ == "__main__":
    run_id = f"RUN_{uuid.uuid4().hex[:8].upper()}"
    train_and_evaluate_all(run_id)

