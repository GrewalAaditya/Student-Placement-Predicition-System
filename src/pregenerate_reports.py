import json
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import DATA_RAW_DIR, MODELS_DIR, REPORTS_DIR, TARGET_COL
from src.utils import load_pickle, logger
from src.preprocessing import prepare_data
from src.evaluate import generate_all_evaluation_plots
from src.explain import get_explainer, generate_global_shap_plots

def run_pregeneration():
    logger.info("Pregenerating reports and evaluation plots...")
    
    raw_csv = DATA_RAW_DIR / "placement_dataset.csv"
    model_path = MODELS_DIR / "placement_model.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    
    if not (raw_csv.exists() and model_path.exists() and preprocessor_path.exists()):
        logger.error("Missing raw dataset or model/preprocessor pickle files.")
        return
        
    model = load_pickle(model_path)
    preprocessor = load_pickle(preprocessor_path)
    
    X_train, X_test, y_train, y_test = prepare_data(str(raw_csv))
    
    logger.info("Transforming test set...")
    X_test_proc = preprocessor.transform(X_test)
    X_test_proc_arr = X_test_proc.values if hasattr(X_test_proc, "values") else X_test_proc
    
    cat_encoder = preprocessor.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_features_ohe = list(cat_encoder.get_feature_names_out(preprocessor.named_steps['preprocessor'].transformers_[1][2]))
    engineered_numerical_cols = preprocessor.named_steps['preprocessor'].transformers_[0][2]
    feature_names = engineered_numerical_cols + cat_features_ohe

    logger.info("Generating evaluation plots...")
    generate_all_evaluation_plots(model, preprocessor, X_test_proc_arr, y_test, feature_names)
    
    logger.info("Calculating SHAP summary...")
    try:
        X_train_proc = preprocessor.transform(X_train)
        X_train_proc_arr = X_train_proc.values if hasattr(X_train_proc, "values") else X_train_proc
        
        explainer, exp_type = get_explainer(model, X_train_proc_arr)
        
        test_sample = X_test_proc_arr[:150]
        generate_global_shap_plots(explainer, test_sample, feature_names, exp_type)
        logger.info("SHAP summary generated.")
    except Exception as shap_err:
        logger.error(f"SHAP generation failed: {shap_err}")
        
    logger.info("Pregeneration complete.")


if __name__ == "__main__":
    run_pregeneration()
