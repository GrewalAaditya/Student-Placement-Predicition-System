import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from src.config import NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL
from src.feature_engineering import FeatureEngineer
from src.utils import logger, normalize_dataframe_columns

class OutlierCapper(BaseEstimator, TransformerMixin):
    """Capping outliers using the IQR method."""
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            if pd.api.types.is_numeric_dtype(X_df[col]):
                q1 = X_df[col].quantile(0.25)
                q3 = X_df[col].quantile(0.75)
                iqr = q3 - q1
                self.lower_bounds_[col] = q1 - self.factor * iqr
                self.upper_bounds_[col] = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if col in self.lower_bounds_:
                X_df[col] = np.clip(X_df[col], self.lower_bounds_[col], self.upper_bounds_[col])
        return X_df

class DataCleaner(BaseEstimator, TransformerMixin):
    """Basic data cleaner for strings."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if pd.api.types.is_string_dtype(X_df[col]) or X_df[col].dtype == object:
                X_df[col] = X_df[col].astype(str).str.strip()
        return X_df

def build_preprocessing_pipeline() -> Pipeline:
    """Build and return preprocessing pipeline."""
    engineered_numerical_cols = NUMERICAL_COLS + [
        "Academic_Performance_Index",
        "Employability_Readiness_Score",
        "Skills_Diversity_Index"
    ]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_capper', OutlierCapper(factor=1.5)),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, engineered_numerical_cols),
            ('cat', cat_pipeline, CATEGORICAL_COLS)
        ],
        remainder='drop'
    )

    full_pipeline = Pipeline([
        ('cleaner', DataCleaner()),
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline

def prepare_data(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """Load, split, and clean raw dataset."""
    try:
        logger.info(f"Loading raw data: {filepath}")
        df = pd.read_csv(filepath)
        df = normalize_dataframe_columns(df)
        
        if TARGET_COL not in df.columns:
            raise KeyError(f"Target '{TARGET_COL}' not found in data.")

        before_count = len(df)
        df = df.drop_duplicates()
        after_count = len(df)
        if before_count > after_count:
            logger.info(f"Dropped {before_count - after_count} duplicate rows.")

        # Expected Salary is derived from placement outcome — exclude it from model features.
        X = df.drop(
            columns=[TARGET_COL, "Student_ID", "Expected Salary"],
            errors='ignore'
        )
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Split data: train_size={X_train.shape[0]}, test_size={X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        raise e

