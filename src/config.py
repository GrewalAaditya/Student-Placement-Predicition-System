import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATABASE_DIR = BASE_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"

for folder in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, DATABASE_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_DIR}/placement.db")

TARGET_COL = "Placement Status"

CATEGORICAL_COLS = [
    "Gender",
    "Branch",
    "Degree",
    "English Proficiency"
]

NUMERICAL_COLS = [
    "Age",
    "CGPA",
    "10th Percentage",
    "12th Percentage",
    "Backlogs",
    "Attendance",
    "Technical Skills",
    "Programming Score",
    "Aptitude Score",
    "Communication Skills",
    "Soft Skills",
    "Projects Completed",
    "Internships",
    "Hackathons",
    "Certifications",
    "Workshops",
    "Leadership Score",
    "Resume Score",
    "Interview Score"
    # NOTE: 'Expected Salary' intentionally excluded — it is derived from the
    # placement label in the synthetic data generator and causes data leakage.
]

FEATURES = CATEGORICAL_COLS + NUMERICAL_COLS

MODEL_PATH = MODELS_DIR / "placement_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"

HYPERPARAMETER_GRIDS = {
    "Random Forest": {
        "n_estimators": [100],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5]
    },
    "XGBoost": {
        "n_estimators": [100],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1]
    },
    "LightGBM": {
        "n_estimators": [100],
        "max_depth": [-1, 5],
        "learning_rate": [0.05, 0.1]
    },
    "CatBoost": {
        "iterations": [100],
        "depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "verbose": [0]
    }
}

