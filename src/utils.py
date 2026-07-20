import logging
import joblib
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from src.config import LOGS_DIR, CATEGORICAL_COLS, FEATURES, TARGET_COL

COLUMN_ALIASES = {
    "placement status": "Placement Status",
    "student id": "Student_ID",
    "student_id": "Student_ID",
    "gender": "Gender",
    "branch": "Branch",
    "degree": "Degree",
    "english proficiency": "English Proficiency",
    "age": "Age",
    "cgpa": "CGPA",
    "10th percentage": "10th Percentage",
    "10thpercentage": "10th Percentage",
    "tenth percentage": "10th Percentage",
    "12th percentage": "12th Percentage",
    "12thpercentage": "12th Percentage",
    "twelfth percentage": "12th Percentage",
    "backlogs": "Backlogs",
    "attendance": "Attendance",
    "technical skills": "Technical Skills",
    "programming score": "Programming Score",
    "aptitude score": "Aptitude Score",
    "communication skills": "Communication Skills",
    "soft skills": "Soft Skills",
    "projects completed": "Projects Completed",
    "internships": "Internships",
    "hackathons": "Hackathons",
    "certifications": "Certifications",
    "workshops": "Workshops",
    "leadership score": "Leadership Score",
    "resume score": "Resume Score",
    "interview score": "Interview Score",
    "expected salary": "Expected Salary",
    "placement probability": "Placement_Probability_%",
    "predicted placement status": "Predicted_Placement_Status",
    "placement readiness score": "Placement_Readiness_Score"
}

DEFAULT_FEATURE_VALUES: Dict[str, Any] = {
    "Age": 21,
    "CGPA": 7.0,
    "10th Percentage": 60.0,
    "12th Percentage": 60.0,
    "Backlogs": 0,
    "Attendance": 75.0,
    "Technical Skills": 60.0,
    "Programming Score": 60.0,
    "Aptitude Score": 60.0,
    "Communication Skills": 60.0,
    "Soft Skills": 60.0,
    "Projects Completed": 1,
    "Internships": 0,
    "Hackathons": 0,
    "Certifications": 0,
    "Workshops": 0,
    "Leadership Score": 60.0,
    "Resume Score": 60.0,
    "Interview Score": 60.0,
    "Expected Salary": 350000.0,
    "Gender": "Male",
    "Branch": "Computer Science",
    "Degree": "B.Tech",
    "English Proficiency": "Medium"
}


def normalize_column_name(column_name: Any) -> Any:
    if not isinstance(column_name, str):
        return column_name
    column_name = column_name.lstrip('\ufeff')
    normalized = column_name.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return COLUMN_ALIASES.get(normalized, column_name.strip())


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]
    return df


def normalize_input_dict(input_data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for key, value in input_data.items():
        normalized_key = normalize_column_name(key)
        normalized[normalized_key] = value
    return normalized


def ensure_feature_columns(df: pd.DataFrame, fill_missing: bool = True) -> pd.DataFrame:
    df = normalize_dataframe_columns(df.copy())
    for feature in FEATURES:
        if feature not in df.columns:
            if fill_missing:
                df[feature] = DEFAULT_FEATURE_VALUES.get(feature, 0 if feature not in CATEGORICAL_COLS else "Male")
            else:
                raise KeyError(f"Missing required feature column: {feature}")
    return df[FEATURES].copy()


def ensure_target_column(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dataframe_columns(df.copy())
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found.")
    return df


def fill_missing_dataframe_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = normalize_dataframe_columns(df.copy())
    for col in columns:
        if col not in df.columns:
            df[col] = DEFAULT_FEATURE_VALUES.get(col, 0 if col not in CATEGORICAL_COLS else "Male")
    return df


def pandas_read_csv(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    return normalize_dataframe_columns(df)


def setup_logging(name: str = "system") -> logging.Logger:
    """Set up logger with console and file handlers."""
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    
    if log.handlers:
        return log

    log_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    log.addHandler(console_handler)

    log_file = LOGS_DIR / f"{name}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(log_format)
    log.addHandler(file_handler)

    return log

logger = setup_logging("placement_system")

def save_pickle(obj: Any, filepath: Path) -> None:
    """Save object to pickle file using joblib."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, filepath)
        logger.info(f"Saved object to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save object to {filepath}: {e}")
        raise e

def load_pickle(filepath: Path) -> Any:
    """Load serialized object from file."""
    try:
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} does not exist.")
        obj = joblib.load(filepath)
        logger.info(f"Loaded object from {filepath}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load object from {filepath}: {e}")
        raise e

