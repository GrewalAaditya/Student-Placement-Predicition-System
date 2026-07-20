import pytest
import pandas as pd
import numpy as np
from src.preprocessing import OutlierCapper, DataCleaner, build_preprocessing_pipeline
from src.config import NUMERICAL_COLS, CATEGORICAL_COLS

def test_outlier_capper():
    """Validates that the outlier capper correctly clips numeric values outside IQR."""
    # Data with a clear outlier (100.0)
    data = pd.DataFrame({"score": [10.0, 12.0, 11.0, 9.0, 100.0]})
    capper = OutlierCapper(factor=1.5)
    
    # Fit and transform
    capper.fit(data)
    capped_df = capper.transform(data)
    
    # Assert outlier was capped
    assert capped_df["score"].max() < 100.0
    assert capped_df["score"].min() == 9.0

def test_data_cleaner():
    """Validates that DataCleaner strips whitespace from categorical inputs."""
    data = pd.DataFrame({"gender": [" Male ", "Female  ", "  Male  "]})
    cleaner = DataCleaner()
    cleaned_df = cleaner.transform(data)
    
    # Assert spacing stripped
    assert cleaned_df["gender"].tolist() == ["Male", "Female", "Male"]

def test_preprocessing_pipeline():
    """Validates the full pipeline outputs correct shapes and structures."""
    # Generate mock student profile
    mock_data = {
        "Gender": ["Male"],
        "Branch": ["Computer Science"],
        "Degree": ["B.Tech"],
        "English Proficiency": ["Medium"],
        "Age": [21],
        "CGPA": [8.5],
        "10th Percentage": [88.0],
        "12th Percentage": [84.0],
        "Backlogs": [0],
        "Attendance": [90.0],
        "Technical Skills": [75.0],
        "Programming Score": [80.0],
        "Aptitude Score": [70.0],
        "Communication Skills": [75.0],
        "Soft Skills": [75.0],
        "Projects Completed": [2],
        "Internships": [1],
        "Hackathons": [0],
        "Certifications": [1],
        "Workshops": [1],
        "Leadership Score": [65.0],
        "Resume Score": [80.0],
        "Interview Score": [75.0],
    }
    
    df = pd.DataFrame(mock_data)
    pipeline = build_preprocessing_pipeline()
    
    # Fit and transform
    proc_arr = pipeline.fit_transform(df)
    
    # Check that output is numeric matrix and shape is correct
    assert isinstance(proc_arr, np.ndarray)
    assert proc_arr.shape[0] == 1
    # Check columns count: 24 (numerical + engineered numerical) + categorical OHE features
    assert proc_arr.shape[1] > 20
