import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils import logger

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer for engineering student placement features."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        try:
            # Academic Performance Index
            cgpa_100 = X_df["CGPA"] * 10.0
            X_df["Academic_Performance_Index"] = (
                cgpa_100 * 0.50 + 
                X_df["10th Percentage"] * 0.25 + 
                X_df["12th Percentage"] * 0.25
            )

            # Employability Readiness Score (out of 100)
            prog_contrib = (X_df["Programming Score"] / 100.0) * 25
            apt_contrib = (X_df["Aptitude Score"] / 100.0) * 15
            int_contrib = (X_df["Interview Score"] / 100.0) * 20
            res_contrib = (X_df["Resume Score"] / 100.0) * 15
            intern_contrib = np.clip(X_df["Internships"] * 10, 0, 20)
            proj_contrib = np.clip(X_df["Projects Completed"] * 2.5, 0, 5)

            X_df["Employability_Readiness_Score"] = (
                prog_contrib + apt_contrib + int_contrib + res_contrib + 
                intern_contrib + proj_contrib
            )

            # Skills Diversity Index
            X_df["Skills_Diversity_Index"] = (
                X_df["Certifications"] + 
                X_df["Workshops"] + 
                X_df["Hackathons"]
            )
            
            return X_df

        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            raise e

