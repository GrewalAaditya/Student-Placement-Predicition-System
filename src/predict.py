import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from src.config import MODEL_PATH, PREPROCESSOR_PATH, FEATURES
from src.utils import logger, load_pickle, normalize_dataframe_columns, ensure_feature_columns, normalize_input_dict, DEFAULT_FEATURE_VALUES

class PlacementPredictor:
    """Predictor engine for individual and batch student placement queries."""
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Load trained model and preprocessing pipelines."""
        try:
            if MODEL_PATH.exists() and PREPROCESSOR_PATH.exists():
                self.model = load_pickle(MODEL_PATH)
                self.preprocessor = load_pickle(PREPROCESSOR_PATH)
                logger.info("Loaded model and preprocessor.")
            else:
                logger.warning("Model and preprocessor files not found. Run training first.")
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")

    def is_ready(self) -> bool:
        """Check if models are loaded."""
        return self.model is not None and self.preprocessor is not None

    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run prediction and analytics for a single student profile."""
        if not self.is_ready():
            self.load_artifacts()
            if not self.is_ready():
                raise RuntimeError("Predictor not ready. Missing model or preprocessor files.")

        normalized_input = normalize_input_dict(input_data)

        # Inject Expected Salary default if the saved preprocessor was trained with it
        # (avoids "columns are missing" errors from older model artifacts).
        if "Expected Salary" not in normalized_input:
            normalized_input["Expected Salary"] = DEFAULT_FEATURE_VALUES.get("Expected Salary", 350000.0)

        df_input = pd.DataFrame([normalized_input])
        df_input = ensure_feature_columns(df_input)

        # Try preprocessing; if Expected Salary is still missing (legacy pipeline),
        # add it back into the DataFrame before passing to the preprocessor.
        try:
            df_proc = self.preprocessor.transform(df_input)
        except Exception as preproc_err:
            missing_msg = str(preproc_err)
            if "Expected Salary" in missing_msg or "Expected_Salary" in missing_msg:
                df_input["Expected Salary"] = normalized_input.get("Expected Salary", 350000.0)
                df_proc = self.preprocessor.transform(df_input)
            else:
                raise preproc_err
        
        prob = float(self.model.predict_proba(df_proc)[0, 1])
        prediction = int(self.model.predict(df_proc)[0])
        
        confidence = prob if prediction == 1 else (1.0 - prob)
        confidence_pct = round(confidence * 100, 2)
        prob_pct = round(prob * 100, 2)
        
        readiness_score = self.calculate_readiness_score(normalized_input)
        salary_range = self.estimate_salary_range(prediction, normalized_input)
        job_category, recommended_companies = self.determine_career_path(normalized_input)
        eligibility = self.check_eligibility(normalized_input)
        suggestions = self.generate_suggestions(normalized_input)

        processed_array = df_proc[0] if isinstance(df_proc, np.ndarray) else df_proc.values[0]

        return {
            "placed": prediction == 1,
            "probability": prob_pct,
            "confidence": confidence_pct,
            "readiness_score": round(readiness_score, 2),
            "expected_salary_range": salary_range,
            "job_category": job_category,
            "recommended_companies": recommended_companies,
            "eligible": eligibility["eligible"],
            "eligibility_remarks": eligibility["remarks"],
            "suggestions": suggestions,
            "processed_input": processed_array.tolist()
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run predictions for a batch DataFrame of students."""
        if not self.is_ready():
            self.load_artifacts()
            if not self.is_ready():
                raise RuntimeError("Predictor not ready. Missing model or preprocessor files.")
        
        df_clean = normalize_dataframe_columns(df.copy())
        df_clean = ensure_feature_columns(df_clean)
        df_proc = self.preprocessor.transform(df_clean)
        
        probs = self.model.predict_proba(df_proc)[:, 1]
        preds = self.model.predict(df_proc)
        
        df_clean["Placement_Probability_%"] = np.round(probs * 100, 2)
        df_clean["Predicted_Placement_Status"] = np.where(preds == 1, "Placed", "Not Placed")
        
        readiness_scores = []
        for _, row in df_clean.iterrows():
            row_dict = normalize_input_dict(row.to_dict())
            readiness_scores.append(round(self.calculate_readiness_score(row_dict), 2))
        
        df_clean["Placement_Readiness_Score"] = readiness_scores
        return df_clean

    def calculate_readiness_score(self, data: Dict[str, Any]) -> float:
        """Calculate readiness score index (0 to 100) based on student input."""
        try:
            prog = (float(data.get("Programming Score", 50)) / 100.0) * 25
            apt = (float(data.get("Aptitude Score", 50)) / 100.0) * 15
            interview = (float(data.get("Interview Score", 50)) / 100.0) * 20
            resume = (float(data.get("Resume Score", 50)) / 100.0) * 15
            internships = min(float(data.get("Internships", 0)) * 10, 20)
            projects = min(float(data.get("Projects Completed", 0)) * 2.5, 5)
            
            return prog + apt + interview + resume + internships + projects
        except Exception:
            return 50.0

    def estimate_salary_range(self, prediction: int, data: Dict[str, Any]) -> str:
        """Estimate placement salary ranges in LPA based on student credentials."""
        cgpa = float(data.get("CGPA", 7.0))
        prog = float(data.get("Programming Score", 50))
        intern = int(data.get("Internships", 0))
        projects = int(data.get("Projects Completed", 0))
        
        if prediction == 1:
            min_sal = 3.6
            min_sal += (cgpa - 6) * 0.9 if cgpa > 6 else 0
            min_sal += (prog - 50) * 0.06 if prog > 50 else 0
            min_sal += intern * 1.2
            min_sal += projects * 0.4
            
            max_sal = min_sal * 1.25
            return f"{round(min_sal, 2)} - {round(max_sal, 2)} LPA"
        else:
            min_sal = 1.8
            min_sal += (cgpa - 5) * 0.3 if cgpa > 5 else 0
            max_sal = min_sal * 1.2
            return f"{round(min_sal, 2)} - {round(max_sal, 2)} LPA"

    def determine_career_path(self, data: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Determine potential roles and target company categories based on student branch and scores."""
        branch = data.get("Branch", "Computer Science")
        prog = float(data.get("Programming Score", 50))
        apt = float(data.get("Aptitude Score", 50))
        cgpa = float(data.get("CGPA", 7.0))
        intern = int(data.get("Internships", 0))

        tier_1 = ["Amazon", "Microsoft", "Google", "Adobe", "Directi", "Tier-1 Tech Startups"]
        tier_2 = ["Accenture", "Capgemini", "TCS Digital", "Cognizant GenC Next", "Wipro Turbo", "Infosys Power Programmer"]
        tier_3 = ["TCS Ninja", "Infosys", "Cognizant GenC", "Tech Mahindra", "Wipro Elite"]
        core_ec = ["Bosch", "Qualcomm", "Intel", "Siemens", "Texas Instruments"]
        core_mech = ["Tata Motors", "L&T", "Ashok Leyland", "Reliance Industries", "Godrej"]

        if "Computer Science" in branch or "Information Technology" in branch:
            if prog > 78 and cgpa >= 8.5 and intern >= 1:
                return "Software Development Engineer (SDE)", tier_1
            elif prog > 60:
                return "Full Stack Developer / DevOps Engineer", tier_2
            else:
                return "System Engineer / QA Analyst", tier_3
                
        elif "Electronics" in branch or "Communication" in branch:
            if prog > 70:
                return "Embedded Software Engineer / IoT Developer", core_ec + ["TCS Digital"]
            else:
                return "Telecom Engineer / Network Operations", core_ec + tier_3
                
        elif "Mechanical" in branch:
            if cgpa > 7.5:
                return "CAD Design Engineer / R&D Specialist", core_mech
            else:
                return "Production Engineer / Quality Inspector", core_mech + ["TCS Ninja"]
                
        elif "Civil" in branch:
            return "Structural Design Engineer / Construction Site Manager", ["L&T", "DLF", "Tata Projects"]
            
        else:
            if apt > 70:
                return "Business Operations Analyst", tier_2
            else:
                return "Technical Support Executive", tier_3

    def check_eligibility(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check basic academic eligibility criteria."""
        backlogs = int(data.get("Backlogs", 0))
        attendance = float(data.get("Attendance", 100))
        
        remarks = []
        eligible = True
        
        if backlogs > 0:
            eligible = False
            remarks.append(f"Blocked by {backlogs} active backlog(s). Zero active backlogs are required for most companies.")
            
        if attendance < 75.0:
            eligible = False
            remarks.append(f"Low attendance ({attendance}%). Minimum 75% attendance is required to sit for campus drives.")
            
        if eligible:
            remarks.append("Eligible! Academic standing meets baseline criteria.")
            
        return {"eligible": eligible, "remarks": remarks}

    def generate_suggestions(self, data: Dict[str, Any]) -> List[str]:
        """Generate practical recommendations based on student weaknesses."""
        cgpa = float(data.get("CGPA", 7.0))
        attendance = float(data.get("Attendance", 100))
        backlogs = int(data.get("Backlogs", 0))
        prog = float(data.get("Programming Score", 50))
        apt = float(data.get("Aptitude Score", 50))
        resume = float(data.get("Resume Score", 50))
        interview = float(data.get("Interview Score", 50))
        comm = float(data.get("Communication Skills", 50))
        projects = int(data.get("Projects Completed", 0))
        internships = int(data.get("Internships", 0))
        certs = int(data.get("Certifications", 0))

        suggestions = []
        
        if backlogs > 0:
            suggestions.append("Clear outstanding backlogs immediately. Most companies require zero active backlogs.")
            
        if attendance < 75.0:
            suggestions.append("Maintain attendance above 75% to meet baseline institutional criteria.")
            
        if cgpa < 7.0:
            suggestions.append("Work on raising your CGPA. Many companies set a minimum cut-off of 7.0.")
            
        if prog < 65:
            suggestions.append("Improve your programming score by practicing coding questions on LeetCode or HackerRank to clear technical rounds.")
            
        if apt < 65:
            suggestions.append("Prepare for aptitude tests and logical reasoning rounds.")
            
        if internships == 0:
            suggestions.append("Try to get an internship or industry project experience.")
            
        if projects < 2:
            suggestions.append("Build at least 2 hands-on projects and share them on GitHub.")
            
        if certs == 0:
            suggestions.append("Consider earning relevant certifications in your field.")
            
        if resume < 70:
            suggestions.append("Format your resume to be clean and simple, highlighting key skills and project details.")
            
        if interview < 70:
            suggestions.append("Practice mock interviews to explain your projects and technical skills confidently.")
            
        if comm < 70:
            suggestions.append("Work on soft skills and presentation skills.")
            
        if not suggestions:
            suggestions.append("Your profile looks solid. Focus on core technical skills and mock interviews.")
            
        return suggestions

