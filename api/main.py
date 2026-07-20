import os
import uuid
import shutil
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.config import (
    DATA_RAW_DIR, MODEL_PATH, PREPROCESSOR_PATH, REPORTS_DIR, TARGET_COL
)
from src.database import get_db, init_db, UploadedDataset, TrainingLog, PredictionRecord
from src.train import train_and_evaluate_all
from src.predict import PlacementPredictor
from src.utils import logger

app = FastAPI(
    title="Placement Portal API",
    description="API backend for managing student records, predictions, and models.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()

predictor = PlacementPredictor()

class StudentFeatures(BaseModel):
    Student_ID: Optional[str] = "STU1001"
    Gender: str = Field(default="Male")
    Age: int = Field(default=21, ge=18, le=35)
    Branch: str = Field(default="Computer Science")
    Degree: str = Field(default="B.Tech")
    CGPA: float = Field(default=8.2, ge=4.0, le=10.0)
    tenth_percentage: float = Field(default=85.0, alias="10th Percentage", ge=40.0, le=100.0)
    twelfth_percentage: float = Field(default=82.0, alias="12th Percentage", ge=40.0, le=100.0)
    Backlogs: int = Field(default=0, ge=0)
    Attendance: float = Field(default=85.0, ge=0.0, le=100.0)
    Technical_Skills: float = Field(default=75.0, alias="Technical Skills", ge=0.0, le=100.0)
    Programming_Score: float = Field(default=80.0, alias="Programming Score", ge=0.0, le=100.0)
    Aptitude_Score: float = Field(default=70.0, alias="Aptitude Score", ge=0.0, le=100.0)
    Communication_Skills: float = Field(default=75.0, alias="Communication Skills", ge=0.0, le=100.0)
    Soft_Skills: float = Field(default=70.0, alias="Soft Skills", ge=0.0, le=100.0)
    Projects_Completed: int = Field(default=2, alias="Projects Completed", ge=0)
    Internships: int = Field(default=1, ge=0)
    Hackathons: int = Field(default=1, ge=0)
    Certifications: int = Field(default=1, ge=0)
    Workshops: int = Field(default=1, ge=0)
    Leadership_Score: float = Field(default=65.0, alias="Leadership Score", ge=0.0, le=100.0)
    Resume_Score: float = Field(default=75.0, alias="Resume Score", ge=0.0, le=100.0)
    Interview_Score: float = Field(default=70.0, alias="Interview Score", ge=0.0, le=100.0)
    English_Proficiency: str = Field(default="Medium", alias="English Proficiency")
    Expected_Salary: float = Field(default=450000.0, alias="Expected Salary", ge=0.0)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Student_ID": "STU9999",
                "Gender": "Male",
                "Age": 22,
                "Branch": "Computer Science",
                "Degree": "B.Tech",
                "CGPA": 8.5,
                "10th Percentage": 88.0,
                "12th Percentage": 84.5,
                "Backlogs": 0,
                "Attendance": 90.0,
                "Technical Skills": 78.0,
                "Programming Score": 82.0,
                "Aptitude Score": 75.0,
                "Communication Skills": 80.0,
                "Soft Skills": 78.0,
                "Projects Completed": 2,
                "Internships": 1,
                "Hackathons": 1,
                "Certifications": 2,
                "Workshops": 1,
                "Leadership Score": 70.0,
                "Resume Score": 80.0,
                "Interview Score": 75.0,
                "English Proficiency": "High",
                "Expected Salary": 500000.0
            }
        }

@app.get("/")
def read_root():
    return {"message": "Placement Portal API is online"}

@app.post("/upload")
def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload new student CSV dataset."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        file_id = uuid.uuid4().hex[:6].upper()
        saved_filename = f"upload_{file_id}_{file.filename}"
        filepath = DATA_RAW_DIR / saved_filename
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        df = pd.read_csv(filepath)
        row_count = len(df)
        
        db_dataset = UploadedDataset(
            filename=file.filename,
            filepath=str(filepath),
            record_count=row_count
        )
        db.add(db_dataset)
        db.commit()
        db.refresh(db_dataset)
        
        return {
            "message": "Dataset uploaded successfully",
            "id": db_dataset.id,
            "filename": db_dataset.filename,
            "records": row_count
        }
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")

def run_training_task(run_id: str):
    try:
        train_and_evaluate_all(run_id)
        predictor.load_artifacts()
        logger.info(f"Training run {run_id} completed.")
    except Exception as e:
        logger.error(f"Training run {run_id} failed: {e}")

@app.post("/train")
def train_models(background_tasks: BackgroundTasks):
    """Trigger training pipeline in background."""
    run_id = f"RUN_{uuid.uuid4().hex[:8].upper()}"
    logger.info(f"Starting training run {run_id}")
    
    raw_csv = DATA_RAW_DIR / "placement_dataset.csv"
    if not raw_csv.exists():
        raise HTTPException(
            status_code=404, 
            detail="Raw dataset 'placement_dataset.csv' not found. Please generate or upload a dataset first."
        )
        
    background_tasks.add_task(run_training_task, run_id)
    return {
        "status": "Training initiated in background",
        "run_id": run_id
    }

@app.post("/predict")
def predict_student(student: StudentFeatures, db: Session = Depends(get_db)):
    """Predict placement status for a single student."""
    if not predictor.is_ready():
        predictor.load_artifacts()
        if not predictor.is_ready():
            raise HTTPException(
                status_code=503, 
                detail="Model artifacts are not loaded. Run /train endpoint to build models."
            )
            
    try:
        student_dict = student.model_dump(by_alias=True)
        result = predictor.predict_single(student_dict)
        
        db_pred = PredictionRecord(
            student_id=student.Student_ID,
            cgpa=student.CGPA,
            branch=student.Branch,
            attendance=student.Attendance,
            programming_score=student.Programming_Score,
            aptitude_score=student.Aptitude_Score,
            interview_score=student.Interview_Score,
            resume_score=student.Resume_Score,
            internships=student.Internships,
            projects_completed=student.Projects_Completed,
            certifications=student.Certifications,
            predicted_placed=result["placed"],
            placement_probability=result["probability"],
            readiness_score=result["readiness_score"],
            expected_salary=float(student.Expected_Salary),
            job_category=result["job_category"],
            input_features_json=json.dumps(student_dict)
        )
        
        db.add(db_pred)
        db.commit()
        db.refresh(db_pred)
        
        result["db_id"] = db_pred.id
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/predict/batch")
def predict_batch(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Run predictions for batch student CSV."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    if not predictor.is_ready():
        predictor.load_artifacts()
        if not predictor.is_ready():
            raise HTTPException(status_code=503, detail="Predictor is not ready. Model is not trained.")
            
    try:
        temp_filepath = DATA_RAW_DIR / f"temp_{uuid.uuid4().hex[:6]}_{file.filename}"
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        df = pd.read_csv(temp_filepath)
        df_pred = predictor.predict_batch(df)
        
        out_filename = f"predicted_{uuid.uuid4().hex[:6]}_{file.filename}"
        out_filepath = DATA_RAW_DIR / out_filename
        df_pred.to_csv(out_filepath, index=False)
        
        os.remove(temp_filepath)
        
        for _, row in df_pred.iterrows():
            try:
                row_dict = row.to_dict()
                db_pred = PredictionRecord(
                    student_id=str(row_dict.get("Student_ID", f"BATCH_{uuid.uuid4().hex[:4]}")),
                    cgpa=float(row_dict.get("CGPA", 0)),
                    branch=str(row_dict.get("Branch", "Unknown")),
                    attendance=float(row_dict.get("Attendance", 0)),
                    programming_score=float(row_dict.get("Programming Score", 0)),
                    aptitude_score=float(row_dict.get("Aptitude Score", 0)),
                    interview_score=float(row_dict.get("Interview Score", 0)),
                    resume_score=float(row_dict.get("Resume Score", 0)),
                    internships=int(row_dict.get("Internships", 0)),
                    projects_completed=int(row_dict.get("Projects Completed", 0)),
                    certifications=int(row_dict.get("Certifications", 0)),
                    predicted_placed=(row_dict.get("Predicted_Placement_Status") == "Placed"),
                    placement_probability=float(row_dict.get("Placement_Probability_%", 0.0)),
                    readiness_score=float(row_dict.get("Placement_Readiness_Score", 0.0)),
                    expected_salary=float(row_dict.get("Expected Salary", 0.0)),
                    job_category=None,
                    input_features_json=json.dumps(row_dict)
                )
                db.add(db_pred)
            except Exception:
                continue
                
        db.commit()
        
        return {
            "message": "Batch predictions completed successfully",
            "prediction_file": out_filename,
            "placed_count": int((df_pred["Predicted_Placement_Status"] == "Placed").sum()),
            "total_count": len(df_pred)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

@app.get("/models")
def get_models_info(db: Session = Depends(get_db)):
    """Retrieve history logs of trained models."""
    logs = db.query(TrainingLog).order_by(TrainingLog.trained_at.desc()).all()
    
    active_model = None
    if predictor.is_ready():
        active_model = {
            "model_name": predictor.model.__class__.__name__,
            "is_loaded": True
        }
        
    return {
        "active_model": active_model,
        "history": [
            {
                "run_id": log.run_id,
                "model_name": log.model_name,
                "accuracy": log.accuracy,
                "precision": log.precision,
                "recall": log.recall,
                "f1_score": log.f1_score,
                "roc_auc": log.roc_auc,
                "hyperparameters": json.loads(log.hyperparameters) if log.hyperparameters else {},
                "trained_at": log.trained_at
            }
            for log in logs
        ]
    }

@app.get("/reports/{filename}")
def get_report_file(filename: str):
    """Download reports or charts."""
    file_path = REPORTS_DIR / filename
    if not file_path.exists():
        file_path = DATA_RAW_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File {filename} not found.")
            
    return FileResponse(path=str(file_path), filename=filename)

