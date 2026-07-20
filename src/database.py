import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from src.config import DATABASE_URL
from src.utils import logger

# DB connection setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="user") # 'admin' or 'user'
    created_at = Column(DateTime, default=datetime.utcnow)

class UploadedDataset(Base):
    __tablename__ = "uploaded_datasets"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    record_count = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

class TrainingLog(Base):
    __tablename__ = "training_logs"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, index=True, nullable=False)
    model_name = Column(String, nullable=False)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=True)
    hyperparameters = Column(Text, nullable=True) # JSON dump
    trained_at = Column(DateTime, default=datetime.utcnow)

class PredictionRecord(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, nullable=True)
    cgpa = Column(Float, nullable=False)
    branch = Column(String, nullable=False)
    attendance = Column(Float, nullable=False)
    programming_score = Column(Float, nullable=False)
    aptitude_score = Column(Float, nullable=False)
    interview_score = Column(Float, nullable=False)
    resume_score = Column(Float, nullable=False)
    internships = Column(Integer, default=0)
    projects_completed = Column(Integer, default=0)
    certifications = Column(Integer, default=0)
    predicted_placed = Column(Boolean, nullable=False)
    placement_probability = Column(Float, nullable=False)
    readiness_score = Column(Float, nullable=False)
    expected_salary = Column(Float, nullable=True)
    job_category = Column(String, nullable=True)
    input_features_json = Column(Text, nullable=False)
    predicted_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    """Create all database tables if they do not exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized.")
    except Exception as e:
        logger.critical(f"Failed to init database: {e}")
        raise e

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

