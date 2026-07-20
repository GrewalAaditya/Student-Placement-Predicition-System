import numpy as np
import pandas as pd
from pathlib import Path
from src.config import DATA_RAW_DIR, CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL
from src.database import init_db
from src.utils import logger

def generate_student_data(num_records: int = 5500, random_seed: int = 42) -> pd.DataFrame:
    """Generate mock student placement dataset."""
    logger.info(f"Generating mock dataset: count={num_records}, seed={random_seed}")
    np.random.seed(random_seed)

    # Base details
    student_ids = [f"STU{1000 + i}" for i in range(num_records)]
    genders = np.random.choice(["Male", "Female"], size=num_records, p=[0.58, 0.42])
    ages = np.random.randint(20, 26, size=num_records)
    branches = np.random.choice(
        ["Computer Science", "Information Technology", "Electronics & Communication", 
         "Electrical Engineering", "Mechanical Engineering", "Civil Engineering"], 
        size=num_records, 
        p=[0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
    )
    degrees = np.random.choice(["B.Tech", "M.Tech", "MCA", "BCA"], size=num_records, p=[0.70, 0.10, 0.15, 0.05])
    
    # Academic indicators
    cgpa = np.random.normal(7.8, 0.9, size=num_records)
    cgpa = np.clip(cgpa, 4.0, 10.0)
    cgpa = np.round(cgpa, 2)
    
    tenth_percentage = np.random.normal(80, 8, size=num_records)
    tenth_percentage = np.clip(tenth_percentage, 50.0, 100.0)
    tenth_percentage = np.round(tenth_percentage, 2)
    
    twelfth_percentage = np.random.normal(78, 9, size=num_records)
    twelfth_percentage = np.clip(twelfth_percentage, 50.0, 100.0)
    twelfth_percentage = np.round(twelfth_percentage, 2)
    
    backlogs = np.random.choice([0, 1, 2, 3, 4], size=num_records, p=[0.82, 0.12, 0.04, 0.015, 0.005])
    
    attendance = np.random.normal(83, 7, size=num_records)
    attendance = np.clip(attendance, 50.0, 100.0)
    attendance = np.round(attendance, 2)

    # Skills and scores
    programming_score = np.random.normal(68, 12, size=num_records)
    for i, branch in enumerate(branches):
        if branch in ["Computer Science", "Information Technology"]:
            programming_score[i] += np.random.uniform(5, 12)
    programming_score = np.clip(programming_score, 30.0, 100.0)
    programming_score = np.round(programming_score, 2)

    technical_skills = np.random.normal(70, 10, size=num_records)
    technical_skills = np.clip(technical_skills, 30.0, 100.0)
    technical_skills = np.round(technical_skills, 2)

    aptitude_score = np.random.normal(65, 12, size=num_records)
    aptitude_score = np.clip(aptitude_score, 30.0, 100.0)
    aptitude_score = np.round(aptitude_score, 2)

    communication_skills = np.random.normal(72, 10, size=num_records)
    communication_skills = np.clip(communication_skills, 30.0, 100.0)
    communication_skills = np.round(communication_skills, 2)

    soft_skills = np.random.normal(70, 10, size=num_records)
    soft_skills = np.clip(soft_skills, 30.0, 100.0)
    soft_skills = np.round(soft_skills, 2)

    leadership_score = np.random.normal(64, 12, size=num_records)
    leadership_score = np.clip(leadership_score, 30.0, 100.0)
    leadership_score = np.round(leadership_score, 2)

    interview_score = np.random.normal(68, 11, size=num_records)
    interview_score = np.clip(interview_score, 30.0, 100.0)
    interview_score = np.round(interview_score, 2)

    english_proficiency = np.random.choice(["High", "Medium", "Low"], size=num_records, p=[0.55, 0.35, 0.10])

    # Extracurriculars
    projects = np.random.choice([0, 1, 2, 3, 4], size=num_records, p=[0.10, 0.40, 0.35, 0.10, 0.05])
    internships = np.random.choice([0, 1, 2, 3], size=num_records, p=[0.60, 0.28, 0.10, 0.02])
    hackathons = np.random.choice([0, 1, 2, 3], size=num_records, p=[0.70, 0.20, 0.08, 0.02])
    certifications = np.random.choice([0, 1, 2, 3, 4], size=num_records, p=[0.30, 0.40, 0.20, 0.08, 0.02])
    workshops = np.random.choice([0, 1, 2, 3], size=num_records, p=[0.40, 0.35, 0.20, 0.05])

    resume_score = (
        (cgpa / 10.0) * 20 + 
        (projects * 5) + 
        (internships * 8) + 
        (certifications * 4) + 
        (technical_skills / 100.0) * 30 + 
        np.random.normal(15, 5, size=num_records)
    )
    resume_score = np.clip(resume_score, 30.0, 100.0)
    resume_score = np.round(resume_score, 2)

    # Target: Placement Status logic
    propensity = (
        ((cgpa - 4) / 6.0) * 0.30 +
        ((programming_score - 30) / 70.0) * 0.15 +
        ((aptitude_score - 30) / 70.0) * 0.10 +
        ((communication_skills - 30) / 70.0) * 0.08 +
        ((interview_score - 30) / 70.0) * 0.12 +
        (internships * 0.08) +
        (projects * 0.04) +
        ((attendance / 100.0) * 0.05) -
        (backlogs * 0.12)
    )
    
    # Penalize poor attendance and multiple backlogs
    for i in range(num_records):
        if attendance[i] < 75.0:
            propensity[i] -= 0.15
        if backlogs[i] > 1:
            propensity[i] -= 0.10

    p_mean = np.mean(propensity)
    p_std = np.std(propensity)
    norm_propensity = (propensity - p_mean) / p_std
    
    placement_prob = 1 / (1 + np.exp(-1.8 * norm_propensity))
    placement_status = np.random.binomial(1, placement_prob)

    # Expected Salary
    expected_salary = np.zeros(num_records)
    for i in range(num_records):
        if placement_status[i] == 1:
            base = 350000
            cgpa_bonus = (cgpa[i] - 6) * 90000 if cgpa[i] > 6 else 0
            skill_bonus = (programming_score[i] - 50) * 6000 if programming_score[i] > 50 else 0
            internship_bonus = internships[i] * 120000
            project_bonus = projects[i] * 40000
            noise = np.random.normal(50000, 30000)
            
            salary = base + cgpa_bonus + skill_bonus + internship_bonus + project_bonus + noise
            expected_salary[i] = np.clip(salary, 320000, 1800000)
        else:
            base = 250000
            cgpa_bonus = (cgpa[i] - 5) * 30000 if cgpa[i] > 5 else 0
            noise = np.random.normal(20000, 15000)
            salary = base + cgpa_bonus + noise
            expected_salary[i] = np.clip(salary, 180000, 450000)
            
    expected_salary = np.round(expected_salary, -3)

    # Construct DataFrame
    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Gender": genders,
        "Age": ages,
        "Branch": branches,
        "Degree": degrees,
        "CGPA": cgpa,
        "10th Percentage": tenth_percentage,
        "12th Percentage": twelfth_percentage,
        "Backlogs": backlogs,
        "Attendance": attendance,
        "Technical Skills": technical_skills,
        "Programming Score": programming_score,
        "Aptitude Score": aptitude_score,
        "Communication Skills": communication_skills,
        "Soft Skills": soft_skills,
        "Projects Completed": projects,
        "Internships": internships,
        "Hackathons": hackathons,
        "Certifications": certifications,
        "Workshops": workshops,
        "Leadership Score": leadership_score,
        "Resume Score": resume_score,
        "Interview Score": interview_score,
        "English Proficiency": english_proficiency,
        "Expected Salary": expected_salary,
        "Placement Status": placement_status
    })

    return df

if __name__ == "__main__":
    init_db()
    
    df = generate_student_data(5500)
    raw_path = DATA_RAW_DIR / "placement_dataset.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Dataset saved to {raw_path}")
    print(f"Dataset created with shape: {df.shape}. Saved to {raw_path}")

