"""
kaggle_adapter.py
-----------------
Adapts the Kaggle student placement dataset (100k records) to the
project's expected column schema defined in src/config.py.

Kaggle Dataset columns → Project columns mapping:
  student_id                  → Student_ID
  age                         → Age
  gender                      → Gender
  cgpa                        → CGPA
  branch                      → Branch  (CSE→Computer Science, etc.)
  internships_count           → Internships
  projects_count              → Projects Completed
  certifications_count        → Certifications
  coding_skill_score          → Programming Score
  aptitude_score              → Aptitude Score
  communication_skill_score   → Communication Skills
  logical_reasoning_score     → (blended into Technical Skills)
  hackathons_participated     → Hackathons
  mock_interview_score        → Interview Score
  attendance_percentage       → Attendance
  backlogs                    → Backlogs
  leadership_score            → Leadership Score
  extracurricular_score       → (blended into Soft Skills)
  placement_status            → Placement Status  (Placed→1, Not Placed→0)
  salary_package_lpa          → Expected Salary   (LPA × 100000 INR)

Derived (not in Kaggle dataset):
  10th Percentage   ← CGPA-correlated with noise
  12th Percentage   ← CGPA-correlated with noise
  Technical Skills  ← coding_skill_score (60%) + logical_reasoning_score (40%)
  Soft Skills       ← extracurricular_score (50%) + communication_skill (50%)
  English Proficiency ← Binned from communication_skill_score
  Resume Score      ← Computed from CGPA + internships + projects + certs + coding
  Workshops         ← Derived from certifications + hackathons
  Degree            ← Default B.Tech (Kaggle dataset has no degree info)
"""

import numpy as np
import pandas as pd
import shutil
from pathlib import Path

from src.config import DATA_RAW_DIR
from src.utils import logger

# ─── Branch abbreviation → project display name ───────────────────────────────
BRANCH_MAP = {
    "CSE": "Computer Science",
    "IT": "Information Technology",
    "ECE": "Electronics & Communication",
    "EEE": "Electrical Engineering",
    "Mechanical": "Mechanical Engineering",
    "Civil": "Civil Engineering",
    # fallback covers any other value
}

RAW_CSV = DATA_RAW_DIR / "placement_dataset.csv"
BACKUP_CSV = DATA_RAW_DIR / "placement_dataset_kaggle_raw.csv"


def _derive_tenth(cgpa: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Derive 10th % from CGPA (correlated + noise)."""
    base = cgpa * 10 * 0.85
    noise = rng.normal(5, 5, len(cgpa))
    return np.clip(base + noise, 50.0, 100.0).round(2)


def _derive_twelfth(cgpa: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Derive 12th % from CGPA (correlated + noise)."""
    base = cgpa * 10 * 0.82
    noise = rng.normal(3, 6, len(cgpa))
    return np.clip(base + noise, 50.0, 100.0).round(2)


def _derive_resume_score(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Compute resume score from CGPA, projects, internships, certs, coding."""
    score = (
        (df["cgpa"] / 10.0) * 20
        + df["projects_count"] * 5
        + df["internships_count"] * 8
        + df["certifications_count"] * 4
        + (df["coding_skill_score"] / 100.0) * 30
        + rng.normal(10, 5, len(df))
    )
    return np.clip(score, 30.0, 100.0).round(2)


def _derive_workshops(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Derive workshops count from certifications + hackathons."""
    raw = (
        df["certifications_count"] * 0.5
        + df["hackathons_participated"] * 0.3
        + rng.uniform(0, 1.5, len(df))
    )
    return np.clip(raw.astype(int), 0, 3)


def _derive_english(comm: pd.Series) -> pd.Series:
    """Bin communication skill into High / Medium / Low."""
    return pd.cut(
        comm,
        bins=[0, 45, 70, 100],
        labels=["Low", "Medium", "High"],
        right=True,
    ).astype(str)


def _convert_salary(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """
    Convert salary from LPA to INR.
    Placed students: salary_package_lpa × 100,000 clipped to [320k, 1.8M]
    Not-placed students (salary == 0): generate realistic expected salary.
    """
    salary_inr = df["salary_package_lpa"] * 100_000

    not_placed_mask = salary_inr == 0
    n_np = not_placed_mask.sum()

    if n_np > 0:
        cgpa_np = df.loc[not_placed_mask, "cgpa"]
        generated = (
            250_000
            + np.clip(cgpa_np - 5, 0, None) * 30_000
            + rng.normal(20_000, 15_000, n_np)
        )
        salary_inr.loc[not_placed_mask] = np.clip(generated, 180_000, 450_000)

    # Cap placed salaries
    placed_mask = ~not_placed_mask
    salary_inr.loc[placed_mask] = np.clip(
        salary_inr.loc[placed_mask], 320_000, 1_800_000
    )

    return np.round(salary_inr, -3)


def adapt_kaggle_dataset(
    input_path: Path = RAW_CSV,
    output_path: Path = RAW_CSV,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Read the Kaggle CSV, map / derive all columns to the project schema,
    and write the result back to output_path.
    Returns the adapted DataFrame.
    """
    logger.info(f"Reading Kaggle dataset from: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {df.shape[0]:,} records with {df.shape[1]} columns.")

    # Backup raw file (only once)
    if not BACKUP_CSV.exists():
        shutil.copy(input_path, BACKUP_CSV)
        logger.info(f"Backup saved → {BACKUP_CSV}")

    rng = np.random.default_rng(random_seed)
    out = pd.DataFrame()

    # ── Identifiers ────────────────────────────────────────────────────────────
    out["Student_ID"] = ["STU" + str(1000 + i) for i in range(len(df))]

    # ── Demographics ───────────────────────────────────────────────────────────
    out["Gender"] = df["gender"]
    out["Age"] = df["age"]
    out["Branch"] = df["branch"].map(BRANCH_MAP).fillna("Computer Science")
    out["Degree"] = "B.Tech"  # Kaggle dataset has no degree column

    # ── Academics ──────────────────────────────────────────────────────────────
    out["CGPA"] = df["cgpa"].round(2)
    out["10th Percentage"] = _derive_tenth(df["cgpa"], rng)
    out["12th Percentage"] = _derive_twelfth(df["cgpa"], rng)
    out["Backlogs"] = df["backlogs"]
    out["Attendance"] = df["attendance_percentage"].round(2)

    # ── Skills ─────────────────────────────────────────────────────────────────
    out["Technical Skills"] = (
        df["coding_skill_score"] * 0.6 + df["logical_reasoning_score"] * 0.4
    ).round(2)
    out["Programming Score"] = df["coding_skill_score"].round(2)
    out["Aptitude Score"] = df["aptitude_score"].round(2)
    out["Communication Skills"] = df["communication_skill_score"].round(2)
    out["Soft Skills"] = (
        df["extracurricular_score"] * 0.5 + df["communication_skill_score"] * 0.5
    ).round(2)
    out["English Proficiency"] = _derive_english(df["communication_skill_score"])

    # ── Experience / Portfolio ─────────────────────────────────────────────────
    out["Projects Completed"] = df["projects_count"]
    out["Internships"] = df["internships_count"]
    out["Hackathons"] = df["hackathons_participated"]
    out["Certifications"] = df["certifications_count"]
    out["Workshops"] = _derive_workshops(df, rng)
    out["Leadership Score"] = df["leadership_score"].round(2)
    out["Resume Score"] = _derive_resume_score(df, rng)
    out["Interview Score"] = df["mock_interview_score"].round(2)

    # ── Target ─────────────────────────────────────────────────────────────────
    out["Expected Salary"] = _convert_salary(df, rng)
    out["Placement Status"] = df["placement_status"].map(
        {"Placed": 1, "Not Placed": 0}
    )

    logger.info(
        f"Adapted dataset: {len(out):,} records | "
        f"Placement rate: {out['Placement Status'].mean()*100:.1f}%"
    )

    out.to_csv(output_path, index=False)
    logger.info(f"Saved adapted dataset → {output_path}")
    return out


if __name__ == "__main__":
    result = adapt_kaggle_dataset()
    print(f"\nDone! Adapted dataset: {result.shape}")
    print(f"   Columns: {result.columns.tolist()}")
    print(f"   Placement rate: {result['Placement Status'].mean()*100:.1f}%")
    print(f"\nSample row:\n{result.iloc[0].to_string()}")
