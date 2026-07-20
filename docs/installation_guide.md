# Installation & Setup Guide

This document provides complete instructions for installing, configuring, running, and troubleshooting the **AI-Powered Student Placement Prediction System**.

---

## Prerequisites

Ensure your development machine meets the following baseline requirements:
* **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+)
* **Python**: Version `3.12` or higher (compatible up to `3.12.x`)
* **Docker** (Optional, for containerized run): Docker Desktop with Compose V2
* **Git**: Command line interface

---

## 1. Local Development Setup

### Step 1: Clone the Repository
Open a terminal and download the codebase:
```bash
git clone <repository_url>
cd placement
```

### Step 2: Configure Virtual Environment
It is highly recommended to isolate dependencies inside a Python virtual environment:

* **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Package Dependencies
Upgrade pip and install the compilation/runtime packages:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> Compiling packages like `shap`, `catboost`, and `lightgbm` on Windows sometimes triggers errors if the **Microsoft Visual C++ Redistributable/Build Tools** are missing. If compilation fails, install the MSVC build tools via [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) or use the Docker installation.

---

## 2. Initialize Data & Train Models

Before running the dashboard, you must generate the cohort dataset and execute the machine learning training pipeline to export the predictive models.

### Step 1: Generate Raw Cohort (5500+ records)
Runs database initialization and saves the synthetic dataset:
```bash
python src/generate_data.py
```
*Output verification*: Check that files `data/raw/placement_dataset.csv` and `database/placement.db` have been created.

### Step 2: Train & Benchmarking ML Models
Triggers cross-validation, hyperparameter tuning, model ranking, leaderboard logging, and diagnostic chart generation:
```bash
python src/train.py
```
*Output verification*: Ensure `models/placement_model.pkl`, `models/preprocessor.pkl`, and metric curves inside `reports/` exist.

---

## 3. Running the Applications

To run the system locally, you must launch both the FastAPI backend and Streamlit web dashboard.

### Step 1: Run FastAPI Application
Launches the API gateway on port `8000`:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to inspect the interactive Swagger API documentation.

### Step 2: Run Streamlit Web Dashboard
In a separate terminal (with virtual environment activated), start the frontend:
```bash
streamlit run streamlit/app.py
```
The dashboard will load automatically at [http://localhost:8501](http://localhost:8501).

---

## 4. Container Deployment (Docker)

To run the complete system inside container sandboxes without manual package installations:

1. **Verify Docker Status**: Ensure Docker is active:
   ```bash
   docker --version
   docker-compose --version
   ```
2. **Build and Spin Up Containers**:
   ```bash
   docker-compose up --build
   ```
3. **Verify running containers**:
   * Streamlit Frontend: [http://localhost:8501](http://localhost:8501)
   * FastAPI swagger gateway: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Troubleshooting

### Issue 1: `ModuleNotFoundError` on running scripts
* **Solution**: Ensure your python path includes the root directory. Execute from root as `python src/train.py`, or set the env variable `PYTHONPATH=.`.

### Issue 2: SQLite database locked errors
* **Solution**: SQLite limits write locks to a single process. Stop duplicate API tasks or active Python debuggers that might have opened transactions.

### Issue 3: SHAP explainer calculations take too long
* **Solution**: The KernelExplainer runs predictions many times. In the predict/explain views, we summarize the background training set to 50 representative samples using k-means clustering to bound calculation times to under 3 seconds. Ensure background data is not too large if altering `src/explain.py`.
