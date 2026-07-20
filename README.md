# Student Placement Prediction System

A web application that predicts student placement outcomes based on academic records, skill scores, and extracurricular activity. The system compares multiple classifiers, selects the best performing model, and exposes it through a REST API with an interactive analytics dashboard.

---

## Features

1. **Multi-model benchmarking**: Evaluates 13 classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, SVM, KNN, MLP, etc.) using 5-fold cross-validation and ranks them by F1-score.
2. **Model explanations (SHAP)**: Shows global feature importance and per-prediction waterfall/force plots so you can see which factors drove each result.
3. **Interactive charts**: Plotly-based visualisations covering cohort demographics, branch-wise placement rates, and skill correlations.
4. **Resume pre-fill**: Reads a plain-text resume and extracts skills/project counts to pre-fill the prediction form.
5. **Batch predictions**: Upload a CSV of student records and download the file annotated with predictions and probability scores.
6. **Prediction audit log**: All predictions are persisted to SQLite so results can be reviewed and exported later.
7. **REST API**: FastAPI backend with Pydantic validation schemas for training, prediction, and dataset upload endpoints.

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Language | Python 3.12+ |
| ML | Scikit-Learn, XGBoost, LightGBM, CatBoost |
| Explanations | SHAP, Matplotlib, Seaborn |
| API | FastAPI, Uvicorn, Pydantic, SQLAlchemy |
| Database | SQLite |
| Dashboard | Streamlit, Plotly, HTML/CSS |
| Testing | Pytest |
| Deployment | Docker, Docker Compose |

---

## Project Structure

```
placement/
├── data/
│   ├── raw/               # Raw CSV files (uploaded or generated)
│   └── processed/         # Scaled and encoded datasets
├── models/                # Saved model and preprocessor files
├── reports/               # Evaluation plots and classification reports
├── notebooks/             # Exploratory notebooks
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── database.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── explain.py
├── api/
│   └── main.py            # FastAPI routes
├── streamlit/
│   ├── app.py             # Entry point
│   ├── css/style.css
│   └── pages/             # Individual dashboard pages
├── database/              # SQLite files
├── tests/                 # Pytest suite
├── docs/                  # Manuals and project report
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Getting Started

### Requirements

- Python 3.12+
- Git

### Local Setup

1. **Clone the repo**:
   ```bash
   git clone <repository_url>
   cd placement
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate sample data** (also initialises the database):
   ```bash
   python src/generate_data.py
   ```

4. **Train models**:
   ```bash
   python src/train.py
   ```

5. **Start the API and dashboard** (two terminals):
   ```bash
   # Terminal 1
   uvicorn api.main:app --reload --port 8000

   # Terminal 2
   streamlit run streamlit/app.py
   ```

6. Open `http://localhost:8501` in your browser.

---

### Docker

```bash
docker-compose up --build
```

- Dashboard: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

---

## Tests

```bash
pytest tests/ -v
```

---

## License

MIT
