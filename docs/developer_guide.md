# Developer & Architecture Guide

This guide is designed for developers, data scientists, and engineers who wish to maintain, extend, or debug the **Student Placement Prediction System**.

---

## 1. Modular Core Architecture

The system is structured according to SOLID and OOP principles. The folder structure segregates preprocessing, model training, database schema definitions, and web interfaces.

### Core Modules (`src/`)

```
src/
├── config.py             # System configurations and parameters
├── utils.py              # Logging setup and object pickling
├── database.py           # Relational schema mappings (SQLAlchemy)
├── preprocessing.py      # Custom Transformers and data transformations
├── feature_engineering.py# Custom engineered column definitions
├── train.py              # Benchmarking comparative loops
├── evaluate.py           # Matplotlib metrics plotting
├── predict.py            # Prediction rule engine class
└── explain.py            # SHAP explainer adapters
```

* **`config.py`**: Centralizes constants, schema column names, path locations, and candidate parameter grids for `GridSearchCV`.
* **`database.py`**: Implements SQLite mapping using SQLAlchemy. Contains tables for users, datasets, training metrics, and prediction records. Includes a context manager for session generation.
* **`preprocessing.py`**: Exports the pipeline builder. Includes two custom estimators:
  * `OutlierCapper`: Automatically caps outliers within numeric boundaries using IQR limits.
  * `DataCleaner`: Standardizes text entries by stripping trailing spaces.
* **`feature_engineering.py`**: Defines standard engineering calculations like `Employability_Readiness_Score` and `Academic_Performance_Index`.
* **`train.py`**: Implements cross-validation on 13 base classifiers, ranks them, runs GridSearch on top tree classifiers, and saves the final best model.
* **`predict.py`**: Serves predictions. Embeds custom rule logic for expected salary estimation, career profile suitability, eligibility checks, and personalized improvement warnings.

---

## 2. Extending the Codebase

### A. Adding a New Column / Feature
If you add a new column to the dataset:
1. Update `CATEGORICAL_COLS` or `NUMERICAL_COLS` in `src/config.py`.
2. If the feature requires customization (e.g. custom binning or imputation), adjust `build_preprocessing_pipeline` in `src/preprocessing.py`.
3. If it's used in prediction logic, update the `StudentFeatures` validator schema in `api/main.py` and form fields in `streamlit/pages/prediction.py`.

### B. Adding a New Classifier to Compare
To add another model to the benchmark leaderboard:
1. Import the model class in `src/train.py`.
2. Append it to the dictionary returned by `get_base_models()`:
   ```python
   "My Classifier": MyClassifier(random_state=42)
   ```
3. If you want to enable hyperparameter tuning for it, define its param grid in `HYPERPARAMETER_GRIDS` in `src/config.py` and register it in `train.py`.

### C. Modifying Suggestion Checklist Rules
To add suggestion items:
* Open `src/predict.py`.
* Navigate to the `generate_suggestions()` method.
* Implement conditional logic checks and append strings to the output list.

---

## 3. Database Maintenance & Migrations
* Database files are persisted inside `database/placement.db` (or mapped docker volumes).
* Table schema modifications require dropping tables (`Base.metadata.drop_all(bind=engine)`) and regenerating them. In production environments, integrate migrations using Alembic.

---

## 4. Running Unit Tests

The test suite validates pipeline steps, dataset caps, model calculations, and API endpoints using `pytest`.

Ensure dependencies are installed, then run tests from the root directory:

```bash
# Standard run
pytest tests/ -v

# Run tests and print logs/captures
pytest tests/ -v -s

# Run specific test file
pytest tests/test_preprocessing.py
```
To check test coverage, install `pytest-cov` and execute:
```bash
pytest --cov=src tests/
```
