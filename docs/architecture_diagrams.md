# System Architecture & Design Diagrams

This document contains visual design specifications, block diagrams, and system layouts for the **AI-Powered Student Placement Prediction System**.

---

## 1. System Block Architecture

Describes the overall microservice interaction between the Streamlit client, FastAPI router, SQLite database, and the core ML processing engine.

```mermaid
graph TD
    subgraph UI Layer
        A[Streamlit Web Client]
        A1[Custom CSS Theme]
    end
    
    subgraph Service API Layer
        B[FastAPI Application Gateway]
        B1[CORS Middleware]
        B2[Pydantic Request Validators]
        B3[Background Task Scheduler]
    end
    
    subgraph Processing Layer
        C[Placement Predictor Inference Engine]
        D[Trained Model / preprocessor.pkl]
        E[SHAP Explainer Engine]
        F[ML Training Pipeline / GridSearchCV]
    end
    
    subgraph Data & Storage Layer
        G[(SQLite Database)]
        H[(Raw Dataset CSV Files)]
        I[(Generated Metric Visual Charts)]
    end

    A -->|1. REST Request JSON/Files| B
    A1 -.-> A
    B -->|2. Check Inputs & Dispatch| B2
    B2 -->|3. Invoke Inference| C
    C -->|Load Parameters| D
    C -->|Calculate XAI Attribution| E
    B -->|Async Request| B3
    B3 -->|4. Run Training| F
    F -->|Export pkl| D
    F -->|Export charts| I
    F -->|Log Metrics| G
    B -->|Query Records| G
    B -->|Upload/Save CSV| H
    G -.->|Query Predictions| A
    I -.->|Render PNGs| A
```

---

## 2. Data Flow Diagram (DFD Level 1)

Illustrates how data enters, processes through pipelines, updates database records, and returns predictions back to the user.

```mermaid
graph TD
    User([Student/Admin User])
    P1[Data Preprocessing & Feature Engineering]
    P2[Model Predictor Inference]
    P3[SHAP Local Calculation]
    P4[Model Trainer Engine]
    
    DB_Pred[(SQLite Predictions table)]
    DB_Train[(SQLite TrainingLogs table)]
    DS_Raw[(Raw Dataset CSV)]
    DS_Model[(Saved Models & Encoders)]
    
    User -->|1. Ingest Dataset CSV| DS_Raw
    DS_Raw -->|2. Pull Training Cohort| P4
    P4 -->|3. Output Performance Leaders| DB_Train
    P4 -->|4. Export optimal model| DS_Model
    
    User -->|5. Single Student Profile Form| P1
    P1 -->|6. Engineered & Scaled Inputs| P2
    P2 -->|7. Load model parameters| DS_Model
    P2 -->|8. Predicted Class & Probability| DB_Pred
    P2 -->|9. Trigger Attribution| P3
    P3 -->|10. shap values waterfall plot| User
    DB_Pred -.->|11. Historical logs audit| User
```

---

## 3. Entity-Relationship (ER) Diagram

Represents the relational database schema implemented in SQLite using SQLAlchemy.

```mermaid
erDiagram
    users {
        int id PK
        string username UK
        string password_hash
        string role
        datetime created_at
    }
    
    uploaded_datasets {
        int id PK
        string filename
        string filepath
        int record_count
        datetime uploaded_at
    }
    
    training_logs {
        int id PK
        string run_id UK
        string model_name
        float accuracy
        float precision
        float recall
        float f1_score
        float roc_auc
        text hyperparameters
        datetime trained_at
    }
    
    predictions {
        int id PK
        string student_id
        float cgpa
        string branch
        float attendance
        float programming_score
        float aptitude_score
        float interview_score
        float resume_score
        int internships
        int projects_completed
        int certifications
        boolean predicted_placed
        float placement_probability
        float readiness_score
        float expected_salary
        string job_category
        text input_features_json
        datetime predicted_at
    }
```

---

## 4. Use Case Diagram

Defines how the roles (Admin and Student) interact with the functional requirements of the system.

```mermaid
left_to_right_direction
graph TD
    Admin([System Admin])
    Student([Student Profile])
    
    UC1(Upload New Training Dataset)
    UC2(Trigger Model Benchmarking)
    UC3(Inspect Leaderboards & Diagnostic Curves)
    UC4(Input Profile Details for Single Prediction)
    UC5(Upload Resume PDF/TXT to pre-fill form)
    UC6(Examine SHAP Waterfall Explanations)
    UC7(Run Batch Prediction CSV)
    UC8(Reset Databases & Purge Records)

    Admin --> UC1
    Admin --> UC2
    Admin --> UC3
    Admin --> UC8
    
    Student --> UC4
    Student --> UC5
    Student --> UC6
    Student --> UC7
    Student --> UC3
```

---

## 5. Sequence Diagram (Single Prediction Flow)

Displays the chronological workflow of operations executing when a student requests a placement prediction.

```mermaid
sequenceDiagram
    autonumber
    actor User as Streamlit Client
    participant API as FastAPI Backend
    participant DB as SQLite DB
    participant Predictor as PlacementPredictor
    participant Preproc as Preprocessing Pipeline
    participant Model as Saved Best Model
    participant SHAP as SHAP Explainer

    User->>API: POST /predict (Profile JSON)
    Note over API: Schema validation (Pydantic)
    API->>Predictor: predict_single(data)
    Predictor->>Preproc: transform(df_raw)
    Preproc-->>Predictor: returns scaled numpy matrix (1, F)
    Predictor->>Model: predict_proba(X_proc)
    Model-->>Predictor: returns binary prediction [1] & probability [0.88]
    
    Predictor->>Predictor: calculate readiness, salary & recommendations
    Predictor-->>API: returns final JSON payload
    
    API->>DB: Log prediction details to predictions table
    DB-->>API: commit ok
    
    API-->>User: Returns 200 OK Prediction Results
    
    Note over User, SHAP: Local XAI attribution triggered
    User->>API: GET /explain (SHAP attributions)
    API->>SHAP: calculate_shap(processed_input)
    SHAP-->>API: return waterfall.png & force.html paths
    API-->>User: render plots to dashboard UI
```

---

## 6. Class Diagram

Exposes the relationships and method boundaries between key modules in the backend design.

```mermaid
classDiagram
    class OutlierCapper {
        +float factor
        +dict lower_bounds_
        +dict upper_bounds_
        +fit(X, y) OutlierCapper
        +transform(X) DataFrame
    }
    
    class DataCleaner {
        +fit(X, y) DataCleaner
        +transform(X) DataFrame
    }
    
    class FeatureEngineer {
        +fit(X, y) FeatureEngineer
        +transform(X) DataFrame
    }
    
    class PlacementPredictor {
        +object model
        +object preprocessor
        +load_artifacts() void
        +is_ready() bool
        +predict_single(data) dict
        +predict_batch(df) DataFrame
        +calculate_readiness_score(data) float
        +estimate_salary_range(pred, data) string
        +determine_career_path(data, prob) tuple
        +check_eligibility(data) dict
        +generate_suggestions(data) list
    }
    
    class User {
        +int id
        +string username
        +string password_hash
        +string role
    }
    
    class TrainingLog {
        +int id
        +string run_id
        +string model_name
        +float accuracy
        +float f1_score
    }

    DataCleaner ..> FeatureEngineer : pre-step
    FeatureEngineer ..> OutlierCapper : features mapping
    PlacementPredictor --> User : accesses profile
    PlacementPredictor --> TrainingLog : references metric runs
```
