# Student Placement Prediction Portal User Manual

Welcome to the **AI-Powered Student Placement Prediction System**. This user manual will guide you through the functionalities, workflows, and pages available in the analytics dashboard.

---

## Dashboard Layout & Sidebar Navigation

The web portal is split into:
1. **Sidebar Control Panel**: Displays database/system health status and provides page-navigation radio selectors.
2. **Main Application Board**: Where interactive forms, plotly graphs, gauges, and explanations are rendered.

---

## 1. 🏠 Home Page Dashboard
* **Purpose**: Serves as the landing page showing active KPI metrics for the database.
* **Key Features**:
  * Displays total predictions recorded, placement rates, average readiness score, and models trained.
  * Shows the **System Block Architecture**.
  * Shows an interactive scatter graph representing historical predictions logged in the SQLite database.

---

## 2. 📂 Upload Dataset
* **Purpose**: Allows institutional administrators to upload raw CSV files containing new cohorts.
* **Instructions**:
  1. Click **Browse files** or drag a CSV file onto the dashed upload container.
  2. Inspect the **Data Preview** grid and check class distribution balances and branch rates plotted automatically.
  3. Click **Ingest to Database & Save** to persist raw files on disk. The system will set the uploaded file as the default cohort for model training.

---

## 3. 📊 Data Analysis
* **Purpose**: Offers deep exploratory data insights on the active cohort.
* **Interactive Tabs**:
  * **Academic Distributions**: Inspects CGPA range spreads by placement outcomes and visualizes secondary school score scatter grids.
  * **Branch Analytics**: Evaluates recruitment rates across individual engineering branches.
  * **Skill Interactions**: Compares programming scores with aptitude/interview metrics to observe clear boundary splits.
  * **Correlation Matrix**: A complete correlation heatmap checking dependencies between all numerical scores.

---

## 4. 🤖 Train Models
* **Purpose**: Automates model benchmarking and grid search operations.
* **Instructions**:
  1. Click the **Trigger Training Pipeline** button. 
  2. If the API is running, training is offloaded in the background. (Alternatively, the system falls back to synchronous local training).
  3. After a brief wait, refresh the page. The dashboard will display the **Model Benchmarking Leaderboard** ranking the 13 algorithms.
  4. Expand the tabs to inspect the **Confusion Matrix**, **ROC Curve**, **Precision-Recall Curve**, **Learning Curve**, and **Feature Importance** for the optimal classifier.

---

## 5. 🎯 Predict Placement
* **Purpose**: Performs individual student prediction inferences and batch predictions.
* **Features**:
  * **Resume parser pre-fill**: Upload a plain-text resume (.txt). The parser extracts technical keywords, estimates projects/internships, and automatically adjusts the input form parameters.
  * **Single Student Profile Form**: Fill academic averages, coding scores, and portfolio items.
  * **Action**: Click **Predict Placement Probability**.
  * **Outputs**:
    * **Placement status gauge**: Displays Placed (Green) or Not Placed (Red) with probability & confidence metrics.
    * **Eligibility badge**: Evaluates if the student is blocked by backlogs or low attendance.
    * **Salary & Career matching**: Identifies expected packages and optimal job profiles.
    * **Personalized Suggestions**: Provides targeted checklist warnings (e.g. increase attendance, clear backlogs, boost programming).

---

## 6. 🧠 Explain Prediction
* **Purpose**: Inspects prediction diagnostics using Explainable AI (SHAP).
* **Instructions**:
  1. Execute a single prediction first on the **Predict Placement** page.
  2. Navigate to **Explain Prediction**.
  3. Click **Generate Local SHAP Graphs**.
  4. Inspect the **SHAP Waterfall Plot** showing feature contributions pushing the prediction away from the average baseline.
  5. Inspect the interactive **SHAP Force Plot** (visualizing opposing positive and negative forces).
  6. Click **Generate/Update Global SHAP Summary Plot** to examine overall feature impact rankings.

---

## 7. ⚙ Settings & Downloads
* **Purpose**: Downloads reports and runs SQLite database maintenance.
* **Key Actions**:
  * **Download classification reports** and metrics log JSONs.
  * **Download diagnostic charts** (Confusion matrices, ROC curves).
  * **Audit history logs**: Review and download prediction records logged in SQLite.
  * **Reset logs**: Purge historical logs or recreate databases from scratch.
