import streamlit as st
import pandas as pd
import os
from pathlib import Path
from src.config import REPORTS_DIR, MODELS_DIR, METRICS_PATH
from src.database import SessionLocal, PredictionRecord, TrainingLog, engine, Base

def render_settings_page():
    """Render the settings, downloads, and maintenance page."""
    st.markdown('<h1 class="main-title">Settings & Downloads</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Download reports, view prediction logs, or reset the database.</p>', unsafe_allow_html=True)

    st.subheader("📥 Reports & Artifacts")

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Summary Reports")

        cls_report = REPORTS_DIR / "classification_report.txt"
        if cls_report.exists():
            with open(cls_report, "r") as f:
                report_data = f.read()
            st.download_button(
                label="Download Classification Report (TXT)",
                data=report_data,
                file_name="classification_report.txt",
                mime="text/plain",
                use_container_width=True
            )
            
        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r") as f:
                json_data = f.read()
            st.download_button(
                label="Download Metrics Log (JSON)",
                data=json_data,
                file_name="metrics.json",
                mime="application/json",
                use_container_width=True
            )
            
    with col2:
        st.write("#### Evaluation Charts")
        
        cm_path = REPORTS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            with open(cm_path, "rb") as f:
                cm_bytes = f.read()
            st.download_button(
                label="Download Confusion Matrix (PNG)",
                data=cm_bytes,
                file_name="confusion_matrix.png",
                mime="image/png",
                use_container_width=True
            )
            
        roc_path = REPORTS_DIR / "roc_curve.png"
        if roc_path.exists():
            with open(roc_path, "rb") as f:
                roc_bytes = f.read()
            st.download_button(
                label="Download ROC-AUC Curve (PNG)",
                data=roc_bytes,
                file_name="roc_curve.png",
                mime="image/png",
                use_container_width=True
            )

    st.write("---")

    st.subheader("📂 Prediction Logs")
    
    try:
        db = SessionLocal()
        try:
            records = db.query(PredictionRecord).order_by(PredictionRecord.predicted_at.desc()).limit(15).all()
            if records:
                df_log = pd.DataFrame([{
                    "ID": r.id,
                    "Student ID": r.student_id,
                    "CGPA": r.cgpa,
                    "Branch": r.branch,
                    "Readiness Score": r.readiness_score,
                    "Prediction": "Placed" if r.predicted_placed else "Not Placed",
                    "Probability %": r.placement_probability,
                    "Timestamp": r.predicted_at.strftime('%Y-%m-%d %H:%M')
                } for r in records])
                
                st.dataframe(df_log, use_container_width=True, hide_index=True)
                
                # Export to csv option
                csv_df = pd.DataFrame([{
                    "id": r.id,
                    "student_id": r.student_id,
                    "cgpa": r.cgpa,
                    "branch": r.branch,
                    "attendance": r.attendance,
                    "programming_score": r.programming_score,
                    "predicted_placed": r.predicted_placed,
                    "probability": r.placement_probability,
                    "readiness_score": r.readiness_score,
                    "predicted_at": r.predicted_at
                } for r in db.query(PredictionRecord).all()])
                
                csv_str = csv_df.to_csv(index=False)
                st.download_button(
                    label="Download Prediction History (CSV)",
                    data=csv_str,
                    file_name="placement_prediction_history.csv",
                    mime="text/csv"
                )
            else:
                st.info("No predictions logged in SQLite database yet.")
        finally:
            db.close()
    except Exception as db_err:
        st.info("Historical database prediction logs are currently unavailable.")

    st.write("---")

    st.subheader("⚙️ Maintenance")

    st.warning("⚠️ The following operations are destructive and cannot be undone.")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        if st.button("Reset Predictions Log Database", type="secondary", use_container_width=True):
            db = SessionLocal()
            try:
                db.query(PredictionRecord).delete()
                db.commit()
                st.success("Successfully purged prediction history logs.")
                st.experimental_rerun()
            except Exception as reset_err:
                st.error(f"Failed to reset predictions: {str(reset_err)}")
            finally:
                db.close()
                
    with m_col2:
        if st.button("Reset & Rebuild Database", type="secondary", use_container_width=True):
            try:
                Base.metadata.drop_all(bind=engine)
                Base.metadata.create_all(bind=engine)
                
                # Delete generated models and reports
                for item in [MODELS_DIR / "placement_model.pkl", MODELS_DIR / "preprocessor.pkl", METRICS_PATH]:
                    if item.exists():
                        os.remove(item)
                        
                st.success("Database drop-and-rebuild succeeded! Cleared active model files.")
                st.experimental_rerun()
            except Exception as db_drop_err:
                st.error(f"Rebuild failed: {str(db_drop_err)}")
