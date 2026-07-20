import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy.orm import Session
from src.database import SessionLocal, PredictionRecord, TrainingLog
from src.config import MODELS_DIR

def render_home_page():
    st.markdown('<h1 class="main-title">Student Placement Portal</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Predictive analytics and student placement metrics.</p>', unsafe_allow_html=True)

    db = SessionLocal()
    try:
        total_preds = db.query(PredictionRecord).count()
        placed_preds = db.query(PredictionRecord).filter(PredictionRecord.predicted_placed == True).count()
        placed_ratio = (placed_preds / total_preds * 100) if total_preds > 0 else 0.0
        
        avg_readiness = db.query(PredictionRecord).with_entities(PredictionRecord.readiness_score).all()
        avg_readiness_val = pd.Series([r[0] for r in avg_readiness]).mean() if avg_readiness else 0.0
        
        models_trained = db.query(TrainingLog).count()
    finally:
        db.close()

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <p style="color:#9CA3AF; margin-bottom:4px; font-size:0.9rem; font-weight:600;">Predictions Made</p>
            <p class="metric-val">{total_preds}</p>
            <span class="badge badge-blue">Real-time Logs</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <p style="color:#9CA3AF; margin-bottom:4px; font-size:0.9rem; font-weight:600;">Avg. Placement Rate</p>
            <p class="metric-val">{placed_ratio:.1f}%</p>
            <span class="badge badge-green">Inference Ratio</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <p style="color:#9CA3AF; margin-bottom:4px; font-size:0.9rem; font-weight:600;">Avg. Readiness Index</p>
            <p class="metric-val">{avg_readiness_val:.1f}</p>
            <span class="badge badge-yellow">Scale 0-100</span>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="glass-card">
            <p style="color:#9CA3AF; margin-bottom:4px; font-size:0.9rem; font-weight:600;">Models Evaluated</p>
            <p class="metric-val">{models_trained}</p>
            <span class="badge badge-blue">Base Algorithms</span>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")

    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader("System Architecture & Data Flows")
        st.markdown("""
        Standard academic, soft skill, and technical indicators are used to model placement outcomes. 
        Raw features are preprocessed and sent to the active model to calculate placement probability.
        """)
        
        st.markdown("""
        ```mermaid
        graph LR
            A[Student Profile Input] --> B[Feature Eng. Pipeline]
            B --> C[Robust Scaler & Encoder]
            C --> D[Active ML Model]
            D --> E[Binary Decision / Probabilities]
            E --> F[SHAP Explanations]
            E --> G[Suggestions & Job Matching]
            classDef default fill:#1E293B,stroke:#475569,stroke-width:1px,color:#F8FAFC;
        ```
        """)
        
        st.info("💡 Tip: Navigate to **Predict Placement** in the sidebar to run predictions for individual student profiles or generate SHAP waterfall charts.")

    with right_col:
        st.subheader("Active Model Status")
        
        model_exists = (MODELS_DIR / "placement_model.pkl").exists()
        preprocessor_exists = (MODELS_DIR / "preprocessor.pkl").exists()

        if model_exists and preprocessor_exists:
            db = SessionLocal()
            try:
                latest_log = db.query(TrainingLog).order_by(TrainingLog.trained_at.desc()).first()
                if latest_log:
                    st.success(f"**Status: READY**")
                    st.write(f"**Active Model:** {latest_log.model_name}")
                    st.write(f"**F1-Score:** {latest_log.f1_score:.4f}")
                    st.write(f"**Accuracy:** {latest_log.accuracy:.4f}")
                    st.write(f"**Last Trained:** {latest_log.trained_at.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    st.warning("Model files found on disk, but training record missing in database logs.")
            finally:
                db.close()
        else:
            st.error("**Status: MODELS UNTRAINED**")
            st.write("No trained model found. Please go to the **Train Models** page to train and export the best model.")
            if st.button("Go to Model Training"):
                st.session_state.page = "🤖 Train Models"
                st.experimental_rerun()
                
    st.write("---")
    
    st.subheader("Historical Analytics (Database Predictions)")
    db = SessionLocal()
    try:
        query = db.query(PredictionRecord).order_by(PredictionRecord.predicted_at.desc()).limit(100).all()
        if query:
            df_preds = pd.DataFrame([{
                "CGPA": q.cgpa,
                "Branch": q.branch,
                "Programming Score": q.programming_score,
                "Readiness Score": q.readiness_score,
                "Predicted Placement": "Placed" if q.predicted_placed else "Not Placed"
            } for q in query])
            
            fig = px.scatter(
                df_preds, x="Programming Score", y="CGPA", 
                color="Predicted Placement", size="Readiness Score",
                color_discrete_map={"Placed": "#10B981", "Not Placed": "#F43F5E"},
                title="CGPA vs Programming Score",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction queries logged yet. Test a student profile in the 'Predict Placement' screen to seed analytics.")
    finally:
        db.close()

