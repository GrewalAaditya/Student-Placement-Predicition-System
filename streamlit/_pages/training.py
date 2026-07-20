import streamlit as st
import json
import pandas as pd
import plotly.express as px
from src.config import METRICS_PATH, REPORTS_DIR, DATA_RAW_DIR
from src.database import SessionLocal, TrainingLog

def render_training_page():
    """Render model training controls, leaderboard, and evaluation plots."""
    st.markdown('<h1 class="main-title">Model Training & Benchmarking</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Train and compare 13 classifiers, then inspect evaluation curves for the best model.</p>', unsafe_allow_html=True)

    raw_dataset = DATA_RAW_DIR / "placement_dataset.csv"
    if not raw_dataset.exists():
        st.warning("⚠️ Active training dataset not found in data/raw/placement_dataset.csv. Upload a full dataset on the Upload page first.")
        st.info("Expected columns: Academic, technical, soft-skill features plus target 'Placement Status'.")

    # 1. Training trigger card
    st.subheader("Training Dashboard Control")
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col_ctrl, col_info = st.columns([1, 2])
        
        with col_ctrl:
            if st.button("Run Training", type="primary", use_container_width=True):
                if not raw_dataset.exists():
                    st.error("Cannot train: no active dataset found in data/raw/placement_dataset.csv.")
                else:
                    from api_client import train_via_api
                    success, res, err = train_via_api()
                    if success:
                        st.success(f"Training initiated in background via API! Run ID: {res.get('run_id')}")
                        st.info("The models are training in the background. Please refresh this page in a few moments to view the updated leaderboard and charts.")
                    else:
                        st.warning(err)
                        
                        import uuid
                        from src.train import train_and_evaluate_all
                        from src.evaluate import generate_all_evaluation_plots
                        try:
                            run_id = f"RUN_LOCAL_{uuid.uuid4().hex[:6].upper()}"
                            with st.spinner("Running training locally as fallback..."):
                                leaderboard, metrics, preprocessor, model, data_tuple = train_and_evaluate_all(run_id)
                                X_train_proc, X_test_proc, y_train, y_test = data_tuple
                                generate_all_evaluation_plots(model, preprocessor, X_test_proc, y_test, metrics["processed_features"])
                            st.success(f"Local training complete! Model: {metrics['best_model_name']} (F1: {metrics['test_f1_score']})")
                            st.experimental_rerun()
                        except Exception as train_err:
                            st.error(f"Local training failed: {str(train_err)}")
        
        with col_info:
            st.markdown("""
            * **Action**: Trains **13 base algorithms** with 5-fold cross-validation.
            * **Tuning**: Grid search on the top candidate model.
            * **Outputs**: Saves `placement_model.pkl`, `preprocessor.pkl`, and evaluation metrics.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("---")

    # 2. Leaderboard visualization
    st.subheader("Model Benchmarking Leaderboard")
    
    # Load metrics from file or database logs
    metrics_loaded = False
    metrics_data = {}
    
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
            metrics_loaded = True
        except Exception:
            pass
            
    if metrics_loaded:
        leaderboard_df = pd.DataFrame(metrics_data["leaderboard"])
        # Format columns for display
        st.write(f"🏆 **Best Performing Model**: `{metrics_data['best_model_name']}` (Hold-out Test Set F1: `{metrics_data['test_f1_score']:.4f}`)")
        
        # Plot metrics comparison chart
        fig_leader = px.bar(
            leaderboard_df, x="Model", y=["CV F1-Score", "CV Accuracy", "CV ROC-AUC"],
            barmode="group",
            title="Algorithm Comparison Across CV Metrics",
            color_discrete_sequence=["#0185FB", "#10B981", "#8B5CF6"],
            template="plotly_dark"
        )
        st.plotly_chart(fig_leader, use_container_width=True)

        st.dataframe(
            leaderboard_df, 
            use_container_width=True,
            column_config={
                "CV Accuracy": st.column_config.NumberColumn(format="%.4f"),
                "CV Precision": st.column_config.NumberColumn(format="%.4f"),
                "CV Recall": st.column_config.NumberColumn(format="%.4f"),
                "CV F1-Score": st.column_config.NumberColumn(format="%.4f"),
                "CV ROC-AUC": st.column_config.NumberColumn(format="%.4f")
            },
            hide_index=True
        )

        st.write("---")

        # 3. Diagnostic Charts
        st.subheader(f"Evaluation Curves for Active Model ({metrics_data['best_model_name']})")
        
        diag_tab1, diag_tab2, diag_tab3 = st.tabs([
            "📈 Performance Curves (ROC & PR)", 
            "🔢 Confusion Matrix & Learning Curve", 
            "⭐ Feature Importance Analysis"
        ])
        
        with diag_tab1:
            col_roc, col_pr = st.columns(2)
            with col_roc:
                roc_img = REPORTS_DIR / "roc_curve.png"
                if roc_img.exists():
                    st.image(str(roc_img), caption="Receiver Operating Characteristic (ROC) Curve", use_container_width=True)
                else:
                    st.info("ROC Curve image not found in reports.")
            with col_pr:
                pr_img = REPORTS_DIR / "precision_recall_curve.png"
                if pr_img.exists():
                    st.image(str(pr_img), caption="Precision-Recall (PR) Curve", use_container_width=True)
                else:
                    st.info("PR Curve image not found in reports.")
                    
        with diag_tab2:
            col_cm, col_lc = st.columns(2)
            with col_cm:
                cm_img = REPORTS_DIR / "confusion_matrix.png"
                if cm_img.exists():
                    st.image(str(cm_img), caption="Confusion Matrix Heatmap", use_container_width=True)
                else:
                    st.info("Confusion Matrix image not found.")
            with col_lc:
                lc_img = REPORTS_DIR / "learning_curve.png"
                if lc_img.exists():
                    st.image(str(lc_img), caption="Model Learning Curve (Sample Size vs Accuracy)", use_container_width=True)
                else:
                    st.info("Learning Curve image not found.")
                    
        with diag_tab3:
            col_fi, col_hyper = st.columns([3, 2])
            with col_fi:
                fi_img = REPORTS_DIR / "feature_importance.png"
                if fi_img.exists():
                    st.image(str(fi_img), caption="Top Feature Importances", use_container_width=True)
                else:
                    st.info("Feature Importance plot not found.")
            with col_hyper:
                st.write("#### Hyperparameter Configuration")
                st.markdown(f"**Selected Best Classifier:** `{metrics_data['best_model_name']}`")
                st.markdown(f"**Tuned Hyperparameters:**")
                st.json(metrics_data.get("best_params", {}))
                
                # Check for active classification report txt
                report_txt = REPORTS_DIR / "classification_report.txt"
                if report_txt.exists():
                    st.write("#### Classification Report Summary")
                    with open(report_txt, "r") as r_f:
                        st.code(r_f.read())
    else:
        st.info("No training metrics loaded. Please run the model training pipeline to generate benchmark leaderboards.")
