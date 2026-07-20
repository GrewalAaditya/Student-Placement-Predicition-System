import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from src.config import DATA_RAW_DIR

def render_analysis_page():
    """Render the data analysis page with EDA charts."""
    st.markdown('<h1 class="main-title">Data Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Exploratory insights and statistics for the active student training dataset.</p>', unsafe_allow_html=True)

    raw_path = DATA_RAW_DIR / "placement_dataset.csv"
    if not raw_path.exists():
        st.warning("⚠️ No training dataset found. Generate a sample dataset or upload a CSV file first.")
        if st.button("Generate Sample Dataset"):
            # Trigger dataset generator script
            import subprocess
            with st.spinner("Generating dataset..."):
                subprocess.run(["python", "src/generate_data.py"])
                st.success("Dataset generated successfully! Reloading...")
                st.experimental_rerun()
        return

    # Load dataset
    df = pd.read_csv(raw_path)

    # Show dataset description
    st.subheader("Cohort Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cohort Size", f"{df.shape[0]} Students")
    with col2:
        st.metric("Total Branch Profiles", f"{df['Branch'].nunique()}")
    with col3:
        placed_pct = (df["Placement Status"].sum() / len(df)) * 100
        st.metric("Cohort Placement Rate", f"{placed_pct:.1f}%")

    st.write("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎓 Academic Distributions",
        "🛠 Branch Analytics",
        "🤝 Skill Interactions",
        "🔥 Correlation Matrix"
    ])

    with tab1:
        st.subheader("Academic Indicators")
        col_la, col_ra = st.columns(2)
        
        with col_la:
            # CGPA Box Plot colored by Placement
            fig_cgpa = px.box(
                df, x="Placement Status", y="CGPA", 
                color="Placement Status",
                color_discrete_map={1: "#10B981", 0: "#F43F5E"},
                labels={"Placement Status": "Recruited (1=Yes, 0=No)"},
                title="CGPA Range by Placement Outcome",
                template="plotly_dark"
            )
            st.plotly_chart(fig_cgpa, use_container_width=True)
            
        with col_ra:
            # 12th vs 10th percentage scatter
            fig_sec = px.scatter(
                df, x="10th Percentage", y="12th Percentage", 
                color="Placement Status",
                color_discrete_map={1: "#10B981", 0: "#F43F5E"},
                title="Secondary (10th) vs Senior Secondary (12th) Performance",
                template="plotly_dark",
                opacity=0.6
            )
            st.plotly_chart(fig_sec, use_container_width=True)

        # CGPA Histogram
        fig_hist = px.histogram(
            df, x="CGPA", color="Placement Status",
            color_discrete_map={1: "#10B981", 0: "#F43F5E"},
            marginal="rug",
            title="Distribution of CGPA across Placed vs Not Placed Students",
            template="plotly_dark"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.subheader("Branch-wise Recruitment Statistics")
        
        # Calculate rates
        branch_stats = df.groupby("Branch")["Placement Status"].agg(["count", "sum"]).reset_index()
        branch_stats.columns = ["Branch", "Total Students", "Placed Students"]
        branch_stats["Placement Rate (%)"] = np.round((branch_stats["Placed Students"] / branch_stats["Total Students"]) * 100, 2)
        
        col_lb, col_rb = st.columns([3, 2])
        
        with col_lb:
            fig_branch = px.bar(
                branch_stats, x="Branch", y="Placement Rate (%)",
                color="Branch",
                title="Placement Percentage by Engineering Discipline",
                template="plotly_dark"
            )
            st.plotly_chart(fig_branch, use_container_width=True)
            
        with col_rb:
            st.write("#### Branch Details Summary")
            st.dataframe(branch_stats, use_container_width=True, hide_index=True)

        # Attendance by branch
        fig_att = px.box(
            df, x="Branch", y="Attendance", color="Placement Status",
            color_discrete_map={1: "#10B981", 0: "#F43F5E"},
            title="Attendance Rates by Branch and Placement Outcome",
            template="plotly_dark"
        )
        st.plotly_chart(fig_att, use_container_width=True)

    with tab3:
        st.subheader("Skill Interactions")
        
        col_lc, col_rc = st.columns(2)
        with col_lc:
            # Coding vs Aptitude Score
            fig_skills = px.scatter(
                df, x="Programming Score", y="Aptitude Score", 
                color="Placement Status",
                color_discrete_map={1: "#10B981", 0: "#F43F5E"},
                title="Programming Score vs Aptitude Score",
                template="plotly_dark",
                opacity=0.7
            )
            st.plotly_chart(fig_skills, use_container_width=True)
            
        with col_rc:
            # Communication vs Interview Score
            fig_comm = px.scatter(
                df, x="Communication Skills", y="Interview Score", 
                color="Placement Status",
                color_discrete_map={1: "#10B981", 0: "#F43F5E"},
                title="Communication Skills vs Interview Performance",
                template="plotly_dark",
                opacity=0.7
            )
            st.plotly_chart(fig_comm, use_container_width=True)

        # Internships vs Projects completed
        fig_projects = px.density_heatmap(
            df, x="Projects Completed", y="Internships", 
            z="Placement Status", histfunc="avg",
            title="Average Placement Probability by Projects and Internships Counts",
            template="plotly_dark",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_projects, use_container_width=True)

    with tab4:
        st.subheader("Numerical Features Correlation Matrix")
        
        numeric_df = df.select_dtypes(include=[np.number])
        # Drop columns with zero variance or ID columns if mapped
        numeric_df = numeric_df.drop(columns=["Expected Salary"], errors="ignore")
        
        corr = numeric_df.corr().round(2)
        
        # Render Heatmap
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Coefficients Heatmap",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            template="plotly_dark"
        )
        fig_corr.update_layout(height=800)
        st.plotly_chart(fig_corr, use_container_width=True)
