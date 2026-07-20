import streamlit as st
import io
from src.config import MODELS_DIR, FEATURES
from src.utils import pandas_read_csv, ensure_feature_columns, normalize_input_dict

def extract_skills_from_text(text: str) -> dict:
    """Parses text to extract skills and estimate profile metrics."""
    skills_list = ["python", "java", "c++", "sql", "javascript", "react", "machine learning", "cloud", "aws", "azure", "docker", "git"]
    found_skills = []
    
    text_lower = text.lower()
    for skill in skills_list:
        if skill in text_lower:
            found_skills.append(skill.capitalize())
            
    # Estimate projects & internships
    project_keywords = ["project", "developed", "built", "github"]
    internship_keywords = ["intern", "internship", "trainee", "work experience"]
    cert_keywords = ["certified", "certification", "certify", "coursera", "udemy"]
    
    project_count = sum(1 for kw in project_keywords if kw in text_lower)
    intern_count = sum(1 for kw in internship_keywords if kw in text_lower)
    cert_count = sum(1 for kw in cert_keywords if kw in text_lower)
    
    # Bound estimated counts
    estimated_projects = min(max(project_count // 2, 0), 4)
    estimated_internships = min(max(intern_count // 2, 0), 3)
    estimated_certs = min(max(cert_count // 2, 0), 4)
    
    # Calculate resume score based on length and elements found
    base_score = 45.0
    base_score += len(found_skills) * 4
    base_score += estimated_projects * 5
    base_score += estimated_internships * 8
    base_score = min(base_score, 100.0)
    
    return {
        "skills": found_skills,
        "projects": estimated_projects,
        "internships": estimated_internships,
        "certifications": estimated_certs,
        "resume_score": base_score
    }

def render_prediction_page():
    """Renders single student inputs, resume upload parser, and batch CSV predictions."""
    st.markdown('<h1 class="main-title">Placement Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Assess student eligibility, run predictive inference, and retrieve recommendations.</p>', unsafe_allow_html=True)

    # Check active model
    model_path = MODELS_DIR / "placement_model.pkl"
    if not model_path.exists():
        st.warning("⚠️ The placement prediction model is not trained yet. Navigate to 'Train Models' to initialize the ML pipeline.")
        return

    # Multitab prediction modes
    tab_single, tab_batch = st.tabs(["🎯 Single Student Prediction", "📂 CSV Batch Prediction"])

    with tab_single:
        # Initialize default values in session state if not existing (for pre-filling via resume upload)
        if "pred_form" not in st.session_state:
            st.session_state.pred_form = {
                "CGPA": 7.8,
                "Branch": "Computer Science",
                "Degree": "B.Tech",
                "Age": 21,
                "Gender": "Male",
                "10th Percentage": 80.0,
                "12th Percentage": 78.0,
                "Backlogs": 0,
                "Attendance": 85.0,
                "Technical Skills": 70.0,
                "Programming Score": 70.0,
                "Aptitude Score": 65.0,
                "Communication Skills": 70.0,
                "Soft Skills": 70.0,
                "Projects Completed": 1,
                "Internships": 0,
                "Hackathons": 0,
                "Certifications": 0,
                "Workshops": 1,
                "Leadership Score": 60.0,
                "Resume Score": 65.0,
                "Interview Score": 68.0,
                "English Proficiency": "Medium",
                "Expected Salary": 450000.0
            }

        with st.expander("Upload resume to auto-fill the form", expanded=False):
            resume_file = st.file_uploader("Upload resume file (TXT format)", type=["txt"])
            if resume_file is not None:
                try:
                    resume_text = resume_file.read().decode("utf-8")
                    extracted = extract_skills_from_text(resume_text)
                    st.success(f"Resume parsed! Extracted skills: {', '.join(extracted['skills'])}")
                    st.session_state.pred_form["Projects Completed"] = extracted["projects"]
                    st.session_state.pred_form["Internships"] = extracted["internships"]
                    st.session_state.pred_form["Certifications"] = extracted["certifications"]
                    st.session_state.pred_form["Resume Score"] = extracted["resume_score"]
                    st.session_state.pred_form["Technical Skills"] = min(60.0 + len(extracted["skills"])*5, 100.0)
                    st.info("Updated form parameters based on resume profile.")
                except Exception as parse_err:
                    st.error(f"Error parsing resume: {parse_err}")

        st.write("---")
        st.subheader("Student Profile Input")
        with st.form("prediction_input_form"):
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            form_col1, form_col2, form_col3 = st.columns([1, 1, 1])

            with form_col1:
                st.markdown("##### 🎓 Academic Profile")
                cgpa = st.slider("CGPA", 4.0, 10.0, st.session_state.pred_form["CGPA"], step=0.1, help="Cumulative grade point average on a 10-point scale")
                branch = st.selectbox("Branch", ["Computer Science", "Information Technology", "Electronics & Communication", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering"], index=0, help="Academic department or specialization")
                degree = st.selectbox("Degree", ["B.Tech", "M.Tech", "MCA", "BCA"], index=0, help="Highest current degree program")
                age = st.slider("Age", 18, 30, st.session_state.pred_form["Age"], help="Confirm the student's current age")
                gender = st.radio("Gender", ["Male", "Female"], horizontal=True, help="Student gender for demographic adjustment")
                tenth = st.number_input("10th Percentage", 40.0, 100.0, st.session_state.pred_form["10th Percentage"], step=0.5, help="Secondary school score percentage")
                twelfth = st.number_input("12th Percentage", 40.0, 100.0, st.session_state.pred_form["12th Percentage"], step=0.5, help="Senior secondary score percentage")
                attendance = st.slider("Attendance %", 40.0, 100.0, st.session_state.pred_form["Attendance"], step=1.0, help="Overall attendance percentage")
                backlogs = st.number_input("Active Backlogs", 0, 8, st.session_state.pred_form["Backlogs"], help="Number of active academic backlogs")

            with form_col2:
                st.markdown("##### 🛠 Skills & Aptitude")
                tech_skills = st.slider("Technical Skills", 30.0, 100.0, st.session_state.pred_form["Technical Skills"], step=1.0, help="Overall technical skill rating")
                prog_score = st.slider("Programming Score", 30.0, 100.0, st.session_state.pred_form["Programming Score"], step=1.0, help="Programming test score")
                aptitude = st.slider("Aptitude Score", 30.0, 100.0, st.session_state.pred_form["Aptitude Score"], step=1.0, help="Aptitude test score")
                comm = st.slider("Communication Skills", 30.0, 100.0, st.session_state.pred_form["Communication Skills"], step=1.0, help="Communication quality rating")
                soft_s = st.slider("Soft Skills", 30.0, 100.0, st.session_state.pred_form["Soft Skills"], step=1.0, help="Soft skills rating")
                english = st.selectbox("English Proficiency", ["High", "Medium", "Low"], index=1, help="English fluency level")
                expected_salary = st.number_input("Expected Salary (INR)", 150000.0, 2000000.0, st.session_state.pred_form["Expected Salary"], step=10000.0, help="Expected annual salary requirement")

            with form_col3:
                st.markdown("##### 🏆 Professional Portfolio")
                projects = st.number_input("Projects Completed", 0, 10, st.session_state.pred_form["Projects Completed"], help="Count of completed projects")
                internships = st.number_input("Internships", 0, 5, st.session_state.pred_form["Internships"], help="Number of internships completed")
                hackathons = st.number_input("Hackathons Won/Attended", 0, 5, st.session_state.pred_form["Hackathons"], help="Hackathon participation count")
                certs = st.number_input("Certifications", 0, 10, st.session_state.pred_form["Certifications"], help="Relevant certifications")
                workshops = st.number_input("Workshops Attended", 0, 10, st.session_state.pred_form["Workshops"], help="Number of workshops attended")
                leadership = st.slider("Leadership Score", 30.0, 100.0, st.session_state.pred_form["Leadership Score"], step=1.0, help="Leadership and teamwork rating")
                resume_s = st.slider("Resume Score", 30.0, 100.0, st.session_state.pred_form["Resume Score"], step=1.0, help="Resume strength rating")
                interview_s = st.slider("Interview Score", 30.0, 100.0, st.session_state.pred_form["Interview Score"], step=1.0, help="Interview preparedness score")

            st.markdown("</div>", unsafe_allow_html=True)
            
            st.session_state.pred_form = {
                "CGPA": cgpa, "Branch": branch, "Degree": degree, "Age": age, "Gender": gender,
                "10th Percentage": tenth, "12th Percentage": twelfth, "Backlogs": backlogs,
                "Attendance": attendance, "Technical Skills": tech_skills, "Programming Score": prog_score,
                "Aptitude Score": aptitude, "Communication Skills": comm, "Soft Skills": soft_s,
                "Projects Completed": projects, "Internships": internships, "Hackathons": hackathons,
                "Certifications": certs, "Workshops": workshops, "Leadership Score": leadership,
                "Resume Score": resume_s, "Interview Score": interview_s, "English Proficiency": english,
                "Expected Salary": expected_salary
            }

            submit_pred = st.form_submit_button("Predict Placement Probability", type="primary")

        if submit_pred:
            with st.spinner("Processing prediction..."):
                try:
                    # Construct request body compatible with fastapi pydantic model
                    req_payload = {
                        "Student_ID": "STU_WEB",
                        "Gender": gender,
                        "Age": age,
                        "Branch": branch,
                        "Degree": degree,
                        "CGPA": cgpa,
                        "10th Percentage": tenth,
                        "12th Percentage": twelfth,
                        "Backlogs": backlogs,
                        "Attendance": attendance,
                        "Technical Skills": tech_skills,
                        "Programming Score": prog_score,
                        "Aptitude Score": aptitude,
                        "Communication Skills": comm,
                        "Soft Skills": soft_s,
                        "Projects Completed": projects,
                        "Internships": internships,
                        "Hackathons": hackathons,
                        "Certifications": certs,
                        "Workshops": workshops,
                        "Leadership Score": leadership,
                        "Resume Score": resume_s,
                        "Interview Score": interview_s,
                        "English Proficiency": english
                        # NOTE: Expected Salary is intentionally excluded from ML features
                        # because it is derived from the placement label (data leakage).
                    }

                    from api_client import predict_single_via_api
                    success, res, err = predict_single_via_api(req_payload)
                    
                    if success:
                        st.session_state.latest_prediction = res
                        feats = normalize_input_dict(req_payload)
                        feats["Expected Salary"] = expected_salary
                        st.session_state.latest_prediction_features = feats
                        for stale_key in ("shap_waterfall_path", "shap_force_path", "shap_explanation"):
                            st.session_state.pop(stale_key, None)
                        st.success("Prediction completed successfully via API.")
                    else:
                        st.warning(err)
                        
                        from src.predict import PlacementPredictor
                        predictor = PlacementPredictor()
                        clean_payload = normalize_input_dict(req_payload)
                        clean_payload["Expected Salary"] = expected_salary
                        res = predictor.predict_single(clean_payload)
                        st.session_state.latest_prediction = res
                        st.session_state.latest_prediction_features = clean_payload
                        for stale_key in ("shap_waterfall_path", "shap_force_path", "shap_explanation"):
                            st.session_state.pop(stale_key, None)
                        st.success("Prediction completed successfully via local fallback.")
                except Exception as local_err:
                    st.error(f"Prediction failed: {str(local_err)}")

        # Display results if available
        if "latest_prediction" in st.session_state:
            res = st.session_state.latest_prediction
            placed = res["placed"]
            prob = res["probability"]
            confidence = res["confidence"]
            readiness = res["readiness_score"]
            salary = res["expected_salary_range"]
            job_cat = res["job_category"]
            companies = res["recommended_companies"]
            eligible = res["eligible"]
            remarks = res["eligibility_remarks"]
            suggestions = res["suggestions"]

            st.write("---")
            st.subheader("Prediction Results Analysis")
            
            # Placement Gauge Meter Card
            col_gauge, col_elig = st.columns([1, 1])
            with col_gauge:
                card_style = "gauge-card" if placed else "danger-card"
                st.markdown(f"""
                <div class="{card_style}">
                    <h3 style="margin-top:0; color:#FFFFFF;">Prediction: {'PLACED' if placed else 'NOT PLACED'}</h3>
                    <p style="font-size:2.8rem; font-weight:800; margin:10px 0; color:#FFFFFF;">{prob}%</p>
                    <p style="margin-bottom:0; color:rgba(255,255,255,0.7); font-size:0.9rem;">Confidence: {confidence}%</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col_elig:
                elig_style = "gauge-card" if eligible else "danger-card"
                elig_label = "Academic Standing: Eligible" if eligible else "Academic Standing: Blocked"
                st.markdown(f"""
                <div class="{elig_style}">
                    <h4 style="margin-top:0; color:#FFFFFF;">{elig_label}</h4>
                    <p style="margin:10px 0; font-size:1rem; text-align:left; color:#FFFFFF;">{'<br>• '.join(remarks)}</p>
                </div>
                """, unsafe_allow_html=True)

            # Detail grids
            col_metrics, col_career = st.columns(2)
            with col_metrics:
                st.write("#### 📊 Placement Readiness Analytics")
                st.metric("Readiness Index (0-100)", f"{readiness}/100")
                st.progress(readiness/100.0)
                st.write(f"**Expected Offer Package:** {salary}")
                
            with col_career:
                st.write("#### 💼 Career Recommendations")
                st.write(f"**Target Role Group:** `{job_cat}`")
                st.write(f"**Target Companies:** {', '.join(companies)}")

            # Improvement Checklists
            st.write("#### 🛠 Personalized Action Plan to Improve Placements")
            for item in suggestions:
                if "CRITICAL" in item:
                    st.error(item)
                elif "HIGH" in item:
                    st.warning(item)
                else:
                    st.info(item)

            # Dynamic button to trigger SHAP explanation page
            if st.button("Generate SHAP Explanation for this Student Profile"):
                st.session_state.page = "🧠 Explain Prediction"
                st.session_state.auto_generate_explanation = True
                st.rerun()

    with tab_batch:
        st.subheader("CSV Batch Prediction Service")
        st.markdown("Upload a CSV file containing students details matching the numerical and categorical columns. The system will append predictions and return the annotated CSV.")
        
        batch_file = st.file_uploader("Choose batch CSV file", type=["csv"], key="batch_uploader")
        
        if batch_file is not None:
            df_batch = pandas_read_csv(batch_file)
            st.write(f"Loaded batch file containing {len(df_batch)} records.")
            
            missing_cols = [c for c in FEATURES if c not in df_batch.columns]
            if missing_cols:
                st.error(f"Error: Missing required columns in batch file: {missing_cols}")
                with st.expander("Show expected feature columns"):
                    st.write(FEATURES)
            else:
                df_batch = ensure_feature_columns(df_batch)
                st.dataframe(df_batch.head(5), use_container_width=True)
                
                if st.button("Process Batch Predictions", type="primary"):
                    with st.spinner("Processing bulk inferences..."):
                        try:
                            from api_client import predict_batch_via_api, download_file_via_api
                            
                            batch_file.seek(0)
                            file_content = batch_file.read()
                            
                            success, res, err = predict_batch_via_api(file_content, batch_file.name)
                            if success:
                                st.success("Batch predictions completed via API!")
                                pred_file = res.get("prediction_file")
                                csv_data = download_file_via_api(pred_file)
                                if csv_data:
                                    df_out = pd.read_csv(io.BytesIO(csv_data))
                                    st.dataframe(df_out.head(10), use_container_width=True)
                                    
                                    st.download_button(
                                        label="Download Predictions CSV File",
                                        data=csv_data,
                                        file_name="student_placement_predictions.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    st.error("Failed to download predictions CSV from API server.")
                            else:
                                st.warning(err)
                                
                                from src.predict import PlacementPredictor
                                local_pred = PlacementPredictor()
                                df_out = local_pred.predict_batch(df_batch)
                                
                                st.success("Batch predictions completed locally!")
                                st.dataframe(df_out.head(10), use_container_width=True)
                                
                                # Log batch predictions to SQLite database
                                from src.database import SessionLocal, PredictionRecord
                                import json
                                db = SessionLocal()
                                try:
                                    for _, row in df_out.iterrows():
                                        row_dict = row.to_dict()
                                        db_pred = PredictionRecord(
                                            student_id=str(row_dict.get("Student_ID", "BATCH_STU")),
                                            cgpa=float(row_dict.get("CGPA", 0.0)),
                                            branch=str(row_dict.get("Branch", "Computer Science")),
                                            attendance=float(row_dict.get("Attendance", 0.0)),
                                            programming_score=float(row_dict.get("Programming Score", 0.0)),
                                            aptitude_score=float(row_dict.get("Aptitude Score", 0.0)),
                                            interview_score=float(row_dict.get("Interview Score", 0.0)),
                                            resume_score=float(row_dict.get("Resume Score", 0.0)),
                                            internships=int(row_dict.get("Internships", 0)),
                                            projects_completed=int(row_dict.get("Projects Completed", 0)),
                                            certifications=int(row_dict.get("Certifications", 0)),
                                            predicted_placed=(row_dict.get("Predicted_Placement_Status") == "Placed"),
                                            placement_probability=float(row_dict.get("Placement_Probability_%", 0.0)),
                                            readiness_score=float(row_dict.get("Placement_Readiness_Score", 0.0)),
                                            expected_salary=float(row_dict.get("Expected Salary", 0.0)),
                                            input_features_json=json.dumps(row_dict)
                                        )
                                        db.add(db_pred)
                                    db.commit()
                                except Exception as bulk_db_err:
                                    st.warning(f"Error logging batch to database: {str(bulk_db_err)}")
                                finally:
                                    db.close()
                                
                                # Download button
                                csv_buffer = io.BytesIO()
                                df_out.to_csv(csv_buffer, index=False)
                                csv_data = csv_buffer.getvalue()
                                
                                st.download_button(
                                    label="Download Predictions CSV File",
                                    data=csv_data,
                                    file_name="student_placement_predictions.csv",
                                    mime="text/csv"
                                )
                        except Exception as e:
                            st.error(f"Failed processing batch file: {str(e)}")
