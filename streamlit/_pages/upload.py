import streamlit as st
import plotly.express as px
from src.config import DATA_RAW_DIR, TARGET_COL, FEATURES
from src.utils import pandas_read_csv

def render_upload_page():
    """Render the dataset upload page."""
    st.markdown('<h1 class="main-title">Upload Dataset</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload a student placement CSV to update the training data.</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            df = pandas_read_csv(uploaded_file)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns successfully.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.subheader("Data Preview")
            st.dataframe(df.head(8), use_container_width=True)

            if TARGET_COL in df.columns:
                left_col, right_col = st.columns([1, 1])
                with left_col:
                    st.subheader("Placement Distribution (Class Balance)")
                    counts = df[TARGET_COL].value_counts().reset_index()
                    counts.columns = [TARGET_COL, "Count"]
                    counts[TARGET_COL] = counts[TARGET_COL].map({1: "Placed", 0: "Not Placed"}).fillna(counts[TARGET_COL].astype(str))
                    fig = px.bar(
                        counts, x=TARGET_COL, y="Count", 
                        color=TARGET_COL,
                        color_discrete_map={"Placed": "#10B981", "Not Placed": "#F43F5E"},
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with right_col:
                    st.subheader("Branch Representation")
                    if "Branch" in df.columns:
                        branch_counts = df["Branch"].value_counts().reset_index()
                        fig = px.pie(
                            branch_counts, names="Branch", values="count" if "count" in branch_counts.columns else branch_counts.iloc[:, 1],
                            template="plotly_dark", hole=0.3
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                missing_cols = [TARGET_COL] if TARGET_COL not in df.columns else []
                st.warning(f"Warning: The uploaded file does not contain the required target column '{TARGET_COL}'. It can be used for batch predictions, but not for model training.")
                with st.expander("Show expected dataset columns"):
                    st.write(FEATURES + [TARGET_COL])
                    if missing_cols:
                        st.warning(f"Missing required training target: {missing_cols}")

            st.write("---")
            if st.button("Save to Dataset", type="primary", use_container_width=True):
                uploaded_file.seek(0)
                with st.spinner("Saving dataset to local raw folder..."):
                    try:
                        uploaded_file.seek(0)
                        with open(DATA_RAW_DIR / "placement_dataset.csv", "wb") as f:
                            f.write(uploaded_file.read())

                        st.success("Saved active dataset to data/raw/placement_dataset.csv.")
                        if TARGET_COL in df.columns:
                            st.info("This upload contains the training target and is ready for model training.")
                        else:
                            st.info("This upload does not contain the target column and is best used for batch prediction only.")
                    except Exception as write_err:
                        st.error(f"Failed to save uploaded dataset: {write_err}")
                        
        except Exception as e:
            st.error(f"Error parsing uploaded CSV file: {str(e)}")
