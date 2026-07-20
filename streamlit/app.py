import sys
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Student Placement Portal",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root so that `from src.X import Y` works in all page files
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Add the _pages dir so `from home import ...` etc. work
# (must come AFTER root so installed `streamlit` lib is not shadowed at root level)
# NOTE: named _pages (not pages) so Streamlit's auto-discovery doesn't create
# duplicate sidebar entries for each file.
_pages_dir = str(Path(__file__).resolve().parent / "_pages")
if _pages_dir not in sys.path:
    sys.path.insert(0, _pages_dir)

from home import render_home_page
from upload import render_upload_page
from analysis import render_analysis_page
from training import render_training_page
from prediction import render_prediction_page
from explanation import render_explanation_page
from settings import render_settings_page

def load_css():
    css_path = Path(__file__).resolve().parent / "css" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h2 style="color: #FFFFFF; font-weight: 800; font-size: 1.5rem; letter-spacing: -0.05em; margin-bottom: 4px;">Placement Portal</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.subheader("Navigation")
menu_selection = st.sidebar.radio(
    label="Go to page",
    options=[
        "🏠 Home",
        "📂 Upload Dataset",
        "📊 Data Analysis",
        "🤖 Train Models",
        "🎯 Predict Placement",
        "🧠 Explain Prediction",
        "⚙ Settings"
    ],
    label_visibility="collapsed"
)

if menu_selection != st.session_state.page:
    st.session_state.page = menu_selection

active_page = st.session_state.page

if active_page == "🏠 Home":
    render_home_page()
elif active_page == "📂 Upload Dataset":
    render_upload_page()
elif active_page == "📊 Data Analysis":
    render_analysis_page()
elif active_page == "🤖 Train Models":
    render_training_page()
elif active_page == "🎯 Predict Placement":
    render_prediction_page()
elif active_page == "🧠 Explain Prediction":
    render_explanation_page()
elif active_page == "⚙ Settings":
    render_settings_page()

st.sidebar.write("---")
st.sidebar.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.8rem; font-weight: 500;">
    Student Placement Portal<br>
    © 2026 All Rights Reserved
</div>
""", unsafe_allow_html=True)

