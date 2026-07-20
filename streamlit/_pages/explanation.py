import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import REPORTS_DIR, DATA_RAW_DIR, MODELS_DIR, CATEGORICAL_COLS
from src.explain import get_explainer, generate_local_shap_explanation, generate_global_shap_plots
from src.utils import load_pickle, ensure_feature_columns, DEFAULT_FEATURE_VALUES


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_feature_names(preprocessor) -> list:
    cat_encoder = preprocessor.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_features_ohe = list(
        cat_encoder.get_feature_names_out(
            preprocessor.named_steps['preprocessor'].transformers_[1][2]
        )
    )
    engineered_numerical_cols = preprocessor.named_steps['preprocessor'].transformers_[0][2]
    return engineered_numerical_cols + cat_features_ohe


def _load_background_data(preprocessor):
    raw_csv = DATA_RAW_DIR / "placement_dataset.csv"
    df_raw = pd.read_csv(str(raw_csv))
    df_sample = df_raw.sample(n=min(100, len(df_raw)), random_state=42)
    
    # Dynamically determine features that the preprocessor pipeline needs
    col_trans = preprocessor.named_steps['preprocessor']
    num_cols_configured = col_trans.transformers[0][2]
    cat_cols_configured = col_trans.transformers[1][2]
    
    # We want to select the columns from df_sample that are expected by the preprocessor's first step.
    # The preprocessor pipeline is: cleaner -> feature_engineer -> preprocessor (ColumnTransformer)
    # The cleaner doesn't filter columns.
    # The feature_engineer expects raw numerical inputs to create the engineered inputs.
    engineered_cols = {"Academic_Performance_Index", "Employability_Readiness_Score", "Skills_Diversity_Index"}
    raw_num_cols = [c for c in num_cols_configured if c not in engineered_cols]
    
    feature_engineer_inputs = [
        "CGPA", "10th Percentage", "12th Percentage", 
        "Programming Score", "Aptitude Score", "Interview Score", 
        "Resume Score", "Internships", "Projects Completed", 
        "Certifications", "Workshops", "Hackathons"
    ]
    
    required_raw_cols = list(set(raw_num_cols + feature_engineer_inputs + cat_cols_configured))
    
    df_X = pd.DataFrame()
    for col in required_raw_cols:
        if col in df_sample.columns:
            df_X[col] = df_sample[col]
        else:
            df_X[col] = DEFAULT_FEATURE_VALUES.get(col, 0 if col not in CATEGORICAL_COLS else "Male")
            
    df_proc = preprocessor.transform(df_X)
    X_bg_proc = df_proc.values if hasattr(df_proc, "values") else df_proc
    return X_bg_proc


def _run_local_explanation(model_path, preprocessor_path) -> None:
    model = load_pickle(model_path)
    preprocessor = load_pickle(preprocessor_path)

    res = st.session_state.latest_prediction
    processed_input = np.array(res["processed_input"]).reshape(1, -1)
    X_bg_proc = _load_background_data(preprocessor)
    feature_names = _get_feature_names(preprocessor)
    explainer, exp_type = get_explainer(model, X_bg_proc)

    raw_feat_df = pd.DataFrame([st.session_state.latest_prediction_features])
    waterfall_img, force_html, explanation = generate_local_shap_explanation(
        explainer, processed_input, raw_feat_df, feature_names, exp_type
    )

    st.session_state.shap_waterfall_path = waterfall_img
    st.session_state.shap_force_path = force_html
    st.session_state.shap_explanation = explanation


# ─────────────────────────────────────────────────────────────────────────────
# Factor card renderers
# ─────────────────────────────────────────────────────────────────────────────

def _impact_bar_html(impact_pct: float, max_impact: float = 20.0) -> str:
    """Return an inline HTML progress-bar for impact percentage."""
    width = min(abs(impact_pct) / max_impact * 100, 100)
    color = "#22c55e" if impact_pct > 0 else "#ef4444"
    return (
        f'<div style="background:#1e293b;border-radius:6px;height:8px;width:100%;margin-top:4px;">'
        f'<div style="background:{color};border-radius:6px;height:8px;width:{width:.1f}%;"></div>'
        f'</div>'
    )


def _render_factor_card(factor: dict, positive: bool) -> None:
    """Render a single elaborated factor explanation card."""
    impact     = factor["impact_pct"]
    label      = factor["label"]
    value      = factor["student_value"]
    direction  = factor["direction"]
    sign       = "+" if impact > 0 else ""

    if positive:
        border_color = "#22c55e"
        icon = "✅"
        badge_bg = "rgba(34,197,94,0.15)"
        impact_color = "#22c55e"
        verdict = "This factor is **working in your favour**."
    else:
        border_color = "#ef4444"
        icon = "⚠️"
        badge_bg = "rgba(239,68,68,0.15)"
        impact_color = "#ef4444"
        verdict = "This factor is **holding you back**."

    bar_html = _impact_bar_html(impact)

    st.markdown(
        f"""
        <div style="
            border-left: 4px solid {border_color};
            background: {badge_bg};
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 12px;
        ">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:1rem;font-weight:700;color:#f1f5f9;">{icon} {label}</span>
                <span style="font-size:1.1rem;font-weight:800;color:{impact_color};">{sign}{impact:.1f}%</span>
            </div>
            <div style="margin-top:4px;font-size:0.82rem;color:#94a3b8;">
                Your value: <strong style="color:#e2e8f0;">{value}</strong>
            </div>
            {bar_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_factor_section(title: str, factors: list, positive: bool) -> None:
    st.markdown(f"#### {title}")
    if not factors:
        st.caption("No notable factors in this category for this student.")
        return
    for factor in factors:
        _render_factor_card(factor, positive)


# ─────────────────────────────────────────────────────────────────────────────
# Narrative builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_narrative(explanation: dict, res: dict) -> str:
    """
    Build a paragraph-style plain-English story about the prediction
    that a student can easily read and act on.
    """
    placed         = res["placed"]
    prob           = res["probability"]
    readiness      = res["readiness_score"]
    salary         = res["expected_salary_range"]
    job_cat        = res["job_category"]
    base_pct       = explanation["base_pct"]
    output_pct     = explanation["output_pct"]
    net_change     = explanation["net_change_pct"]
    pos_factors    = explanation.get("positive_factors", [])
    neg_factors    = explanation.get("negative_factors", [])

    # Outcome sentence
    if placed:
        outcome_sent = (
            f"The model predicts that **this student is likely to be placed**, "
            f"with a placement probability of **{prob}%**."
        )
    else:
        outcome_sent = (
            f"The model predicts that **this student may struggle to secure a placement**, "
            f"with only a **{prob}%** placement probability."
        )

    # Comparison with average
    if net_change >= 0:
        compare_sent = (
            f"Compared to the average student (baseline: **{base_pct}%**), "
            f"this profile scores **{abs(net_change):.1f} percentage points higher**."
        )
    else:
        compare_sent = (
            f"Compared to the average student (baseline: **{base_pct}%**), "
            f"this profile scores **{abs(net_change):.1f} percentage points lower**, "
            f"meaning the profile is below average in placement likelihood."
        )

    # Top strength
    strength_sent = ""
    if pos_factors:
        top = pos_factors[0]
        strength_sent = (
            f"The biggest strength here is **{top['label']}** "
            f"(your value: `{top['student_value']}`), which alone boosts placement "
            f"probability by **+{top['impact_pct']:.1f}%**."
        )

    # Top weakness
    weakness_sent = ""
    if neg_factors:
        top_neg = neg_factors[0]
        weakness_sent = (
            f"The most significant area for improvement is **{top_neg['label']}** "
            f"(your value: `{top_neg['student_value']}`), which reduces placement "
            f"probability by **{top_neg['impact_pct']:.1f}%**."
        )

    # Career outlook
    career_sent = (
        f"Based on this profile, the most suitable target role is "
        f"**{job_cat}**, with an estimated salary package of **{salary}**. "
        f"The overall Placement Readiness Score is **{readiness}/100**."
    )

    parts = [outcome_sent, compare_sent]
    if strength_sent:
        parts.append(strength_sent)
    if weakness_sent:
        parts.append(weakness_sent)
    parts.append(career_sent)

    return "  \n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# "What does this score mean?" guide
# ─────────────────────────────────────────────────────────────────────────────

def _render_score_guide(output_pct: float) -> None:
    st.markdown("#### 🧭 How to Interpret Your Score")
    tiers = [
        (80, 100, "#22c55e", "🏆 Strong Placement Candidate",
         "Your profile is well above average. You are a strong candidate for campus placements. "
         "Focus on mock interviews and company-specific preparation."),
        (60,  80, "#84cc16", "✅ Good Placement Prospect",
         "Your profile is competitive. With a few targeted improvements (internships, projects, "
         "aptitude practice), you can significantly raise your chances."),
        (40,  60, "#f59e0b", "⚡ Average — Needs Improvement",
         "You are in the borderline zone. Focused effort on your weakest areas can push you "
         "into the 'placed' category. Prioritise your red flags below."),
        ( 0,  40, "#ef4444", "❌ At-Risk — Action Required",
         "Your placement probability is low. Immediate action on backlogs, attendance, "
         "programming skills, and communication is critical."),
    ]
    for lo, hi, color, label, desc in tiers:
        is_active = lo <= output_pct < hi or (hi == 100 and output_pct == 100)
        bg = f"rgba(255,255,255,0.06)" if is_active else "transparent"
        border = f"2px solid {color}" if is_active else f"1px solid rgba(255,255,255,0.08)"
        glow = f"box-shadow: 0 0 8px {color}44;" if is_active else ""
        marker = " ← **You are here**" if is_active else ""
        st.markdown(
            f"""<div style="border:{border};{glow}background:{bg};border-radius:8px;
                           padding:10px 16px;margin-bottom:8px;">
                    <span style="color:{color};font-weight:700;">{label}</span>
                    <span style="color:#94a3b8;font-size:0.8rem;"> ({lo}%–{hi}%){marker}</span>
                    <br><span style="color:#cbd5e1;font-size:0.85rem;">{desc}</span>
                </div>""",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Action plan from negative factors
# ─────────────────────────────────────────────────────────────────────────────

_IMPROVEMENT_TIPS = {
    "Programming Score": (
        "Practice DSA problems daily on **LeetCode / HackerRank** (aim for 2–3 problems/day). "
        "Focus on arrays, linked lists, trees, and dynamic programming. "
        "Reach a score of 70+ to unlock Tier-2 and Tier-1 company opportunities."
    ),
    "Aptitude Score": (
        "Use **IndiaBix** or **RS Aggarwal** for quantitative aptitude. "
        "Practice at least 30 minutes daily. Logical reasoning and verbal sections are equally important."
    ),
    "Interview Score": (
        "Do **at least 3 mock interviews per week** (use Pramp, Interviewing.io, or ask friends). "
        "Practise explaining your projects in 2 minutes. Work on STAR-format answers for HR questions."
    ),
    "Resume Score": (
        "Keep your resume to **1 page maximum**. Use action verbs (built, designed, optimised). "
        "Highlight quantifiable achievements. Use a clean ATS-friendly template from **Overleaf** or **Canva**."
    ),
    "Communication Skills": (
        "Join a **Toastmasters** club or do daily English speaking practice. "
        "Record yourself answering HR questions and review the playback. "
        "Read English news for 15 minutes daily to improve vocabulary."
    ),
    "Internships": (
        "Apply to **Internshala**, **LinkedIn**, and **Unstop** for internships NOW — "
        "even 1–2 month virtual internships count. A single real-world experience dramatically boosts your profile."
    ),
    "Projects Completed": (
        "Start a project this week. Ideas: a web app, ML model, or automation script. "
        "Upload to **GitHub** with a clear README. Aim for at least 2 substantial projects before placements."
    ),
    "Certifications": (
        "Earn at least one recognised certification: **Google**, **AWS**, **Microsoft**, or "
        "**NPTEL** courses. Free options include Coursera audit mode and Google's free certificates."
    ),
    "CGPA": (
        "A CGPA above **7.0** opens doors to most companies; above **8.0** unlocks Tier-1 companies. "
        "Focus on your upcoming semester exams — every 0.1 improvement matters."
    ),
    "Attendance": (
        "Many companies and college placement cells require a **minimum 75% attendance**. "
        "If you are below this, you may be blocked from sitting for campus drives entirely."
    ),
    "Backlogs": (
        "**Clear all backlogs immediately**. Most reputed companies have a strict zero-backlog policy. "
        "Contact your academic office about supplementary exams."
    ),
}


def _render_action_plan(neg_factors: list) -> None:
    if not neg_factors:
        st.success(
            "🎉 No major weaknesses detected! Your profile is well-rounded. "
            "Focus on company-specific preparation and interview practice."
        )
        return

    st.markdown("#### 🗺️ Your Personalised Action Plan")
    st.markdown(
        "These are the areas where improving will have the **biggest positive impact** "
        "on your placement probability, listed in order of importance:"
    )

    for i, factor in enumerate(neg_factors, 1):
        label   = factor["label"]
        value   = factor["student_value"]
        impact  = factor["impact_pct"]

        # Try to match a tip
        tip = None
        for key, tip_text in _IMPROVEMENT_TIPS.items():
            if key.lower() in label.lower() or label.lower() in key.lower():
                tip = tip_text
                break

        with st.expander(
            f"#{i}  ·  {label}  —  current value: `{value}`  ·  impact: {impact:.1f}%",
            expanded=(i == 1),
        ):
            if tip:
                st.markdown(f"**What to do:** {tip}")
            else:
                st.markdown(
                    f"Work on improving **{label}** from your current value of `{value}`. "
                    "Consult your mentor or placement coordinator for specific advice."
                )
            st.markdown(
                f"📈 *Improving this factor could raise your placement probability by up to "
                f"**{abs(impact):.1f} percentage points**.*"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────────────────────

def render_explanation_page():
    """Renders elaborated SHAP explainers with plain-language narratives."""
    st.markdown('<h1 class="main-title">Explain My Prediction</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">'
        "Understand <em>exactly</em> why the model predicted your result — "
        "and what you can do about it."
        "</p>",
        unsafe_allow_html=True,
    )

    model_path        = MODELS_DIR / "placement_model.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    if not (model_path.exists() and preprocessor_path.exists()):
        st.warning("⚠️ Model artifacts not found. Run training first to generate explanations.")
        return

    tab_local, tab_global = st.tabs(
        ["🎯 My Prediction Explained", "🌍 Overall Feature Importance"]
    )

    # ── LOCAL EXPLANATION TAB ───────────────────────────────────────────────
    with tab_local:
        if "latest_prediction" not in st.session_state:
            st.info(
                "💡 **No prediction yet.**  \n"
                "Go to **Predict Placement**, run a prediction for a student profile, "
                "then come back here to see a detailed explanation."
            )
            return

        res    = st.session_state.latest_prediction
        placed = res["placed"]
        prob   = res["probability"]

        # ── Hero banner ─────────────────────────────────────────────────────
        if placed:
            banner_bg    = "linear-gradient(135deg, #14532d 0%, #166534 100%)"
            banner_icon  = "🎓"
            banner_label = "PLACED"
            banner_sub   = f"Placement Probability: {prob}%"
        else:
            banner_bg    = "linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%)"
            banner_icon  = "📋"
            banner_label = "NOT PLACED"
            banner_sub   = f"Placement Probability: {prob}%  ·  See the action plan below"

        st.markdown(
            f"""
            <div style="
                background:{banner_bg};
                border-radius:14px;
                padding:28px 36px;
                margin-bottom:24px;
                text-align:center;
            ">
                <div style="font-size:3rem;">{banner_icon}</div>
                <div style="font-size:2rem;font-weight:900;color:#ffffff;letter-spacing:2px;">
                    {banner_label}
                </div>
                <div style="font-size:1rem;color:rgba(255,255,255,0.75);margin-top:6px;">
                    {banner_sub}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Generate button ──────────────────────────────────────────────────
        auto_run = st.session_state.pop("auto_generate_explanation", False)
        if auto_run or st.button("🔍 Generate Detailed Explanation", type="primary"):
            with st.spinner(
                "Analysing which factors helped or hurt this prediction — this may take 10–30 seconds…"
            ):
                try:
                    _run_local_explanation(model_path, preprocessor_path)
                    st.success("✅ Explanation ready! Scroll down to read it.")
                except Exception as shap_err:
                    st.error(f"Failed to generate explanation: {shap_err}")

        explanation = st.session_state.get("shap_explanation")
        if not explanation or "error" in explanation:
            if explanation and "error" in explanation:
                st.error(f"Explanation error: {explanation['error']}")
            return

        # ── What does this mean? Guide ────────────────────────────────────────
        st.write("---")
        _render_score_guide(explanation["output_pct"])

        # ── Plain-English narrative ──────────────────────────────────────────
        st.write("---")
        st.markdown("#### 📖 Plain-English Summary")

        narrative = _build_narrative(explanation, res)
        st.info(narrative)

        # ── Key metrics row ──────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "📊 Placement Probability",
            f"{explanation['output_pct']}%",
            help="The model's predicted probability that this student will be placed.",
        )
        m2.metric(
            "📈 vs. Average Student",
            f"{explanation['net_change_pct']:+.1f} pts",
            help=f"Average baseline is {explanation['base_pct']}%. Positive = above average.",
        )
        m3.metric(
            "🏃 Readiness Score",
            f"{res['readiness_score']}/100",
            help="Composite employability readiness index (0–100).",
        )
        m4.metric(
            "💰 Est. Salary Range",
            res["expected_salary_range"],
            help="Estimated CTC package based on current profile.",
        )

        # ── What helped / what hurt ──────────────────────────────────────────
        st.write("---")
        st.markdown(
            """
            #### 🔬 Factor-by-Factor Breakdown

            Each bar below shows how much a specific factor **pushed your placement probability
            up (✅ green)** or **pulled it down (⚠️ red)** compared to an average student.
            The wider the bar, the stronger the influence.
            """
        )

        col_pos, col_neg = st.columns(2)
        with col_pos:
            _render_factor_section(
                "✅ Factors Working in Your Favour",
                explanation.get("positive_factors", []),
                positive=True,
            )
        with col_neg:
            _render_factor_section(
                "⚠️ Factors Pulling Your Score Down",
                explanation.get("negative_factors", []),
                positive=False,
            )

        # ── Contextual reading guide ─────────────────────────────────────────
        st.markdown(
            """
            > **How to read the impact numbers:**  
            > A value like **+8.5%** means that factor raised your placement probability by  
            > 8.5 percentage points above the average. **−6.2%** means it lowered it by 6.2 points.  
            > These are **additive shifts** relative to the average student baseline of
            > **{base}%**.
            """.format(base=explanation["base_pct"])
        )

        # ── Action Plan ──────────────────────────────────────────────────────
        st.write("---")
        _render_action_plan(explanation.get("negative_factors", []))

        # ── Visual SHAP Charts ───────────────────────────────────────────────
        if "shap_waterfall_path" in st.session_state:
            st.write("---")
            st.markdown("#### 📊 Visual Explanation Charts")

            wt_col, guide_col = st.columns([5, 3])
            with wt_col:
                wt_path = Path(st.session_state.shap_waterfall_path)
                if wt_path.exists():
                    st.image(
                        str(wt_path),
                        caption="SHAP Waterfall Chart — step-by-step impact of each factor on placement probability",
                        use_container_width=True,
                    )

            with guide_col:
                st.markdown("##### 📖 How to Read This Chart")
                st.markdown(
                    """
                    The waterfall chart starts from the **average student baseline** (E[f(x)])
                    and adds or subtracts each factor's contribution one by one until it reaches
                    **your final predicted probability** (f(x)).

                    | Colour | Meaning |
                    |--------|---------|
                    | 🟥 Red bar | This factor **increases** your placement probability |
                    | 🟦 Blue bar | This factor **decreases** your placement probability |
                    | Width | How **strong** the factor's influence is |

                    > **Tip:** The longer the bar, the more important that feature is for
                    > your specific result.
                    """
                )

            st.write("---")
            st.markdown("##### 🎛️ Interactive Force Plot")
            st.markdown(
                "The force plot is an interactive view that shows all factors at once. "
                "Factors in **red push the prediction higher** (toward Placed); "
                "factors in **blue push it lower** (toward Not Placed). "
                "You can hover over the plot to see exact values."
            )
            force_path = Path(st.session_state.shap_force_path)
            if force_path.exists():
                with open(force_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                components.html(html_content, height=200, scrolling=True)
            else:
                st.info("Interactive force plot not available.")

    # ── GLOBAL EXPLANATION TAB ──────────────────────────────────────────────
    with tab_global:
        st.markdown("#### 🌍 What Matters Most — Across All Students")
        st.markdown(
            """
            This chart shows which features have the **biggest overall influence** on
            placement predictions across the entire student dataset — not just for one student.

            It answers the question: *"If I could improve just one thing to maximise
            my chances, what would it be?"*
            """
        )

        if st.button("Generate / Refresh Global Feature Importance", type="primary"):
            with st.spinner("Computing global SHAP values — please wait…"):
                try:
                    model        = load_pickle(model_path)
                    preprocessor = load_pickle(preprocessor_path)
                    X_bg_proc    = _load_background_data(preprocessor)
                    feature_names = _get_feature_names(preprocessor)
                    explainer, exp_type = get_explainer(model, X_bg_proc)
                    generate_global_shap_plots(explainer, X_bg_proc[:150], feature_names, exp_type)
                    st.success("✅ Global summary plot generated!")
                except Exception as glob_err:
                    st.error(f"Failed to generate global summary: {glob_err}")

        glob_path = REPORTS_DIR / "shap_summary.png"
        if glob_path.exists():
            st.image(
                str(glob_path),
                caption="Global Feature Importance — SHAP Summary Plot",
                use_container_width=True,
            )
            st.markdown(
                """
                #### 📖 How to Read the Global Chart

                | Element | Meaning |
                |---------|---------|
                | **Feature position (top → bottom)** | Features at the **top** have the strongest overall impact |
                | **Red dots** | Student has a **high value** for that feature |
                | **Blue dots** | Student has a **low value** for that feature |
                | **Dot position (left ↔ right)** | Dots further **right** → increases placement probability; further **left** → decreases it |
                | **Dot spread** | A wide horizontal spread means the feature has **variable impact** across students |

                #### 💡 Key Insight
                Features like **Programming Score**, **CGPA**, **Interview Score**, and **Internships**
                typically appear near the top — meaning they are the most powerful levers you can pull
                to improve your placement outcome.
                """
            )
        else:
            st.info("Global SHAP plot not generated yet. Click the button above to create it.")
