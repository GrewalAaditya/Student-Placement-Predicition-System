import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple
from src.config import REPORTS_DIR
from src.utils import logger

ENGINEERED_FEATURE_LABELS = {
    "Academic_Performance_Index": "Overall Academic Performance (CGPA + school scores combined)",
    "Employability_Readiness_Score": "Employability Readiness (skills, projects, internships combined)",
    "Skills_Diversity_Index": "Skills & Activities Diversity (certs + workshops + hackathons)",
}

OHE_FIELD_PREFIXES = ["Gender", "Branch", "Degree", "English Proficiency"]


def humanize_feature_name(feature_name: str) -> str:
    """Convert encoded model feature names into readable labels."""
    if feature_name in ENGINEERED_FEATURE_LABELS:
        return ENGINEERED_FEATURE_LABELS[feature_name]

    for prefix in OHE_FIELD_PREFIXES:
        token = prefix.replace(" ", "_")
        if feature_name.startswith(f"{token}_"):
            value = feature_name[len(token) + 1:].replace("_", " ")
            return f"{prefix}: {value}"
        if feature_name.startswith(f"{prefix}_"):
            value = feature_name[len(prefix) + 1:].replace("_", " ")
            return f"{prefix}: {value}"

    return feature_name.replace("_", " ")


def _student_value_for_feature(feature_name: str, raw_features: Dict[str, Any]) -> str:
    """Return a display string for the student's value on a given feature."""
    for prefix in OHE_FIELD_PREFIXES:
        token = prefix.replace(" ", "_")
        if feature_name.startswith(f"{token}_") or feature_name.startswith(f"{prefix}_"):
            return str(raw_features.get(prefix, "—"))

    if feature_name in raw_features:
        value = raw_features[feature_name]
        if isinstance(value, float):
            if value == int(value):
                return str(int(value))
            return f"{value:.1f}"
        return str(value)

    return "—"


def _feature_applies_to_student(feature_name: str, raw_features: Dict[str, Any]) -> bool:
    """Check whether a one-hot encoded feature matches this student's profile."""
    for prefix in OHE_FIELD_PREFIXES:
        token = prefix.replace(" ", "_")
        for lead in (f"{token}_", f"{prefix}_"):
            if feature_name.startswith(lead):
                encoded_value = feature_name[len(lead):].replace("_", " ")
                student_value = str(raw_features.get(prefix, "")).strip()
                return encoded_value.lower() == student_value.lower()
    return True


def build_plain_language_explanation(
    shap_values: np.ndarray,
    feature_names: List[str],
    raw_features: Dict[str, Any],
    base_value: float,
    output_value: float,
    placed: bool,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Build an easy-to-read explanation from SHAP values and the student's profile."""
    contributions: List[Dict[str, Any]] = []

    for name, impact in zip(feature_names, shap_values):
        if not _feature_applies_to_student(name, raw_features):
            continue

        impact_pct = round(float(impact) * 100, 1)
        if abs(impact_pct) < 0.05:
            continue

        contributions.append({
            "feature": name,
            "label": humanize_feature_name(name),
            "student_value": _student_value_for_feature(name, raw_features),
            "impact_pct": impact_pct,
            "direction": "helps" if impact_pct > 0 else "hurts",
        })

    contributions.sort(key=lambda item: abs(item["impact_pct"]), reverse=True)

    positive = [c for c in contributions if c["impact_pct"] > 0][:top_n]
    negative = [c for c in contributions if c["impact_pct"] < 0][:top_n]

    base_pct = round(float(base_value) * 100, 1)
    output_pct = round(float(output_value) * 100, 1)
    net_change = round(output_pct - base_pct, 1)

    if placed:
        headline = (
            f"This student is predicted **PLACED** with **{output_pct}%** placement probability."
        )
    else:
        headline = (
            f"This student is predicted **NOT PLACED** with only **{output_pct}%** placement probability."
        )

    if net_change >= 0:
        summary = (
            f"Compared to the average student baseline of **{base_pct}%**, this profile is "
            f"**{abs(net_change):.1f} percentage points higher**. "
        )
    else:
        summary = (
            f"Compared to the average student baseline of **{base_pct}%**, this profile is "
            f"**{abs(net_change):.1f} percentage points lower**. "
        )

    if positive and negative:
        summary += (
            f"The strongest positive drivers are **{positive[0]['label']}** "
            f"({positive[0]['impact_pct']:+.1f}%), while the biggest concerns are "
            f"**{negative[0]['label']}** ({negative[0]['impact_pct']:+.1f}%)."
        )
    elif positive:
        summary += (
            f"The profile is mainly supported by **{positive[0]['label']}** "
            f"({positive[0]['impact_pct']:+.1f}%)."
        )
    elif negative:
        summary += (
            f"The result is mainly pulled down by **{negative[0]['label']}** "
            f"({negative[0]['impact_pct']:+.1f}%)."
        )
    else:
        summary += "Feature impacts are relatively balanced for this profile."

    return {
        "headline": headline,
        "summary": summary,
        "base_pct": base_pct,
        "output_pct": output_pct,
        "net_change_pct": net_change,
        "positive_factors": positive,
        "negative_factors": negative,
        "all_factors": contributions[:top_n * 2],
    }


def get_explainer(model: Any, X_train_proc: np.ndarray) -> Tuple[Any, str]:
    """Get the appropriate SHAP explainer for the model type."""
    model_name = model.__class__.__name__
    logger.info(f"Setting up SHAP explainer for: {model_name}")

    tree_models = [
        "RandomForestClassifier", "XGBClassifier", "LGBMClassifier",
        "CatBoostClassifier", "GradientBoostingClassifier",
        "ExtraTreesClassifier", "DecisionTreeClassifier"
    ]

    try:
        if model_name in tree_models:
            try:
                explainer = shap.TreeExplainer(model, data=X_train_proc, model_output="probability")
                logger.info("Successfully built SHAP TreeExplainer with model_output='probability'")
                return explainer, "tree_probability"
            except Exception as tree_err:
                logger.warning(
                    f"Could not build probability TreeExplainer: {tree_err}. "
                    "Falling back to default margin-space TreeExplainer."
                )
                explainer = shap.TreeExplainer(model)
                return explainer, "tree"
        elif model_name in ["LogisticRegression", "LinearSVC"]:
            logger.info("Using KernelExplainer for LogisticRegression/LinearSVC.")
            background = shap.sample(X_train_proc, 50) if len(X_train_proc) > 50 else X_train_proc
            pred_func = model.predict_proba if hasattr(model, "predict_proba") else model.predict
            explainer = shap.KernelExplainer(pred_func, background)
            return explainer, "kernel"
        else:
            background = shap.sample(X_train_proc, 50) if len(X_train_proc) > 50 else X_train_proc
            pred_func = model.predict_proba if hasattr(model, "predict_proba") else model.predict
            explainer = shap.KernelExplainer(pred_func, background)
            return explainer, "kernel"
    except Exception as e:
        logger.warning(f"Error creating SHAP explainer, falling back to Explainer: {e}")
        explainer = shap.Explainer(model, X_train_proc)
        return explainer, "general"


def generate_global_shap_plots(
    explainer: Any,
    X_test_proc: np.ndarray,
    feature_names: list,
    explainer_type: str,
) -> None:
    """Generate and save global SHAP summary plot."""
    try:
        plt.figure(figsize=(10, 6))

        if explainer_type == "kernel":
            shap_values = explainer.shap_values(shap.sample(X_test_proc, 50))
            X_plot = shap.sample(X_test_proc, 50)
        else:
            shap_values = explainer.shap_values(X_test_proc)
            X_plot = X_test_proc

        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_vals_to_plot = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_vals_to_plot = shap_values[:, :, 1]
        else:
            shap_vals_to_plot = shap_values

        df_plot = pd.DataFrame(X_plot, columns=feature_names)
        shap.summary_plot(shap_vals_to_plot, df_plot, show=False)
        plt.title("SHAP Global Feature Importance", fontsize=14, pad=15)
        plt.tight_layout()

        filepath = REPORTS_DIR / "shap_summary.png"
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved global SHAP summary: {filepath}")
    except Exception as e:
        logger.error(f"Error generating global SHAP plots: {e}")


def generate_local_shap_explanation(
    explainer: Any,
    sample_proc: np.ndarray,
    raw_features_df: pd.DataFrame,
    feature_names: list,
    explainer_type: str,
) -> Tuple[str, str, Dict[str, Any]]:
    """Generate local waterfall and force plots plus a plain-language explanation."""
    waterfall_path = REPORTS_DIR / "shap_waterfall.png"
    force_html_path = REPORTS_DIR / "shap_force.html"
    explanation: Dict[str, Any] = {}

    try:
        shap_values = explainer(sample_proc)

        if len(shap_values.shape) > 1 and shap_values.shape[-1] == 2:
            local_shap = shap_values[0, :, 1]
        elif len(shap_values.shape) == 3:
            local_shap = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2:
            local_shap = shap_values[0]
        else:
            local_shap = shap_values

        local_shap.feature_names = feature_names

        base_val = local_shap.base_values
        if isinstance(base_val, np.ndarray):
            base_val = float(base_val[0] if len(base_val) > 0 else base_val)
        else:
            base_val = float(base_val)

        vals = np.array(local_shap.values).ravel()
        output_val = base_val + float(np.sum(vals))

        raw_features = raw_features_df.iloc[0].to_dict()
        explanation = build_plain_language_explanation(
            shap_values=vals,
            feature_names=feature_names,
            raw_features=raw_features,
            base_value=base_val,
            output_value=output_val,
            placed=output_val >= 0.5,
        )

        plt.figure(figsize=(9, 5))
        shap.plots.waterfall(local_shap, show=False)
        plt.title("Local Prediction Explanation (SHAP Waterfall)", fontsize=12, pad=10)
        plt.tight_layout()
        plt.savefig(waterfall_path, dpi=250)
        plt.close()
        logger.info(f"Saved SHAP waterfall: {waterfall_path}")

        features_array = np.array(sample_proc).ravel()
        html_force = shap.plots.force(
            base_value=base_val,
            shap_values=vals,
            features=features_array,
            feature_names=feature_names,
            matplotlib=False,
        )
        shap.save_html(str(force_html_path), html_force)
        logger.info(f"Saved SHAP force plot: {force_html_path}")

    except Exception as e:
        logger.error(f"Error generating local SHAP explanation: {e}")
        plt.figure()
        plt.text(0.5, 0.5, f"SHAP explanation failed:\n{e}", ha='center', va='center')
        plt.savefig(waterfall_path, dpi=100)
        plt.close()
        with open(force_html_path, "w", encoding="utf-8") as f:
            f.write(f"<h3>SHAP Explanation Error</h3><p>{e}</p>")
        explanation = {"error": str(e)}

    return str(waterfall_path), str(force_html_path), explanation
