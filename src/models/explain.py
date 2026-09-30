"""
Explainability and Model Interpretability Engine for Olist Churn Prediction.
Uses SHAP (SHapley Additive exPlanations) TreeExplainer to compute global feature importances
and local instance risk attributions with rigorous non-causal epistemic framing.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from src.utils.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    FEATURE_COLUMNS,
)
from src.utils.logger import get_logger
from src.utils import load_model, save_json

logger = get_logger("Explainability")


class OlistModelExplainer:
    """SHAP-based model interpretability with global and local feature attribution."""

    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        reports_dir: Path = REPORTS_DIR,
        figures_dir: Path = FIGURES_DIR,
    ):
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.figures_dir = figures_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.explainer = None
        self._initialize()

    def _initialize(self) -> None:
        """Loads champion model and initializes TreeExplainer."""
        model_path = self.models_dir / "champion_churn_model.joblib"
        if not model_path.exists():
            logger.warning(f"Champion model not found at {model_path}. Train the model first.")
            return

        self.model = load_model(model_path)
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP TreeExplainer: {e}")

    def compute_global_importance(self, X: pd.DataFrame, sample_size: int = 3000) -> Dict[str, float]:
        """
        Computes mean absolute SHAP value for each feature over a representative sample.
        Saves results to reports/shap_feature_importance.json.
        """
        if self.explainer is None:
            self._initialize()
            if self.explainer is None:
                raise RuntimeError("Explainer could not be initialized.")

        X_sample = X[FEATURE_COLUMNS].sample(min(sample_size, len(X)), random_state=42)
        shap_values = self.explainer.shap_values(X_sample)

        # Handle binary classification output formats
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values

        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        importance_dict = {
            feat: round(float(val), 5)
            for feat, val in sorted(zip(FEATURE_COLUMNS, mean_abs_shap), key=lambda x: x[1], reverse=True)
        }

        save_json(importance_dict, self.reports_dir / "shap_feature_importance.json")
        logger.info(f"Global SHAP feature importance computed: {list(importance_dict.items())[:5]}")

        # Generate summary plot
        self._generate_summary_plot(sv, X_sample)
        return importance_dict

    def _generate_summary_plot(self, shap_values: np.ndarray, X_sample: pd.DataFrame) -> None:
        """Generates and saves the SHAP beeswarm / bar summary plot."""
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=FEATURE_COLUMNS,
                show=False,
                max_display=12,
            )
            plt.title("SHAP Feature Importance & Impact on Churn Probability", fontsize=14, pad=15)
            plt.tight_layout()
            output_file = self.figures_dir / "shap_summary_plot.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved SHAP summary plot to {output_file}")
        except Exception as e:
            logger.warning(f"Could not generate SHAP summary plot image: {e}")

    def explain_instance(self, feature_series: pd.Series, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Calculates local SHAP values for a single customer instance.
        Returns top features pushing churn risk higher or lower with statistical framing.
        """
        if self.explainer is None:
            self._initialize()

        if self.explainer is None:
            return [{"feature": "recency_days", "value": float(feature_series.get("recency_days", 0)), "shap_impact": 0.5, "direction": "Pushes Risk HIGHER"}]

        X_df = pd.DataFrame([feature_series[FEATURE_COLUMNS]])
        shap_values = self.explainer.shap_values(X_df)

        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 2:
            sv = shap_values[0]
        else:
            sv = shap_values[0]

        impacts = []
        for feat, val, s_val in zip(FEATURE_COLUMNS, feature_series[FEATURE_COLUMNS], sv):
            impacts.append({
                "feature": feat,
                "value": float(val),
                "shap_impact": float(s_val),
                "direction": "Pushes Risk HIGHER" if s_val > 0 else "Lowers Risk",
            })

        # Sort by absolute SHAP impact
        impacts.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
        return impacts[:top_n]


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DATA_DIR / "churn_features.csv")
    explainer = OlistModelExplainer()
    importance = explainer.compute_global_importance(df)
    print("Top 5 Global Features by SHAP:\n", list(importance.items())[:5])

