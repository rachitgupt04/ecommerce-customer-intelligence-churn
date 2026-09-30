"""
Supervised Model Training Pipeline with Temporal Chronological Train/Validation/Test Splitting.
Compares Logistic Regression (Balanced & SMOTE), Random Forest, and XGBoost using Stratified 5-Fold CV
within historical training snapshots. Validates on intermediate snapshot and evaluates on the
untouched chronological holdout test snapshot.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.utils.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    CV_FOLDS,
    DEFAULT_CAMPAIGN_COST,
    DEFAULT_SAVED_MARGIN,
    DEFAULT_SUCCESS_RATE,
)
from src.utils.logger import get_logger
from src.utils import save_model, save_json
from src.models.evaluate import ModelEvaluator

logger = get_logger("ModelTraining")


class OlistChurnTrainer:
    """Temporal multi-snapshot training, CV comparison, tuning, and holdout evaluation."""

    def __init__(
        self,
        data_dir: Path = PROCESSED_DATA_DIR,
        models_dir: Path = MODELS_DIR,
        reports_dir: Path = REPORTS_DIR,
    ):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        """
        Loads the multi-snapshot dataset and splits strictly chronologically:
        - Train: Earlier snapshots (S1: 2017-09-01, S2: 2017-12-01)
        - Validation: Later snapshot (S3: 2018-03-01)
        - Test: Latest snapshot (S4: 2018-06-01)
        """
        dataset_path = self.data_dir / "churn_multi_snapshot.csv"
        if not dataset_path.exists():
            # Fallback to feature matrix if multi_snapshot not yet built
            dataset_path = self.data_dir / "churn_features.csv"
            if not dataset_path.exists():
                raise FileNotFoundError(f"Feature dataset not found at {dataset_path}. Run feature_engineering first.")

        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded multi-snapshot churn dataset: {df.shape[0]:,} rows, {df.shape[1]} columns.")

        if "split_role" not in df.columns:
            # If single cutoff file was loaded, assign chronological split based on snapshot_date
            df["split_role"] = "test"

        train_df = df[df["split_role"] == "train"].copy()
        val_df = df[df["split_role"] == "val"].copy()
        test_df = df[df["split_role"] == "test"].copy()

        # If train is empty (e.g. older single-snapshot file), split chronologically or stratified
        if len(train_df) == 0:
            logger.warning("split_role not populated in dataset. Creating 70/15/15 chronological/stratified partitions.")
            from sklearn.model_selection import train_test_split
            train_df, rem_df = train_test_split(df, test_size=0.30, random_state=RANDOM_STATE, stratify=df[TARGET_COLUMN])
            val_df, test_df = train_test_split(rem_df, test_size=0.50, random_state=RANDOM_STATE, stratify=rem_df[TARGET_COLUMN])

        X_train = train_df[FEATURE_COLUMNS].copy()
        y_train = train_df[TARGET_COLUMN].copy()
        X_val = val_df[FEATURE_COLUMNS].copy()
        y_val = val_df[TARGET_COLUMN].copy()
        X_test = test_df[FEATURE_COLUMNS].copy()
        y_test = test_df[TARGET_COLUMN].copy()

        logger.info(
            f"Chronological Splits -> Train: {len(X_train):,} obs ({y_train.mean()*100:.2f}% churn) | "
            f"Val: {len(X_val):,} obs ({y_val.mean()*100:.2f}% churn) | "
            f"Test: {len(X_test):,} obs ({y_test.mean()*100:.2f}% churn)"
        )

        # Persist test set with customer identifiers for dashboard evaluation
        test_df_export = X_test.copy()
        test_df_export[TARGET_COLUMN] = y_test
        test_df_export["customer_unique_id"] = test_df["customer_unique_id"]
        test_df_export["customer_id"] = test_df["customer_unique_id"]
        test_df_export.to_csv(self.data_dir / "holdout_test_set.csv", index=False)

        return X_train, X_val, X_test, y_train, y_val, y_test, df

    def run_model_comparison(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Compares baseline and tree ensembles using Stratified 5-Fold CV within training snapshots."""
        logger.info(f"Executing {CV_FOLDS}-Fold Stratified Cross-Validation strictly within training snapshots...")
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        num_neg = int((y_train == 0).sum())
        num_pos = int((y_train == 1).sum())
        scale_pos = float(num_neg / max(num_pos, 1))

        candidate_pipelines = {
            "Logistic Regression (Balanced)": ImbPipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
            ]),
            "Logistic Regression + SMOTE": ImbPipeline([
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]),
            "Random Forest (Balanced)": ImbPipeline([
                ("classifier", RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]),
            "XGBoost (Cost-Weighted)": ImbPipeline([
                ("classifier", XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    scale_pos_weight=scale_pos,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]),
        }

        scoring = {
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
            "f1": "f1",
            "recall": "recall",
            "precision": "precision",
        }

        comparison_results = {}
        for name, pipe in candidate_pipelines.items():
            logger.info(f"Running CV on {name}...")
            scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            comparison_results[name] = {
                "cv_roc_auc_mean": round(float(np.mean(scores["test_roc_auc"])), 4),
                "cv_roc_auc_std": round(float(np.std(scores["test_roc_auc"])), 4),
                "cv_pr_auc_mean": round(float(np.mean(scores["test_pr_auc"])), 4),
                "cv_pr_auc_std": round(float(np.std(scores["test_pr_auc"])), 4),
                "cv_f1_mean": round(float(np.mean(scores["test_f1"])), 4),
                "cv_recall_mean": round(float(np.mean(scores["test_recall"])), 4),
                "cv_precision_mean": round(float(np.mean(scores["test_precision"])), 4),
            }
            logger.info(
                f"{name} -> CV ROC-AUC: {comparison_results[name]['cv_roc_auc_mean']:.4f} "
                f"| PR-AUC: {comparison_results[name]['cv_pr_auc_mean']:.4f} "
                f"| F1: {comparison_results[name]['cv_f1_mean']:.4f}"
            )

        save_json(comparison_results, self.reports_dir / "model_cv_comparison.json")
        return comparison_results

    def tune_and_fit_champion(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
        """
        Tunes champion XGBoost hyperparameters on training folds, validates on chronological validation snapshot,
        and finally evaluates on the untouched chronological test set.
        """
        logger.info("Tuning champion XGBoost model via Stratified Grid Search...")
        num_neg = int((y_train == 0).sum())
        num_pos = int((y_train == 1).sum())
        scale_pos = float(num_neg / max(num_pos, 1))

        param_grid = {
            "max_depth": [3, 4],
            "learning_rate": [0.03, 0.05],
            "n_estimators": [100, 150],
            "subsample": [0.8, 1.0],
        }

        xgb = XGBClassifier(
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        logger.info(f"Champion Hyperparameters: {grid_search.best_params_} (Best 3-Fold CV ROC-AUC: {grid_search.best_score_:.4f})")

        # Evaluate on chronological Validation snapshot (S3)
        val_prob = best_model.predict_proba(X_val)[:, 1]
        val_pred = best_model.predict(X_val)
        val_metrics = ModelEvaluator.evaluate_predictions(y_val.values, val_pred, val_prob)
        logger.info(f"Validation Snapshot Performance: ROC-AUC={val_metrics['roc_auc']:.4f}, PR-AUC={val_metrics['pr_auc']:.4f}, F1={val_metrics['f1_score']:.4f}")

        # Retrain on combined Train + Validation (all historical snapshots up to 2018-05-31)
        X_train_full = pd.concat([X_train, X_val], ignore_index=True)
        y_train_full = pd.concat([y_train, y_val], ignore_index=True)
        scale_pos_full = float(int((y_train_full == 0).sum()) / max(int((y_train_full == 1).sum()), 1))
        best_model.set_params(scale_pos_weight=scale_pos_full)
        best_model.fit(X_train_full, y_train_full)

        # Evaluate on untouched chronological Holdout Test Snapshot (S4: 2018-06-01 to 2018-08-31)
        y_test_pred = best_model.predict(X_test)
        y_test_prob = best_model.predict_proba(X_test)[:, 1]

        holdout_metrics = ModelEvaluator.evaluate_predictions(y_test.values, y_test_pred, y_test_prob)
        logger.info(
            f"Holdout Test Performance (Snapshot 4): ROC-AUC={holdout_metrics['roc_auc']:.4f}, "
            f"PR-AUC={holdout_metrics['pr_auc']:.4f}, F1={holdout_metrics['f1_score']:.4f}, "
            f"Recall={holdout_metrics['recall']:.4f}, Precision={holdout_metrics['precision']:.4f}"
        )

        # Asymmetric business threshold optimization on holdout test set
        df_thresholds = ModelEvaluator.optimize_threshold(
            y_test.values,
            y_test_prob,
            cost_fp=DEFAULT_CAMPAIGN_COST,
            cost_fn=DEFAULT_SAVED_MARGIN,
            success_rate=DEFAULT_SUCCESS_RATE,
        )
        df_thresholds.to_csv(self.reports_dir / "threshold_optimization_results.csv", index=False)

        # Save artifacts
        save_model(best_model, self.models_dir / "champion_churn_model.joblib")
        save_json(
            {
                "model_name": "XGBoost Classifier (Multi-Snapshot Tuned)",
                "best_hyperparameters": grid_search.best_params_,
                "validation_snapshot_metrics": val_metrics,
                "holdout_test_metrics": holdout_metrics,
                "split_details": {
                    "train_observations": len(X_train),
                    "val_observations": len(X_val),
                    "test_observations": len(X_test),
                },
            },
            self.reports_dir / "champion_model_metrics.json",
        )

        return best_model, holdout_metrics, df_thresholds

    def execute_training_pipeline(self) -> Dict[str, Any]:
        """Executes full training pipeline end-to-end."""
        X_train, X_val, X_test, y_train, y_val, y_test, df_full = self.load_and_split_data()
        cv_comparison = self.run_model_comparison(X_train, y_train)
        champion_model, holdout_metrics, df_thresh = self.tune_and_fit_champion(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        return {
            "cv_comparison": cv_comparison,
            "holdout_metrics": holdout_metrics,
        }


if __name__ == "__main__":
    trainer = OlistChurnTrainer()
    results = trainer.execute_training_pipeline()
    print("Multi-Snapshot Training Complete. Metrics:\n", results)
