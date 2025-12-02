from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import textwrap
import platform
import sys
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    average_precision_score,
    classification_report,
)


Blocks = Mapping[str, pd.DataFrame]


@dataclass
class MUSE:
    """
    MUSE – Multi-block Utility for Safe & Explainable learning.

    A light-weight wrapper around any scikit-learn style classifier that:

    - Accepts multiple named feature blocks (e.g. 'clinical_block', 'omics_block').
    - Trains a classifier.
    - Provides global & local SHAP explanations aggregated at feature
      and block level.
    - Emits a regulator-style model card as a Python dict.

    Parameters
    ----------
    model :
        Any estimator implementing scikit-learn's fit / predict_proba API.
        Defaults to RandomForestClassifier with sensible parameters.
    random_state :
        Random seed for internal operations (model and sampling).
    max_background :
        Maximum number of background samples used for SHAP calculations.
        Larger values improve stability at the cost of runtime.
    """

    model: Optional[object] = None
    random_state: int = 42
    max_background: int = 150

    # internal attributes initialised after fitting
    fitted_: bool = field(init=False, default=False)
    feature_to_block_: Dict[str, str] = field(init=False, default_factory=dict)
    metrics_: Dict[str, float] = field(init=False, default_factory=dict)
    block_importance_: Optional[pd.Series] = field(init=False, default=None)
    feature_importance_: Optional[pd.DataFrame] = field(init=False, default=None)
    shap_background_: Optional[pd.DataFrame] = field(init=False, default=None)
    classification_report_: Optional[str] = field(init=False, default=None)
    model_card_: Optional[dict] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = RandomForestClassifier(
                n_estimators=500,
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=-1,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _concat_blocks(self, blocks: Blocks) -> pd.DataFrame:
        """
        Concatenate block DataFrames in deterministic order, recording which
        block each feature came from.
        """
        frames = []
        feature_to_block: Dict[str, str] = {}

        for block_name in sorted(blocks.keys()):
            df = blocks[block_name]
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Block '{block_name}' must be a pandas DataFrame.")
            frames.append(df)
            for col in df.columns:
                feature_to_block[col] = block_name

        X_concat = pd.concat(frames, axis=1)
        # update mapping
        self.feature_to_block_ = feature_to_block
        return X_concat

    @staticmethod
    def _to_shap_matrix(shap_values) -> np.ndarray:
        """
        Convert different SHAP outputs (list, Explanation, array)
        into a 2D numpy array for the positive class.

        Returns
        -------
        arr : ndarray of shape (n_samples, n_features)
        """
        sv = shap_values

        # Binary classifiers often return [shap_vals_class0, shap_vals_class1]
        if isinstance(sv, list):
            sv = sv[1]

        # Newer SHAP: Explanation object with .values
        if hasattr(sv, "values"):
            sv = sv.values

        arr = np.asarray(sv)

        # Ensure 2D
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        return arr

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------
    def fit(self, blocks_train: Blocks, y_train: pd.Series) -> "MUSE":
        """
        Fit the underlying classifier on concatenated multi-block features.
        """
        X_train = self._concat_blocks(blocks_train)
        self.model.fit(X_train, y_train)
        self.fitted_ = True
        return self

    def evaluate(
        self,
        blocks_test: Blocks,
        y_test: pd.Series,
        target_names: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the fitted model on a held-out test set.

        Returns
        -------
        metrics : dict
            accuracy, roc_auc, f1, average_precision
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before evaluate().")

        X_test = self._concat_blocks(blocks_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "f1": float(f1_score(y_test, y_pred)),
            "average_precision": float(average_precision_score(y_test, y_prob)),
        }
        self.metrics_ = metrics

        if target_names is None:
            target_names = ("class 0", "class 1")

        self.classification_report_ = classification_report(
            y_test, y_pred, target_names=list(target_names), digits=3
        )
        return metrics

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    def explain_global(
        self,
        blocks_background: Blocks,
        max_features_plot: int = 20,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Compute global SHAP importance.

        Parameters
        ----------
        blocks_background :
            Blocks to use as SHAP background (typically training set).
        max_features_plot :
            Number of top features to display in the bar plot.

        Returns
        -------
        feature_importance : DataFrame
            Columns: ['feature', 'mean_abs_shap', 'block'] sorted descending.
        block_importance : Series
            Sum of mean_abs_shap per block.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before explain_global().")

        X_bg = self._concat_blocks(blocks_background)

        # subsample for speed
        if len(X_bg) > self.max_background:
            X_bg = X_bg.sample(self.max_background, random_state=self.random_state)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_bg)
        shap_matrix = self._to_shap_matrix(shap_values)

        abs_shap = np.abs(shap_matrix)
        mean_abs = np.asarray(abs_shap.mean(axis=0)).ravel()

        rows = []
        for feat_name, m in zip(X_bg.columns, mean_abs):
            rows.append(
                {
                    "feature": feat_name,
                    "mean_abs_shap": float(m),
                    "block": self.feature_to_block_.get(feat_name, "unknown"),
                }
            )

        feat_importance = pd.DataFrame(rows).sort_values(
            "mean_abs_shap", ascending=False
        )

        block_importance = (
            feat_importance.groupby("block")["mean_abs_shap"]
            .sum()
            .sort_values(ascending=False)
        )

        self.feature_importance_ = feat_importance
        self.block_importance_ = block_importance
        self.shap_background_ = X_bg

        # simple global bar plot
        top = feat_importance.head(max_features_plot)
        plt.figure(figsize=(7, 6))
        plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
        plt.xlabel("mean |SHAP value| (impact on positive class)")
        plt.title("Global feature importance (MUSE)")
        plt.tight_layout()
        plt.show()

        return feat_importance, block_importance

    def explain_local(
        self,
        blocks_sample: Blocks,
        sample_index,
        max_features_plot: int = 15,
    ) -> pd.DataFrame:
        """
        Local explanation for a single sample.

        Parameters
        ----------
        blocks_sample :
            Blocks containing the sample (e.g. test blocks).
        sample_index :
            Index of the row to explain (must exist in the blocks' indices).
        max_features_plot :
            Number of features to show in the local bar plot.

        Returns
        -------
        df_local : DataFrame
            Columns: ['feature', 'value', 'shap_value', 'block', 'abs_shap']
            sorted by |SHAP|.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before explain_local().")

        X_all = self._concat_blocks(blocks_sample)
        if sample_index not in X_all.index:
            raise ValueError(f"sample_index {sample_index!r} not found in blocks.")

        row = X_all.loc[[sample_index]]
        features = list(row.columns)
        values = row.values.flatten().tolist()
        n_features = len(features)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(row)
        shap_matrix = self._to_shap_matrix(shap_values)
        shap_vec = np.asarray(shap_matrix).reshape(-1)

        if shap_vec.shape[0] != n_features:
            shap_vec = shap_vec[:n_features]
        shap_list = shap_vec.tolist()

        blocks = [self.feature_to_block_.get(f, "unknown") for f in features]

        df_local = pd.DataFrame(
            {
                "feature": features,
                "value": values,
                "shap_value": shap_list,
                "block": blocks,
            }
        )
        df_local["abs_shap"] = df_local["shap_value"].abs()
        df_local = df_local.sort_values("abs_shap", ascending=False)

        # local bar plot
        top = df_local.head(max_features_plot)
        plt.figure(figsize=(6, 8))
        plt.barh(top["feature"][::-1], top["shap_value"][::-1])
        plt.xlabel("SHAP value (impact on positive class)")
        plt.title(f"Local explanation for sample {sample_index}")
        plt.tight_layout()
        plt.show()

        return df_local

    # ------------------------------------------------------------------
    # Model card
    # ------------------------------------------------------------------
    def generate_model_card(
        self,
        dataset_name: str,
        dataset_reference: str,
        task_description: str,
        pipeline_description: Optional[str] = None,
        known_issues: Optional[list] = None,
        mitigations: Optional[list] = None,
    ) -> dict:
        """
        Generate a regulator-style model card for the fitted model.

        Parameters
        ----------
        dataset_name :
            Human-readable dataset name.
        dataset_reference :
            How to obtain / cite the dataset.
        task_description :
            One-line plain language description of the prediction task.
        pipeline_description :
            Optional narrative of your data & modelling pipeline.
            If None, a generic description is used.
        known_issues :
            Optional list of known model limitations.
        mitigations :
            Optional list of mitigation strategies.

        Returns
        -------
        card : dict
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before generate_model_card().")

        now_iso = datetime.utcnow().isoformat() + "Z"

        # Collect accurate version info for main dependencies (use importlib.metadata)
        try:
            from importlib import metadata as importlib_metadata  # py3.8+
        except Exception:
            import importlib_metadata  # type: ignore

        def _pkg_version(pkg_name: str) -> str:
            try:
                return importlib_metadata.version(pkg_name)
            except Exception:
                return "unknown"

        # package name for this project on PyPI is 'muse-xai' (adjust if different)
        env_info = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "muse_version": getattr(__import__("muse_xai"), "__version__", "0.0.0"),
            "numpy_version": _pkg_version("numpy"),
            "pandas_version": _pkg_version("pandas"),
            "scikit_learn_version": _pkg_version("scikit-learn"),
            "shap_version": _pkg_version("shap"),
            "matplotlib_version": _pkg_version("matplotlib"),
        }

        # generic fallbacks
        if pipeline_description is None:
            pipeline_description = textwrap.dedent(
                """
                1. Organise raw features into named blocks (e.g. clinical, omics, radiomics).
                2. Preprocess each block (imputation, encoding, scaling as needed).
                3. Concatenate blocks in a fixed order to form a design matrix.
                4. Train an sklearn-compatible classifier on the training data.
                5. Evaluate on held-out data with ROC AUC, F1 and average precision.
                6. Compute SHAP values on a background sample.
                7. Aggregate SHAP at feature and block level for interpretability.
                8. Generate this model card summarising performance and limitations.
                """
            ).strip()

        if known_issues is None:
            known_issues = [
                "No external validation cohort included.",
                "No explicit calibration or fairness analysis in this configuration.",
            ]
        if mitigations is None:
            mitigations = [
                "Add cross-validation and external validation datasets where possible.",
                "Assess calibration and subgroup performance before deployment.",
                "Re-train or update the model when data drift is detected.",
            ]

        blocks_summary = {
            block: {
                "n_features": int(
                    sum(1 for f, b in self.feature_to_block_.items() if b == block)
                )
            }
            for block in sorted(set(self.feature_to_block_.values()))
        }

        card = {
            "model_overview": {
                "framework_name": "MUSE – Multi-block Utility for Safe & Explainable learning",
                "created_at_utc": now_iso,
                "model_type": type(self.model).__name__,
                "intended_use": task_description,
                "not_intended_for": textwrap.dedent(
                    """
                    - Direct clinical decision-making without human oversight.
                    - Populations or settings substantially different from the training data.
                    """
                ).strip(),
            },
            "data": {
                "dataset_name": dataset_name,
                "dataset_reference": dataset_reference,
                "n_features": int(len(self.feature_to_block_)),
                "blocks": blocks_summary,
            },
            "performance": {
                "primary_metrics": self.metrics_,
                "decision_threshold": 0.5,
                "classification_report": self.classification_report_,
            },
            "explainability": {
                "method": "SHAP (TreeExplainer, global & local, positive class)",
                "top_blocks_by_importance": (
                    self.block_importance_.to_dict()
                    if self.block_importance_ is not None
                    else {}
                ),
                "notes": (
                    "Feature-level SHAP values are aggregated per block "
                    "to obtain high-level, modality-style importance."
                ),
            },
            "risk_and_limitations": {
                "known_issues": known_issues,
                "mitigations": mitigations,
            },
            "reproducibility": {
                "random_state": self.random_state,
                "environment": env_info,
                "training_pipeline": pipeline_description,
            },
        }

        self.model_card_ = card
        return card

