---
title: "MUSE: Multi-block Utility for Safe & Explainable Learning"
date: 25 November 2025
keywords: [machine learning, explainable AI, multi-omics, radiomics, biomedical informatics]
bibliography: paper.bib
---

# Summary

MUSE (Multi-block Utility for Safe & Explainable Learning) is an open-source Python toolkit that simplifies building, explaining, and documenting supervised models on block-structured tabular biomedical data. In many biomedical studies different sources of information (demographics, clinical labs, multi-omics assays, imaging/radiomics) are naturally grouped into modality blocks. Standard ML workflows flatten these blocks into a single table, making it hard to track which modality drives model decisions — a crucial shortcoming for clinical interpretation, validation and regulatory review.

MUSE accepts a dictionary of pandas DataFrames (one DataFrame per named block), aligns rows by sample identifier, and produces a reproducible modeling pipeline that (i) constructs a deterministic design matrix, (ii) trains any scikit-learn compatible estimator (default: balanced Random Forests [@breiman2001random]), (iii) computes SHAP-based global and local explanations [@Lundberg2017], and (iv) auto-generates a regulator-aligned model card summarizing intended use, data provenance, metrics, and explanation summaries [@Mitchell2019]. The package is intentionally lightweight and follows scikit-learn conventions (fit, predict, predict_proba) so researchers can adopt it with minimal friction. MUSE helps teams produce auditable, interpretable models that are easier to validate, communicate, and (where appropriate) advance toward clinical deployment.

# Statement of need

Biomedical prediction problems increasingly combine heterogeneous modalities: a single prediction may draw on patient history, blood biomarkers, genomic assays, and image-derived features. Although tools exist for modeling tabular data (e.g., scikit-learn [@pedregosa2011scikit]) and for post-hoc explanations (e.g., LIME [@Ribeiro2016], SHAP [@Lundberg2017]), there is no widely adopted open toolkit that (1) retains modality structure through the modeling lifecycle, (2) computes aggregated block-level importances automatically, and (3) produces concise, reproducible documentation suited to regulatory review (e.g., model cards). In practice, researchers custom-code merges, explanation aggregation, and reports: an error-prone process that hinders reproducibility and transparent communication.

To our knowledge, no existing open-source package provides MUSE’s combination of capabilities. While many libraries can train models or compute feature-level explanations, MUSE is the only tool that offers: (1) native, first-class support for block-structured tabular data; (2) automatic SHAP aggregation at the block level to quantify modality contributions; (3) integrated, regulator-aligned model cards generated out-of-the-box; and (4) a simple scikit-learn–style interface requiring no deep ML expertise. This unified workflow enables biomedical researchers to build transparent, reproducible multi-modal models without writing custom data-merging scripts, explanation pipelines, or documentation templates: addressing a critical gap in current machine-learning tooling.

# Implementation and architecture

MUSE is implemented in Python and built on standard scientific-Python libraries (pandas, NumPy, scikit-learn, SHAP). The user API is intentionally concise:
1. MUSE(estimator=..., random_state=..., max_background=...) — construct the wrapper.
2. fit(blocks: Dict[str, DataFrame], y: Series) — align blocks, build design matrix, train estimator.
3. evaluate(blocks, y_test) — compute accuracy, ROC AUC, F1, average precision and store results.
4. explain_global(blocks_background) — compute TreeSHAP on a background sample, return feature-level mean |SHAP| and aggregated block importances.
5. explain_local(blocks_sample, sample_id) — compute per-sample SHAP contributions.
6. generate_model_card(dataset_name, dataset_reference, task_description) — return a serializable model card (dict) containing metadata, environment versions, reproducibility notes, performance, and explainability summaries.

Internally, MUSE keeps a feature_to_block mapping so all outputs reference original block provenance. The default estimator is a Random Forest with class_weight="balanced" and multi-core training — a conservative choice that handles heterogeneous features and missingness robustly. Users can supply any estimator implementing the scikit-learn interface (e.g., logistic regression, gradient boosting, neural networks wrapped as sklearn estimators).

Figure 1 (file: model_architecture.png) illustrates the pipeline: block ingestion → deterministic concatenation → estimator training → SHAP explainability → block aggregation → model card generation.

# Example usage

A compact example (also included in the repository notebooks):

```python
from muse_xai import MUSE
from sklearn.ensemble import RandomForestClassifier

model = MUSE(
    estimator=RandomForestClassifier(n_estimators=100, class_weight="balanced"),
    random_state=42
)

model.fit({"clinical": X_clin, "omics": X_omics}, y_train)
metrics = model.evaluate({"clinical": Xc_test, "omics": Xo_test}, y_test)
feat_imp, block_imp = model.explain_global({"clinical": X_cl, "omics": X_om})
card = model.generate_model_card("Example dataset", "internal", "Binary classification")
```
The repository contains three detailed example notebooks: WDBC (UCI) [@wolberg1993wdbc], CBIS-DDSM radiomics [@Lee2017], and a WAW-TACE radiomics + clinical use case. These illustrate reproducible workflows and the model card outputs; the notebooks are structured so reviewers can re-run the examples on sampled data.

# Performance / evaluation

We validated MUSE across the included examples. On WDBC (569 samples, three logical blocks of 10 features each), the default Random Forest achieves ~97.4% accuracy and ~0.99 AUC; SHAP aggregation highlights the standard-error block as most influential. On the CBIS-DDSM case-descriptions benchmark, a descriptor-heavy model achieved ~79% accuracy (AUC ≈ 0.86) and the radiomic descriptor block dominated importance. On the WAW-TACE dataset (230 patients, ~3,300 radiomic features + 41 clinical features), MUSE handles high dimensionality and shows radiomics contributing ≈70% of SHAP importance while also highlighting limitations in AUC — information the auto model card records for transparency. Full experimental details, random seeds (default random_state=42 used in notebooks), and environment versions are recorded in the repository examples and in generated model cards to enable exact reproduction.

# Quality control

MUSE includes unit tests that verify block alignment, feature→block mapping, SHAP aggregation correctness on synthetic ground-truth models, and model card serialization. Continuous integration runs the test suite on supported Python versions (3.9–3.12). Example notebooks serve as end-to-end regression checks. The project follows best practices for scientific software packaging and documentation [@Smith2018].

# Software availability

- Repository: https://github.com/vpthehuman/MUSE-multi-block-xai (archived release)
- Release (Zenodo DOI): 10.5281/zenodo.17793183
- CITATION file: `CITATION.cff` at repository root.
- License: MIT (see `LICENSE`)
- Installation: `pip install git+https://github.com/vpthehuman/MUSE-multi-block-xai.git` or install the archived release via DOI.
- Supported Python versions: 3.9 — 3.12
- Dependencies: See `requirements/requirements.txt` for core runtime dependencies and `requirements/requirements-dev.txt` for development and testing.
- Reproducibility: Example notebooks and the `run_quick_demo.sh` script include `random_state=42` seeds. Each generated model card includes exact environment versions (python, numpy, pandas, scikit-learn, shap, matplotlib) and the package version for reproducibility.

# Acknowledgements
We thank the maintainers of scikit-learn, SHAP, pandas, and NumPy for foundational tools. We acknowledge the UCI Machine Learning Repository and The Cancer Imaging Archive for WDBC and CBIS-DDSM datasets [@wolberg1993wdbc; @Lee2017]. We gratefully acknowledge colleagues and lab members for feedback on design and examples.

# References
<!-- End of content; references will be included in paper.bib -->


