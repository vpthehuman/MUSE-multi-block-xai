---
title: "MUSE: Multi-block Utility for Safe & Explainable Learning"
tags:
  - machine learning
  - explainable AI
  - biomedical informatics
  - multi-omics
  - radiomics
authors:
  - name: Matthew Chua
    affiliation: 1
  - name: Vishnupriya Kannan
    orcid: 0009-0007-1326-4549
    affiliation: 1
  - name: Khanh Lee
    affiliation: 1
affiliations:
  - name: National University of Singapore, Singapore
    index: 1
date: 2025-11-25
bibliography: paper.bib
---

# Summary

MUSE (Multi-block Utility for Safe & Explainable Learning) is an open-source Python package for machine learning on block-structured tabular biomedical data. It enables data scientists to train classification models across multiple feature blocks (e.g. clinical measurements, imaging-derived features, multi-omics data) in a unified framework. Emphasizing safety and explainability, MUSE integrates robust learning algorithms with post-hoc explanations and model documentation. By default, it uses ensemble tree models (e.g. Random Forests [@breiman2001random]) with balanced learning to handle class imbalances common in clinical datasets. For interpretability, MUSE leverages SHAP (SHapley Additive exPlanation) values [@Lundberg2017] to provide both global and local explanations of model predictions. Additionally, it automatically generates regulator-aligned model cards [@Mitchell2019] summarizing the model’s intended use, data, performance, and key insights, facilitating transparent reporting. In summary, MUSE helps researchers build transparent, documented machine learning models for biomedical applications in a way that aligns with emerging requirements for safe and explainable AI.

# Statement of need

Machine learning applications in biomedicine often involve heterogeneous tabular data divided into logical blocks (or modalities). For example, a cancer diagnosis task may combine patient demographics, blood biomarkers, and imaging-derived features as separate feature blocks. However, most popular ML libraries (e.g. scikit-learn [@pedregosa2011scikit]) and AutoML tools treat such data as a single flat table, lacking native support for multi-block structures or per-block interpretation. This makes it difficult to assess which data sources drive a model’s predictions – a critical concern in biomedical research. While the field of multi-omics data analysis has grown, there is a lack of general-purpose, open tools to seamlessly train models on multiple feature sets and attribute importance at the block level. Researchers often resort to custom scripts to merge data sources and manually compute importances, which is error-prone and not standardized.

Explainability is another pressing need in clinical ML. Domain experts and regulators require interpretable models due to high stakes and a demand for trust in AI predictions. Existing model-agnostic explanation techniques like LIME [@Ribeiro2016] or SHAP [@Lundberg2017] can explain individual predictions, but integrating these tools into a pipeline and aggregating insights (for example, understanding which block of features is most influential overall) is non-trivial. Moreover, emerging regulatory guidelines for AI in healthcare emphasize documentation of model behavior, intended use, and performance (often in formats similar to model cards [@Mitchell2019]). Yet, few software packages automatically produce such documentation. This gap between what is needed (block-aware modeling, built-in explainability, and thorough documentation) and what existing tools offer motivates the development of MUSE. To address these needs, we designed MUSE as a unified framework that simplifies multi-block model training and provides transparent explanations and reports “out-of-the-box”. By providing these capabilities in one toolkit, MUSE reduces the effort for researchers to create safe (well-validated, documented) and explainable predictive models on complex biomedical data.

# Implementation and architecture

MUSE is implemented in Python and built on the familiar scikit-learn API, making it easy to integrate into existing workflows. Users organize their input as a dictionary of pandas DataFrames, where each entry corresponds to a named feature block (for example, `{"clinical": df_clinical, "imaging": df_imaging}`). Under the hood, MUSE concatenates these blocks (ensuring proper alignment by sample ID) to form the full feature matrix for model training. By default, the package uses a `RandomForestClassifier` [@breiman2001random] (from scikit-learn) with 500 trees, `class_weight="balanced"`, and parallel inference, as a robust baseline classifier. This choice reflects an emphasis on safe learning – ensemble trees tend to handle feature heterogeneity and missing values gracefully, and balanced class weights mitigate skewed class distributions common in medical datasets. The framework is extensible: users may optionally supply a different model (any scikit-learn compatible estimator) at initialization, and hyperparameters can be adjusted as needed. After training (`fit`), MUSE provides a built-in `evaluate` method that computes common performance metrics (accuracy, AUROC, F1, etc.) on a test set and stores them internally for reporting.

**Explainability through SHAP.** MUSE tightly integrates SHAP-based explanations to help users and stakeholders understand model behavior. Globally, users can call `explain_global` with a background dataset (typically the training set or a sample thereof) to compute feature importances. This uses TreeSHAP [@Lundberg2017] (for tree-based models) to calculate Shapley values for each feature across the background samples. The result is presented in two levels: (1) a feature-level importance table, where each feature’s mean |SHAP value| indicates its overall contribution to the prediction outcome; and (2) a block-level importance summary, which aggregates the feature importances by their block, quantifying the influence of each data block in the model’s decisions. For instance, in a breast cancer prediction model combining clinical and imaging features, MUSE can reveal if imaging-derived features contribute, say, 80% of the total importance versus 20% from clinical data. Such block-level explanation is invaluable for researchers to identify dominant modalities and for clinicians to interpret which source of information the model relies on most. In addition to global explanations, MUSE offers `explain_local` for case-by-case interpretability: given a specific sample, it returns the SHAP value of each feature for that sample’s prediction, highlighting which patient-specific factors pushed the model towards a certain diagnosis. These local explanations can be used to generate patient-level reason codes (e.g. “high tumor size and radiomics texture X increased malignant risk”) which improve trust and transparency in a clinical setting. All explanation outputs are provided as pandas DataFrames for easy downstream analysis or visualization.

**Regulator-aligned model cards.** Beyond model training and interpretation, MUSE generates a structured model card for each trained model via its `generate_model_card` utility. The model card is a dictionary (easily exportable to JSON or YAML) that documents key information about the model in alignment with proposals by Mitchell et al. [@Mitchell2019] and the needs of regulatory review. It includes sections detailing: the context and intended use of the model (user-provided description of the task and scope, e.g. “Binary classifier to detect malignant breast tumor cases from combined clinical and imaging data”); dataset details (name and reference of the dataset used, e.g. public repositories like the UCI Wisconsin Breast Cancer dataset [@wolberg1993wdbc] or CBIS-DDSM [@Lee2017] for mammography, along with train/test split information); model specifications (algorithm type, version of MUSE and dependencies, training date, etc.); performance metrics (the evaluation results on test data, including overall accuracy, AUC, precision, recall, etc., as well as class-specific performance if applicable); and explainability analysis (for example, a summary of the top features or blocks by importance, or statements about which inputs most strongly drive decisions). By compiling this information, the model card serves as a concise report that can be reviewed by regulatory bodies or project stakeholders to assess the model’s readiness and limitations. Importantly, it encourages users to document any ethical or safety considerations manually within provided fields (e.g. potential biases, appropriate use cases, and limitations of the model), thus aligning the development process with safe AI practices and transparency. The model card format is inspired by guidelines from the healthcare AI community and scientific software best-practice papers such as JOSS’s own description [@Smith2018], ensuring that models built with MUSE are accompanied by clear and reproducible descriptions.

Internally, the architecture of MUSE is kept lightweight. The core is a single class that encapsulates the model and all associated information (feature group mappings, metrics, explanations, etc.). This design makes it straightforward to save and reuse a trained MUSE model, as all necessary components are properties of the class instance. MUSE relies on well-tested libraries: data handling with pandas, modeling with scikit-learn [@pedregosa2011scikit], and shap for explainability. By building on these standard foundations, the package ensures compatibility and stability, while MUSE’s own contributions lie in gluing them together for the multi-block use-case and adding domain-specific guardrails (like class balancing, integrated evaluation, and standardized reporting). The API follows scikit-learn conventions (`fit`, `predict`, `predict_proba`, etc.), lowering the learning curve for new users. Overall, the implementation emphasizes clarity and safety – each step from data input to model output is transparent and can be audited, which is crucial in biomedical machine learning pipelines.

# Example usage

MUSE is designed to be straightforward to use for data scientists familiar with pandas and scikit-learn. The following snippet illustrates a typical workflow using MUSE on a hypothetical multi-block dataset:

```python
from muse_xai import MUSE
from sklearn.ensemble import RandomForestClassifier

# Prepare data blocks (e.g., clinical and omics DataFrames) and labels
blocks_train = {"clinical_block": X_clinical_train, "omics_block": X_omics_train}
blocks_test  = {"clinical_block": X_clinical_test,  "omics_block": X_omics_test}

# Initialize MUSE with a Random Forest backend
model = MUSE(
    estimator=RandomForestClassifier(n_estimators=100, class_weight="balanced"),
    random_state=42,
)

# Train the model
model.fit(blocks_train, y_train)

# Evaluate on test data
metrics = model.evaluate(blocks_test, y_test)
print("Test metrics:", metrics)  # e.g., accuracy, ROC AUC, F1, etc.

# Generate global explanations (SHAP feature importances)
feature_importance, block_importance = model.explain_global(blocks_train)

# Produce a model card report
model_card_text = model.generate_model_card(
    dataset_name="Example multi-block dataset",
    dataset_reference="internal or public reference",
    task_description="Binary classification using clinical + omics blocks.",
)
print(model_card_text)

```

<!-- End of content; references will be included in paper.bib -->

