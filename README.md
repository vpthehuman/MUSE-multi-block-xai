# MUSE – Multi-block Utility for Safe & Explainable learning

MUSE is a small Python library for **block-structured tabular ML** with
**SHAP-based explanations** and **regulator-style model cards**.

It is designed for applications like multi-omics, radiomics + clinical data,
and other multi-view biomedical datasets.

## Features

- Treats each data modality as a named **block** (e.g. `clinical_block`,
  `omics_block`, `radiomics_block`).
- Works with any scikit-learn style classifier (RandomForest by default).
- Global and local **SHAP** explanations with aggregation at block level.
- Simple API that plays nicely with pandas and Jupyter.

## Installation

```bash
pip install git+https://github.com/vpthehuman/MUSE-multi-block-xai.git 
