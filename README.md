# MUSE – Multi-block Utility for Safe & Explainable learning

MUSE is a small Python library for block-structured tabular ML with
SHAP-based explanations and regulator-style model cards.
It is designed for applications like multi-omics, radiomics + clinical data,
and other multi-view biomedical datasets.


# Features 
1. Treats each data modality as a named block
(e.g. clinical_block, omics_block, radiomics_block).
2. Works with any scikit-learn style classifier
(RandomForest by default, but you can plug in your own model).
3. Global and local SHAP explanations with aggregation at block level.
4. Automatic model card generation for internal review, audits or papers.
5. Simple API that plays nicely with pandas and Jupyter notebooks.

# Installation:
pip install git+https://github.com/vpthehuman/MUSE-multi-block-xai.git

# Quickstart
```python
import pandas as pd
from muse_xai import MUSE

# Example: two blocks of features
blocks_train = {
    "clinical_block": X_clin_train,   # pandas DataFrame (n_samples, p1)
    "omics_block": X_omics_train,     # pandas DataFrame (n_samples, p2)
}
blocks_test = {
    "clinical_block": X_clin_test,
    "omics_block": X_omics_test,
}

y_train = y_train_series   # 0/1 labels
y_test = y_test_series

# Initialise MUSE (RandomForest backend by default)
muse = MUSE(random_state=42)

# Fit and evaluate
muse.fit(blocks_train, y_train)
metrics = muse.evaluate(blocks_test, y_test, target_names=("benign", "malignant"))
print(metrics)

# Global explanations (returns feature + block importance and shows a plot)
feat_imp, block_imp = muse.explain_global(blocks_train)

# Local explanation for a single sample
sample_idx = blocks_test["clinical_block"].index[0]
local_exp = muse.explain_local(blocks_test, sample_idx)

# Model card as a Python dict
card = muse.generate_model_card(
    dataset_name="Wisconsin Diagnostic Breast Cancer (WDBC)",
    dataset_reference="UCI Machine Learning Repository",
    task_description="Binary classification of breast masses (benign vs malignant).",
)


```

# Example notebooks
The examples/ folder contains python templates that reproduce the main use-cases
described in the paper:
1. wdbc_demo.py: morphological breast cancer features split into mean / SE / worst blocks.
2. cbis_mass_demo.py: CBIS-DDSM MASS case descriptions with a clinical block and a descriptor block.
3. wawtace_demo.py: WAW-TACE hepatocellular carcinoma dataset with clinical and CT radiomics blocks.

Each notebook shows:
1. how to load the public dataset,
2. how to build the blocks dictionary,
3. how to run MUSE, visualise SHAP plots, and generate a model card.

# Project structure

```text
MUSE-multi-block-xai/
  src/muse_xai/
    __init__.py
    core.py
  tests/
    test_muse_synthetic.py
  examples/
    wdbc_demo.py
    cbis_mass_demo.py
    wawtace_demo.py
  README.md
  LICENSE
  pyproject.toml
  model_architecture.png
  paper.md
  paper.bib
```

# License
MUSE is released under the MIT License. See LICENSE for details.
