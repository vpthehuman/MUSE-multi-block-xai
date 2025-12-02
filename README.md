# MUSE – Multi-block Utility for Safe & Explainable learning

MUSE is a lightweight Python toolkit for building transparent, reproducible machine-learning models on block-structured tabular data—common in multi-omics, radiomics + clinical, and other multi-view biomedical datasets.
The library treats each data modality as a separate block, trains any scikit-learn-style estimator, produces global + local SHAP explanations, aggregates importances at the block level, and automatically generates regulator-aligned model cards.


# Features 
1. Native support for named blocks
(e.g., "clinical_block", "omics_block", "radiomics_block").
2. Works with any scikit-learn classifier
(defaults to balanced Random Forests, but users can plug in any estimator).
3. Explainability built-in
- TreeSHAP-based global & local explanations
- Automatic block-level aggregation for multi-modality interpretation
4. Automatic model cards: summaries for internal review, audits, reproducibility, and publications.
5. Simple, clean API: fully compatible with pandas, scikit-learn, and Jupyter workflows.

# Installation:
pip install git+https://github.com/vpthehuman/MUSE-multi-block-xai.git

# Quickstart
```python
import pandas as pd
from muse_xai import MUSE

# Example blocks
blocks_train = {
    "clinical_block": X_clin_train,
    "omics_block": X_omics_train,
}
blocks_test = {
    "clinical_block": X_clin_test,
    "omics_block": X_omics_test,
}

y_train = y_train_series
y_test = y_test_series

# Initialise MUSE (RandomForest backend by default)
muse = MUSE(random_state=42)

# Train and evaluate
muse.fit(blocks_train, y_train)
metrics = muse.evaluate(blocks_test, y_test, target_names=("benign", "malignant"))
print(metrics)

# SHAP-based global explanation
feat_imp, block_imp = muse.explain_global(blocks_train)

# Local SHAP explanation for a single sample
sample_idx = blocks_test["clinical_block"].index[0]
local_exp = muse.explain_local(blocks_test, sample_idx)

# Model card
card = muse.generate_model_card(
    dataset_name="Wisconsin Diagnostic Breast Cancer (WDBC)",
    dataset_reference="UCI Machine Learning Repository",
    task_description="Binary classification of breast masses (benign vs malignant).",
)

```
## Quick demo (fast, synthetic data)

MUSE includes a reproducible synthetic dataset so users and reviewers can run demonstrations without downloading large datasets.
```bash
# Install dependencies
pip install -e .
pip install -r requirements/requirements-dev.txt   # optional: tests, notebooks

# Generate synthetic demo data
python scripts/make_sample_data.py

# Run three quick example demos
./run_quick_demo.sh
```

# Example notebooks
Example scripts (Python templates) are available in examples/:
1. wdbc_demo.py: WDBC breast cancer dataset split into mean, se, and worst blocks.
2. cbis_mass_demo.py: CBIS-DDSM mass case-descriptions with clinical + descriptor blocks.
3. wawtace_demo.py: WAW-TACE liver cancer dataset integrating clinical + CT radiomics.

Each notebook shows:
✔ How to load data
✔ How to construct modality blocks
✔ How to run MUSE (fit → evaluate → explain → model card)
✔ How to visualise SHAP outputs

## Citation & DOI

If you use MUSE in your work, please cite:
Chua M., Vishnupriya K., Lee K. (2025).
MUSE: Multi-block Utility for Safe & Explainable Learning. Zenodo.
DOI: 10.5281/zenodo.17793183 


# Project structure

```text
MUSE-multi-block-xai/
  src/muse_xai/
    __init__.py
    core.py
  tests/
    test_muse_synthetic.py
  scripts/
    make_sample_data.py
    run_quick_demo.sh
  examples/
    wdbc_demo.py
    cbis_mass_demo.py
    wawtace_demo.py
  requirements/
    requirements.txt
    requirements-dev.txt
  workflows/
    tests.yml
  paper/
    paper.md
    paper.bib
  model_architecture.png
  README.md
  LICENSE
  pyproject.toml
```

# License
MUSE is released under the MIT License. See LICENSE for details.
