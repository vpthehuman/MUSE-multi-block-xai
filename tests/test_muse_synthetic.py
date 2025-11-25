import numpy as np
import pandas as pd

from muse_xai import MUSE


def test_muse_end_to_end():
    rng = np.random.RandomState(0)

    n = 120
    # Block 1: clinical-like
    X1 = pd.DataFrame(
        rng.normal(size=(n, 3)),
        columns=["age", "bmi", "lab_a"],
    )
    # Block 2: omics-like
    X2 = pd.DataFrame(
        rng.normal(size=(n, 5)),
        columns=[f"gene_{i}" for i in range(5)],
    )

    # Simple nonlinear label
    y = (X1["age"] + X2["gene_0"] * 0.8 + rng.normal(scale=0.5, size=n) > 0).astype(int)

    # Train/test split
    train_idx = np.arange(0, 80)
    test_idx = np.arange(80, n)

    blocks_train = {
        "clinical_block": X1.iloc[train_idx],
        "omics_block": X2.iloc[train_idx],
    }
    blocks_test = {
        "clinical_block": X1.iloc[test_idx],
        "omics_block": X2.iloc[test_idx],
    }

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    muse = MUSE(random_state=42)
    muse.fit(blocks_train, y_train)
    metrics = muse.evaluate(blocks_test, y_test)

    # sanity check: better than random
    assert metrics["accuracy"] > 0.6

    feat_imp, block_imp = muse.explain_global(blocks_train)
    assert not feat_imp.empty
    assert set(block_imp.index) == {"clinical_block", "omics_block"}

    # local explanation for one test sample
    idx = blocks_test["clinical_block"].index[0]
    local = muse.explain_local(blocks_test, idx)
    assert "feature" in local.columns
    assert "shap_value" in local.columns

    card = muse.generate_model_card(
        dataset_name="Synthetic demo",
        dataset_reference="Generated in tests.test_muse_synthetic",
        task_description="Toy binary classification with two feature blocks.",
    )
    assert "model_overview" in card
    assert "data" in card

