#!/usr/bin/env python3
"""
Generate tiny sample CSVs used by example demos so reviewers can run the
demos without downloading large public datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path

out = Path("sample_data")
out.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# 1) WDBC-like (use light sample - 30 features)
n = 50
stats = ["mean", "se", "worst"]
features = [
    "radius", "texture", "perimeter", "area", "smoothness", "compactness",
    "concavity", "concave_points", "symmetry", "fractal_dimension"
]
cols = ["ID", "diagnosis"] + [f"{s}_{f}" for s in stats for f in features]
data = []
for i in range(n):
    ID = 1000 + i
    diag = rng.choice(["B", "M"], p=[0.7, 0.3])
    row = [ID, diag] + list(rng.normal(loc=0.0, scale=1.0, size=len(cols)-2))
    data.append(row)
df_wdbc = pd.DataFrame(data, columns=cols)
df_wdbc.to_csv(out / "wdbc.data", index=False)

# 2) CBIS-MASS like (mass_case_description_train/test)
n2 = 40
margins = ["circumscribed", "ill-defined", "spiculated"]
ages = rng.integers(30, 80, size=n2)
paths = rng.choice(["MALIGNANT", "BENIGN"], size=n2, p=[0.3, 0.7])
marg = rng.choice(margins, size=n2)
df_mass = pd.DataFrame({
    "age": ages,
    "pathology": paths,
    "margin": marg
})
# split into train/test
train = df_mass.sample(frac=0.7, random_state=42)
test = df_mass.drop(train.index)
train.to_csv(out / "mass_case_description_train_set.csv", index=False)
test.to_csv(out / "mass_case_description_test_set.csv", index=False)

# 3) Tiny WAW-TACE like clinical + radiomics
n3 = 30
clinical_cols = ["age", "bmi", "stage", "early_response"]
clinical = pd.DataFrame({
    "age": rng.integers(40, 80, size=n3),
    "bmi": rng.normal(23, 3, size=n3),
    "stage": rng.choice([1, 2, 3], size=n3),
    "early_response": rng.choice(["NR", "R"], size=n3, p=[0.6, 0.4]),
}, index=[f"pt{i}" for i in range(n3)])
# small radiomics with 15 features
rad = pd.DataFrame(rng.normal(size=(n3, 15)), columns=[f"rad_{i}" for i in range(15)], index=clinical.index)

clinical.to_excel(out / "clinical_data_wawtace_v2_15_07_2024.xlsx")
rad.to_excel(out / "radiomics_data_wawtace_09_05_2024.xlsx")

print("Generated sample data in 'sample_data/'")

