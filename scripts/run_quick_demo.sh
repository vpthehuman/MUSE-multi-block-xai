#!/usr/bin/env bash
set -euo pipefail

# create sample data
python scripts/make_sample_data.py

# run demos (these scripts expect files in working dir sample_data or same dir)
echo "Running WDBC demo..."
cp sample_data/wdbc.data wdbc.data
python examples/wdbc_demo.py

echo "Running CBIS MASS demo..."
cp sample_data/mass_case_description_train_set.csv mass_case_description_train_set.csv
cp sample_data/mass_case_description_test_set.csv mass_case_description_test_set.csv
python examples/cbis_mass_demo.py

echo "Running WAW-TACE demo..."
cp sample_data/clinical_data_wawtace_v2_15_07_2024.xlsx clinical_data_wawtace_v2_15_07_2024.xlsx
cp sample_data/radiomics_data_wawtace_09_05_2024.xlsx radiomics_data_wawtace_09_05_2024.xlsx
python examples/wawtace_demo.py

echo "Quick demos finished."
