# Contributing to MUSE

Thanks for your interest in contributing to MUSE! We welcome bug reports, feature requests, code, tests, documentation improvements, and example notebooks.

## Quickstart

1. Fork the repository and create a feature branch:
```bash
git clone https://github.com/your-username/MUSE-multi-block-xai.git
git checkout -b feat/your-feature
```

2. Install the package (development mode recommended):
```python
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements/requirements-dev.txt
```

3. run tests
```python
pytest -q
```

4. Make changes, add tests for new behavior, and run the test suite again.
5. Push your branch and open a Pull Request against main. In your PR description:
- Explain what you changed and why.
- Tag related issues (if any).
- Note any API changes or docs that need updates.
