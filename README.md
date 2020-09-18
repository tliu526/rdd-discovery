# Interpretable and automated RDD discovery using recursive partitioning

Code repository for working paper, "Interpretable and automated RDD discovery using recursive partitioning."

## File Descriptions

- `llr.py`: Utilities for computing log-likelihood ratios for discontinuity search. Implements equations 9-10 in Herlands et al.
- `sim.py`: Utilities for simulated data. Currently supports fuzzy RDD data generation in two dimensions.
- `tree.py`: Utilities for defining the tree-based discontinuity search mechanism. A simple implementation of a decision tree as a proof of concept. Final implementation likely should leverage existing tree implementations, such as GRF or scikit's DecisionTreeClassifier.
- `test_tree.py`: Unit tests for `tree.py`.
