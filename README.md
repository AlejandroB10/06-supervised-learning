# Lab 6 — Supervised Learning (graded)

This repository includes only the materials for handout 6 (graded assignment). The goal is to build a multiclass classifier (A–E) on IMU sensor data (`ds_01.csv`), achieve macro F1 > 90%, and document the pipeline and results.

## Contents
- `06-handout/ml_lab6_es.ipynb`: Spanish notebook with data prep, PCA (99% variance), single models from previous labs, and ensemble methods; evaluation via stratified CV and confusion matrices.
- `06-handout/ml_lab6.ipynb`: English version of the same notebook.
- `06-handout/README_lab6.md`: assignment requirements and dataset notes.
- `06-handout/handout_6_supervised learning (graded).pdf`: official assignment statement.
- Data: `06-handout/ds_01.csv` (training) and `06-handout/common.csv` (common test set; `class` appears to be an ID 1..20).

## Environment
Use the `ml-env` conda environment (Python 3.13). Required packages: `pandas`, `scikit-learn`, `matplotlib`, `numpy`.

Example setup:
```bash
conda activate ml-env
pip install pandas scikit-learn matplotlib
```

## How to run
1. Open `06-handout/ml_lab6_es.ipynb` (or the English `ml_lab6.ipynb`).
2. Run the cleaning cells (drops ID/time cols, fixed 100% NA and zero-variance cols; high-NA threshold configurable) on copies `X_work`/`y_work`.
3. (Optional) Enable the outlier filter (IsolationForest) and note the retained ratio.
4. Evaluate single models (SVC RBF, Logistic Regression, Decision Tree, Perceptron) and ensembles (Voting, Bagging, RandomForest, ExtraTrees) with macro F1 in stratified CV; use the provided grids for tuning if needed.
5. For the best single and ensemble models, use `cross_val_predict` to obtain confusion matrices and accuracy.
6. Predict `common.csv` with the best ensemble. If true labels are available (currently `class` looks like IDs 1..20), compute confusion/accuracy; otherwise, report predictions only.

## Notes
- The notebooks do not modify the raw CSVs; all work is done on in-memory copies.
- At least one of the final models (single or ensemble) should reach macro F1 > 90% to satisfy the assignment.
