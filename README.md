# Automatic Incident & Log Agent

Course project for *Practical Machine Learning & AI Engineering for Software Engineers* (12-day outline, v2).
An agent that watches system logs, flags failures early, groups errors, and investigates via tools (MCP) to suggest causes and fixes.

**Dataset:** [Loghub HDFS_v1](https://github.com/logpai/loghub): 11M HDFS log lines, 575,061 blocks labelled normal/anomaly (2.93% anomalies, ~38 h of logs).

## Progress

| Day | Topic | Notebook | Status |
|---|---|---|---|
| 1 | ML mindset, statistics, vectorisation, gradient descent, logistic regression from scratch | [day01_foundations.ipynb](notebooks/day01_foundations.ipynb) | ✅ |
| 2 | EDA, pipeline, imbalance, time-based split, leakage proof | [day02_data_prep.ipynb](notebooks/day02_data_prep.ipynb) | ✅ (redone on project data) |
| 3 | LightGBM, SHAP, Optuna, MLflow | [day03_classic_models.ipynb](notebooks/day03_classic_models.ipynb) | ✅ |
| 4 | Evaluation, calibration, cost threshold, regression, ONNX + FastAPI + Docker | [day04_evaluate_and_ship.ipynb](notebooks/day04_evaluate_and_ship.ipynb) · [service.py](src/logagent/service.py) · [tests](tests/test_service.py) | ✅ Part 1 done |
| 5 | Neural networks + training pipelines (PyTorch) | | ⏳ next |

### Key results so far
- **Day 1:** NumPy `@` is ~110× faster than a Python loop for a 1M-element dot product. Gradient descent recovers w=2.97, b=2.02 (true 3, 2). From-scratch logistic regression matches scikit-learn's coefficients to within 1e-5 and scores PR-AUC 0.43 on validation (random ≈ 0.03).
- **Task:** early warning, i.e. predict a block failure from its **first 8 log events**. With full-lifetime counts, LightGBM hits PR-AUC 1.000, which is too easy and too late to be useful.
- **Split:** chronological 70/15/15. Shuffled CV reports PR-AUC 0.997 while time-aware CV reports 0.954, so shuffling gives an overly optimistic score.
- **Leakage proof:** a simulated post-incident "ticket" field raises a weak model from 0.15 to 0.79 PR-AUC. Computed as of prediction time, it adds nothing.
- **Early-warning PR-AUC (validation):** dummy 0.03 · logistic regression 0.40 · random forest 0.44 · tuned LightGBM **0.44** (test 0.84; the test period has a lower anomaly rate).
- Every run is tracked in MLflow; the best model is registered as `hdfs-anomaly-lgbm`, and same-seed retraining gives identical predictions.
- **Day 4 evaluation:** ROC-AUC 0.71 vs PR-AUC 0.40 on validation (ROC flatters rare events). Error slices show most missed failures start with the ordinary E9/E11 pattern and can't be told apart in 8 events.
- **Calibration:** the model is already close to calibrated (Brier 0.02049 raw → 0.02045 with sigmoid). Sigmoid is shipped; isotonic lowered PR-AUC and broke under ONNX float32.
- **Cost threshold (missed failure = 50, false page = 1):** validation picks 0.015, but on the test period it fires 43k false pages. Test has a lower anomaly rate and a different failure mix, so the threshold doesn't transfer. This is recorded as a known risk; the threshold was **not** re-tuned on test.
- **Regression:** lifetime targets are right-censored by the log window (mean latency 23k s in train → 950 s in test), so regression uses fully observed blocks only. Predicting `n_events`: LightGBM R² 0.27, linear 0.12, polynomial −7.7 (overfits).
- **Serving:** ONNX export matches LightGBM to within 2e-7, with 0 of 86,260 alerts different. ONNX Runtime is ~45× faster per single request (0.016 ms vs 0.76 ms). FastAPI `/predict` has input validation; 17 pytest tests pass, including 10 golden examples.

## Run it

```bash
uv sync
# download HDFS_v1.zip (186 MB) into data/raw/ and extract the preprocessed CSVs
curl -L -o data/raw/HDFS_v1.zip https://zenodo.org/api/records/8196385/files/HDFS_v1.zip/content
(cd data/raw && unzip -o HDFS_v1.zip 'preprocessed/*')
uv run python -m logagent.data          # builds data/processed/blocks.parquet (~5 min)
uv run jupyter lab notebooks/
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # experiment tracking UI

# serve the early-warning model
uv run pytest -q
uv run uvicorn logagent.service:app --port 8000
curl -s localhost:8000/predict -H 'content-type: application/json' \
     -d '{"events": ["E5","E22","E5","E5","E11","E9","E11","E9"], "elapsed_seconds": 1.0}'
docker build -t hdfs-early-warning . && docker run -p 8000:8000 hdfs-early-warning
```

## Layout
```
src/logagent/data.py     # raw HDFS archive -> per-block feature table (+ first-seen timestamps, early-warning features)
src/logagent/features.py # early-warning feature definition, shared by training and serving
src/logagent/splits.py   # shared feature lists + leakage-free time split
src/logagent/service.py  # FastAPI service: ONNX model + sigmoid calibration + cost-based threshold
models/                  # hdfs_lgbm.onnx + serving_config.json (exported by the Day 4 notebook)
tests/                   # API tests + golden examples
Dockerfile               # serving image (onnxruntime + FastAPI only)
notebooks/               # one notebook per course day, executed with outputs
```
