# Customer Churn Prediction System with CI/CD

End-to-end ML project: preprocessing -> feature engineering -> model training -> FastAPI service
-> Docker -> GitHub Actions CI/CD -> image published to GitHub Container Registry (GHCR).

## Project structure
```
app/                 FastAPI app (main.py, schemas.py)
src/                 config.py, preprocessing.py (cleaning + features), train.py
scripts/             generate_sample_data.py (practice data only)
tests/               pytest tests (preprocessing, training, API)
data/telco_churn.csv dataset (sample here -> REPLACE with the examiner's file)
models/              churn_model.joblib (best model) + metrics.json
model_config.json    model version + decision threshold
Dockerfile, requirements*.txt, setup.cfg
.github/workflows/ci-cd.yml
```

## Task mapping
| Task | Where |
|---|---|
| 1. Load + preprocess (missing/invalid values, encoding) | `src/preprocessing.py` (`clean_data`, imputers, `OneHotEncoder`) |
| 2. Feature engineering | `engineer_features`: `tenure_group`, `avg_monthly_spend`, `num_services`, `support_services`, `streaming_services`, `has_internet`, `is_auto_pay` |
| 3. >= 2 models + evaluation | `src/train.py` (LogReg, RandomForest, GradientBoosting; accuracy/precision/recall/F1/ROC-AUC/confusion matrix -> `models/metrics.json`) |
| 4. FastAPI `/predict`, `/health` | `app/main.py` |
| 5. GitHub repo | steps below |
| 6-7. CI/CD, Docker, GHCR | `.github/workflows/ci-cd.yml`, `Dockerfile` |
| 8-9. Trigger + verify | "Demo" section below |

## 1. Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Use the examiner's dataset: copy it to data/telco_churn.csv (Telco churn column names)
python -m src.train            # trains models, prints metrics, saves best model
pytest -v                      # run tests
uvicorn app.main:app --reload  # http://127.0.0.1:8000/docs
```
If the examiner's CSV uses different column names, edit the column lists at the top of
`src/preprocessing.py` (and the fields in `app/schemas.py`).

Test the API:
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
 "gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":3,
 "PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
 "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
 "StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month",
 "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
 "MonthlyCharges":95.5,"TotalCharges":286.5}'
```

## 2. Create the GitHub repository (Task 5)
```bash
git init
git add .
git commit -m "Initial commit: churn prediction system"
git branch -M main
git remote add origin https://github.com/<your-username>/churn-prediction-cicd.git
git push -u origin main
```
(Create the empty repo on github.com first. The first push triggers the workflow.)
Make sure the repo's **Settings -> Actions -> General -> Workflow permissions** is
"Read and write permissions" so the workflow can push to GHCR.

## 3. CI/CD pipeline (Tasks 6-7)
`ci-cd.yml` runs on every push / PR to `main`:
1. **test job** - checkout, set up Python 3.12, install dependencies, code validation
   (`compileall` + `flake8`), automated tests (`pytest`).
2. **build-and-push job** (only after tests pass) - builds the Docker image and, on pushes to
   `main`, publishes `ghcr.io/<owner>/<repo>:latest` and `:<short-sha>` using `GITHUB_TOKEN`.

## 4. Demo: change code -> pipeline triggers (Task 8)
Example change: raise the decision threshold and bump the version in `model_config.json`:
```json
{ "model_version": "1.1.0", "decision_threshold": 0.6, "model_path": "models/churn_model.joblib" }
```
```bash
git add model_config.json
git commit -m "Change threshold to 0.6, bump model version to 1.1.0"
git push
```
Open the repo -> **Actions** tab: a new "Churn ML CI/CD" run starts automatically.

## 5. Verify and run the new image (Task 9)
1. Actions -> click the run -> open each step to inspect logs (install, flake8, pytest, build, push).
2. Repo main page -> **Packages** shows the image in GHCR.
3. Pull and run the freshly built image:
```bash
echo <YOUR_GITHUB_PAT_with_read:packages> | docker login ghcr.io -u <username> --password-stdin
docker pull ghcr.io/<username>/churn-prediction-cicd:latest
docker run -d -p 8000:8000 --name churn ghcr.io/<username>/churn-prediction-cicd:latest
curl http://localhost:8000/health        # shows model_version "1.1.0"
```
(If the package is public no login is needed. Image names must be lowercase.)
Build locally instead: `docker build -t churn-api . && docker run -p 8000:8000 churn-api`

## Notes
- `data/telco_churn.csv` here is **synthetic practice data** (same schema as the IBM Telco
  dataset). Retrain on the real file before the exam: `python -m src.train`, then commit
  the new `models/churn_model.joblib`.
- `requirements.txt` pins scikit-learn/pandas/numpy to the versions the model was trained with;
  if you retrain on another version, update the pins.
