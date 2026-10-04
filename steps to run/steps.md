# Run the Telco Churn Project Locally

Project folder: `C:\Telco`

Architecture:

```
User input -> Streamlit UI -> FastAPI /predict -> saved preprocessing + XGBoost pipeline
           -> churn probability -> threshold (0.602) -> Churn / No Churn -> UI result
```

---

## Option A: Docker (recommended)

1. **Start Docker Desktop** and wait until it shows the engine is running.
2. **Open PowerShell and go to the project:**
   ```powershell
   cd C:\Telco
   ```
3. **Start both containers:**
   ```powershell
   docker compose up -d
   ```
   Add `--build` only if you changed code or retrained the model:
   ```powershell
   docker compose up -d --build
   ```
4. **Open the app:**

   | What | URL |
   |---|---|
   | UI | http://localhost:9124 |
   | API docs | http://localhost:9123/docs |
   | Health check | http://localhost:9123/health (should show `"threshold":0.602`) |

5. **Stop it when finished:**
   ```powershell
   docker compose down
   ```

Useful commands:

```powershell
docker compose ps            # status of the containers
docker compose logs api      # API logs
docker compose logs ui       # UI logs
```

---

## Option B: Without Docker (virtual environment)

Use two PowerShell windows, both started from `C:\Telco`. Keep both open while using the app.

**Window 1: API**
```powershell
cd C:\Telco
.venv\Scripts\activate
uvicorn app.main:app --host 127.0.0.1 --port 9123
```

**Window 2: UI**
```powershell
cd C:\Telco
.venv\Scripts\activate
streamlit run ui/streamlit_app.py --server.port 9124
```

Open http://localhost:9124. Stop each window with `Ctrl+C`.

---

## When something changes

| Situation | What to run |
|---|---|
| Changed the data or want to retrain | `python -m src.train`, then `python -m src.predict` to check, then `docker compose up -d --build` |
| Changed API or UI code | `docker compose up -d --build` |
| Run the tests | `python -m pytest tests -v` (with `.venv` active) |
| Installed or upgraded a package | Update the matching `requirements-*.txt` so Docker uses the same versions the model was trained with |

Retraining must be followed by a rebuild, because the Docker image contains a copy of `model/churn_xgb.joblib`.

---

## Quick sanity check

Send a known customer to the API (PowerShell):

```powershell
$body = @{
  gender="Female"; SeniorCitizen=0; Partner="No"; Dependents="No"; tenure=24
  PhoneService="Yes"; MultipleLines="No"; InternetService="Fiber optic"
  OnlineSecurity="No"; OnlineBackup="No"; DeviceProtection="No"
  TechSupport="No"; StreamingTV="No"; StreamingMovies="No"
  Contract="Month-to-month"; PaperlessBilling="Yes"
  PaymentMethod="Credit card (automatic)"; MonthlyCharges=70.0
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:9123/predict -Method Post -Body $body -ContentType "application/json"
```

Expected: `churn_probability` of about `0.551` and `prediction: No Churn` (threshold `0.602`).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `docker` not recognized, or engine connection error | Docker Desktop is not running yet. Start it, then open a **new** PowerShell window |
| `WinError 10013` or "ports are not available" | Windows has reserved the port. Change the port numbers: in `docker-compose.yml` edit the left-hand numbers (`9123`, `9124`); for Option B choose another high port |
| UI says "Cannot reach the API" | The API is not running. Check `docker compose ps`, or start Window 1 in Option B |
| Page does not load right after `up -d` | Wait about 20 seconds for the API health check to pass |
| `ModuleNotFoundError: No module named 'src'` or `'app'` | You are not in `C:\Telco`. Run `cd C:\Telco` first |
| `FileNotFoundError ... churn_xgb.joblib` | The model has not been trained. Run `python -m src.train` |
| `InconsistentVersionWarning` in the API logs | Library versions differ from the ones that saved the model. Fix `requirements-api.txt` and retrain |

---

## Project layout

```
Telco/
  data/telco_clean.csv       cleaned dataset
  src/
    preprocessing.py         data loading, split, preprocessing pipeline
    train.py                 trains, picks threshold, evaluates, saves model
    predict.py               loads model, predict_one(customer)
  model/churn_xgb.joblib     saved pipeline + threshold + metadata
  app/main.py                FastAPI app (/health, /predict)
  ui/streamlit_app.py        Streamlit form
  tests/test_api.py          pytest tests for the API
  requirements-api.txt       pinned API dependencies
  requirements-ui.txt        pinned UI dependencies
  Dockerfile                 API image
  Dockerfile.ui              UI image
  docker-compose.yml         runs API + UI together
  .dockerignore
```

---

## Final model (for reference)

| Item | Value |
|---|---|
| Model | XGBoost inside a scikit-learn pipeline |
| Decision threshold | 0.602 (best F1 on out-of-fold training predictions) |
| Test ROC-AUC | 0.8475 |
| Churn precision / recall / F1 | 0.58 / 0.73 / 0.64 |