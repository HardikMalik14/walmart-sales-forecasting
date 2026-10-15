# Deploy Sales Pred Engine

This update belongs in the existing `HardikMalik14/walmart-sales-forecasting`
repository. It adds a FastAPI portfolio dashboard and preserves the original
research pipeline, model implementations, and archived outputs.

## 1. Apply the update with PowerShell

Download `Sales_Pred_Engine_Original_Repo_Vercel_Update.zip` to Downloads.
Use a fresh local folder to avoid overwriting uncommitted local work. Cloning
creates a local copy of your existing repository; it does not create a new
GitHub repository.

```powershell
cd "$HOME\Downloads"
git clone https://github.com/HardikMalik14/walmart-sales-forecasting.git Walmart-original-deploy
cd .\Walmart-original-deploy
Expand-Archive -LiteralPath "$HOME\Downloads\Sales_Pred_Engine_Original_Repo_Vercel_Update.zip" -DestinationPath . -Force
git add .
git commit -m "Add Sales Pred Engine dashboard and Vercel deployment"
git push origin main
```

If `Walmart-original-deploy` already exists, choose another unused local folder
name for the clone and use that name in the following `cd` command. Git may
open browser authentication during push; sign in to your GitHub account.

## 2. Import on Vercel

1. Open Vercel and choose **Add New → Project**.
2. Import your existing **walmart-sales-forecasting** repository.
3. Use project name **sales-pred-engine**. The site branding is **Sales Pred Engine**.
4. Use the detected **FastAPI** framework and root directory **./**.
5. Leave custom build, output, and install overrides disabled. No environment
   variables are required for this demo.
6. Deploy. If this repository already has a Vercel project, pushing the update
   should trigger its deployment; confirm these settings in that project.

The Python version is pinned to 3.12. The deployment entrypoint is `index:app`.
`requirements.txt` uses the CPU-only XGBoost distribution and omits the large
notebook and optional research-model stack. Keep `requirements-research.txt`
out of Vercel's install override.

## 3. Check the deployed site

- Benchmark explorer: saved four-model predictions, date filters, recalculated
  metrics, model selection, comparison chart, residuals, and CSV export.
- Forecast lab: click **Train & forecast**; change the seed and rerun to test
  different input data. Upload your own CSV or download a generated sample.
- Download the run ZIP and inspect future predictions, held-out predictions,
  sales history, and the JSON report.
- Methodology: archived and live results are clearly separated.

The benchmark's original XGBoost MAPE is about 1.84%. This is an archived
holdout result, not a promised accuracy for future dates or other data.

## CSV requirements

UTF-8 CSV under 1.5 MB, at most 8,000 rows and 20 columns. Each store needs
80–200 consecutive weekly Fridays on the same calendar as all other stores.
Dates accept `YYYY-MM-DD` or `DD-MM-YYYY`. No duplicate Store/Date pairs.

| Column | Requirement |
| --- | --- |
| Date | Required; consecutive Fridays |
| Weekly_Sales | Required; finite numeric values within ±10 billion |
| Store | Optional; positive integer ID; defaults to 1; at most 45 stores |
| Holiday_Flag | Optional; 0 or 1; otherwise original US calendar is used |
| Temperature | Optional; Fahrenheit between -100 and 150; defaults to 60°F |
| Fuel_Price, CPI, Unemployment | Optional finite numbers; not used by the two live models |

The live lab supports horizons of 1–16 weeks and chronological holdouts of
15–30%. Its synthetic generator supports 1–10 stores and 80–180 weeks.

## Modeling and hosting limits

The four-model benchmark comes from existing `outputs/metrics.json` and
`outputs/predictions.csv`. Date selection recomputes metrics without retraining.
The live lab fits Seasonal Naive and XGBoost using the original wrappers.
XGBoost uses a bounded 300-tree probe, an eight-week validation window, the
original final tree-count rule, and one CPU thread. This is a smaller live
configuration than the archived research experiment.

Evaluation uses only training targets for holdout prediction. Models are then
refit on full supplied history to predict future dates. Future temperatures
hold the last value per store plus your chosen scenario shift; holiday flags
use the original US calendar. Future sales have no known accuracy or calibrated
prediction intervals. The saved benchmark excludes Q4 peaks and uses historical
covariates; it should be presented with the original README's caveats.

Uploads and trained models are not persisted. The page keeps the current run
until refreshed. Download the ZIP to retain results. One live run at a time
per process is allowed; a busy response can be retried. Cold starts and larger
uploads can take longer, and hosting execution limits still apply.

## Local development and validation

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn index:app --reload
```

For all tests, install `requirements-dev.txt`, then run
`python -m pytest tests -q`. For the original CLI and notebooks, install
`requirements-research.txt` and follow the original README pipeline instructions.

Validation performed for this update: 45 tests passed, covering the original
models and seven web tests. Browser checks covered model/date selection,
exports, live fits, changed seed, CSV upload, and mobile layouts. These are
local checks; verify the final Vercel build and deployed URL after importing.

References: [Vercel FastAPI](https://vercel.com/docs/frameworks/backend/fastapi),
[XGBoost installation](https://xgboost.readthedocs.io/en/stable/install.html).
