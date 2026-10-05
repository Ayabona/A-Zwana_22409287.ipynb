# Johannesburg Election Explorer

A local Streamlit dashboard for the supplied Johannesburg election files. Historical observations (2011, 2016 and 2021), derived measures and conditional 2026 estimates are labelled separately. The original notebook and source files are preserved.

## Run

Use Python 3.11 or later from this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address shown by Streamlit, normally http://localhost:8501.
After setup, you can also double-click **Start Dashboard.bat**, which opens the dashboard on port 8502. If it is already running, visit http://localhost:8502 instead of starting a second copy.

## Examiner views

- **Historical EDA · 2011–2021:** PR/Ward ballot selector, election-year range, municipality/ward scope and party selection; weighted turnout, party trends, ward variation, descriptive statistics and CSV download. Empty selections and wards absent in a given year have explicit messages. Party filters never change the vote-share denominator or turnout.
- **Predictions · 2026:** four numbered outputs matching the supplied brief: (1) metro vote-share leader and coalition interpretation, (2) conditional leaders in three selected wards, (3) party shares and totals, (4) voter count and turnout rate. Defaults are wards 1, 50 and 100 from the notebook. Metro outputs always include all 135 wards from 2021. Forecast downloads include their status and assumptions.
- **Model & data:** later-election backtest, regression and leader metrics, comparison models, district-ID audit, definitions, local sources and limitations.

## Sources and calculations

Keep all six `jhb_YYYY_detailed_results.csv` / `jhb_YYYY_voter_turnout.xls` files beside `app.py`. `election_data.py` reads their actual schemas. It handles the UTF-16/UTF-8 differences and each turnout workbook's changing header position without guessing column positions. Original download URLs/acquisition dates were not supplied; they must be added for independently traceable provenance before submission.

PR and Ward ballots remain separate. Registered voters and spoilt votes repeat on every party row; metadata is counted once per voting district and ballot. Turnout uses the spreadsheet's reported voter count divided by **registered voters + MEC7 votes**. The report defines its voter count using the higher Ward/PR ballot count; it is a turnout proxy. Metro rates are calculated from summed counts, never an unweighted mean of ward percentages.

Seven named party categories (ANC, DA, EFF, ActionSA, IFP, PA, VF+) and Other exhaust the valid votes. Other is retained in every denominator and forecast normalization. It combines unrelated smaller parties/independents and is never declared a winning party. A zero party share in an election means no votes under that standardized category in that election's supplied records; it does not imply that every party existed or contested that election.

## Model and uncertainty

`election_model.py` implements the notebook's Random Forest model family in a standalone, corrected pipeline. It does **not** reproduce the notebook's saved numerical forecasts:

1. The notebook's saved output used ward IDs as registered-voter counts. This dashboard uses the actual registered-voter column and reconciles counts against the results files.
2. The notebook combined ballot types and omitted Other from final share normalization. The dashboard fits ballot-specific share models with all categories, including material parties previously pooled in Other.
3. Instead of two lags and a random split of 2021 wards, the dashboard uses one-election-lag shares and turnout. Train on 2011 → 2016, test on 2016 → 2021, then refit on both transitions for 2021 → 2026. This permits an actual later-election backtest with three cycles. There are 130 training pairs and 135 testing pairs.
4. Each output uses a deterministic Random Forest regressor with 150 trees, minimum leaf size 3 and random seed 42. Shares are clipped and normalized; turnout is bounded to 0–100%. The turnout model uses only previous turnout, giving the same projected people count for PR and Ward ballots. Ridge (standardized inputs, alpha 10) and last-election persistence are displayed as benchmarks. Random Forest is retained for continuity with the notebook, not claimed to dominate every benchmark.
5. Forecast voter counts hold the 2021 registered + MEC7 electorate constant. Forecast valid votes apply the ward's 2021 valid-votes/reported-voters ratio for that ballot. Party totals sum to valid votes; people are never summed across ballot types.
6. Displayed sensitivity ranges are point estimates ± historical ward RMSE, bounded to 0–100%. These are stress tests, **not 95% confidence/prediction intervals**. No unsupported coalition probability, seat majority, winner probability or governance prediction is produced.

The 2026 date is taken from the supplied examination scenario, not independently asserted as a confirmed election schedule. No election data later than 2021 are used. ActionSA appears only in 2021 in these inputs, limiting its extrapolation and making the earlier-election backtest particularly demanding for emerging parties.

## Geography and remaining analytical limits

The history and model join recorded ward IDs. There are 130 in 2011 and 135 in 2016/2021. Equal IDs do not establish equal boundaries. `geography_audit` compares district-ID membership and exposes changes; it is **not spatial reconciliation**. All three default case-study wards have district membership changes. No verified boundary crosswalk or 2026 delimitation was supplied, so actual 2026 ward-boundary leader forecasts remain unsupported. The dashboard reports conditional estimates for recorded ward-ID series and states this beside the results.

The notebook's unverified descriptions of the selected wards as swing/demographically changing wards are not repeated. Selection retains its case studies and presents their measured 2021 profiles for comparison. A stronger justification and actual spatial alignment require additional source evidence. This dashboard implementation does not imply that all requirements for the wider notebook assessment are complete.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests reconcile known source totals and turnout denominators, verify full forecast coverage and conservation of shares/votes, and exercise the three views and filter edge cases using [Streamlit's AppTest](https://docs.streamlit.io/develop/api-reference/app-testing).

`app.py` owns presentation; `election_data.py` owns data preparation; `election_model.py` owns forecasting/evaluation. Source changes invalidate the Streamlit data cache using file size and modification time.
