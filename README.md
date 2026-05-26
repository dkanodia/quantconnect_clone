# Quantconnect Clone — Backtesting Engine + Team UI

A modular, event-driven backtesting engine paired with a multi-user Streamlit dashboard. Built for teams that need reproducible strategy research, role-based access control, and a clean path from local SQLite development to cloud PostgreSQL production.

**Key features**

- Event-driven and vectorized execution paths in the same engine
- Walk-Forward Optimisation (WFO) as a first-class feature
- Pluggable slippage, commission, and execution models via clean interfaces
- Optuna-powered hyperparameter optimisation with grid-search fallback
- Multi-user roles (`admin`, `analyst`, `viewer`) with per-run visibility controls
- Full tearsheets: equity curve, metrics strip, trade log, comment threads
- Side-by-side run comparison with overlaid equity curves
- In-app notification inbox with per-type badges and one-click mark-read

---

## Project structure

### `backtester/` — Engine

The engine is split into focused sub-packages. `core/` contains the event bus, order manager, and the `Backtester` orchestrator that drives both event-driven and vectorised execution. `data/` provides feed adapters (yfinance, CSV, synthetic). `execution/` houses the three pluggable interfaces — `SlippageModel`, `CommissionModel`, and `ExecutionModel` — with concrete implementations. `analytics/` computes metrics (Sharpe, CAGR, max drawdown, win rate) and produces `DictReporter` output that the UI persists as JSON. `optimize/` contains the `GridSearchOptimizer`, `OptunaOptimizer`, and `WalkForwardOptimiser`. `portfolio/` and `risk/` implement portfolio accounting and position-sizing risk models. `strategy/` defines the base `Strategy` ABC that user strategies extend.

### `ui/` — Dashboard

The UI layer is a strict three-layer stack: `ui/db.py` (SQLAlchemy 2.0 models + query helpers), `ui/auth.py` (bcrypt login, session state, role guards), and `ui/components/` (reusable Streamlit widgets: `run_card`, `equity_chart`, `metrics_grid`, `sidebar`). Pages in `ui/pages/` import only from `ui.auth`, `ui.db`, and `ui.components` — never from each other and never from the engine. `ui/app.py` is the single entry point that bootstraps the database, renders the sidebar, and routes between pages.

```
quantconnect_clone/
├── backtester/
│   ├── analytics/          # Metrics, reporters
│   ├── core/               # Event bus, order manager, backtester
│   ├── data/               # Feed adapters (yfinance, CSV, synthetic)
│   ├── execution/          # Slippage, commission, execution models
│   ├── optimize/           # Grid search, Optuna, WFO
│   ├── portfolio/          # Portfolio accounting
│   ├── risk/               # Risk / position-sizing models
│   ├── strategy/           # Strategy base class
│   ├── exceptions.py
│   └── interfaces.py
├── ui/
│   ├── components/
│   │   ├── equity_chart.py
│   │   ├── metrics_grid.py
│   │   ├── run_card.py
│   │   └── sidebar.py
│   ├── pages/
│   │   ├── admin.py
│   │   ├── compare.py
│   │   ├── dashboard.py
│   │   ├── login.py
│   │   ├── new_run.py
│   │   ├── notifications.py
│   │   ├── run_detail.py
│   │   ├── run_history.py
│   │   └── strategy_library.py
│   ├── app.py
│   ├── auth.py
│   └── db.py
├── tests/
│   ├── analytics/
│   ├── core/
│   ├── data/
│   ├── execution/
│   ├── optimize/
│   ├── portfolio/
│   ├── risk/
│   ├── strategy/
│   └── ui/
├── scripts/
│   └── smoke_test_wfo.py
├── .env.example
├── .gitignore
├── Procfile
├── railway.toml
├── requirements.txt
└── README.md
```

---

## Local setup

```bash
git clone <repo>
cd quantconnect_clone
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with your settings
python -m pytest tests/ -v          # verify engine: 998 tests
streamlit run ui/app.py             # launch UI
```

The app will be available at `http://localhost:8501`.

---

## First login

`init_db()` runs automatically on every start-up and is idempotent — it creates tables if they do not exist and is safe to call multiple times.

To seed the first admin account, set `ADMIN_EMAIL`, `ADMIN_NAME`, and `ADMIN_PASSWORD` in your `.env` file, then run:

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from ui.db import get_db, init_db
from ui.auth import seed_admin
import os
init_db()
with get_db() as db:
    seed_admin(db, os.environ['ADMIN_EMAIL'], os.environ['ADMIN_NAME'], os.environ['ADMIN_PASSWORD'])
print('Admin created.')
"
```

Then open `http://localhost:8501` and log in with the credentials you just set.

---

## Running tests

```bash
python -m pytest tests/ -v                        # all 998 tests
python -m pytest tests/data/ -v                   # data layer only
python -m pytest tests/ui/ -v                     # UI layer only
python scripts/smoke_test_wfo.py                  # WFO end-to-end smoke test
```

Tests use an in-memory SQLite database and mock all Streamlit calls — no running server or live network access required.

---

## Deploying to Railway

1. Push the repo to GitHub.
2. Create a new Railway project → **Deploy from GitHub repo**.
3. Add the **PostgreSQL** plugin → Railway automatically sets `DATABASE_URL` in your environment.
4. Set the following environment variables in the Railway dashboard:
   - `ADMIN_EMAIL`
   - `ADMIN_NAME`
   - `ADMIN_PASSWORD`
5. Railway reads `railway.toml` automatically — no further configuration is needed.
6. On first deploy, seed the admin account via Railway's **Shell** tab:
   ```bash
   python -c "
   from ui.db import get_db, init_db
   from ui.auth import seed_admin
   import os
   init_db()
   with get_db() as db:
       seed_admin(db, os.environ['ADMIN_EMAIL'], os.environ['ADMIN_NAME'], os.environ['ADMIN_PASSWORD'])
   print('Admin created.')
   "
   ```

---

## Deploying to Render

1. Push the repo to GitHub.
2. Create a **New Web Service** in Render → connect the repo.
3. Set **Build command**: `pip install -r requirements.txt`
4. Set **Start command**:
   ```
   streamlit run ui/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
   ```
5. Create a **PostgreSQL** database in Render → copy the **External Database URL** into an environment variable named `DATABASE_URL`.
6. Add environment variables: `ADMIN_EMAIL`, `ADMIN_NAME`, `ADMIN_PASSWORD`.
7. Seed the admin on first deploy using Render's **Shell** tab with the same command shown in the Railway section above.

---

## Architecture decisions

**Polars for data processing** — The engine's inner loops operate on DataFrames with lazy evaluation and columnar memory layout. Polars achieves 5–20× the throughput of pandas for the rolling-window, resample, and join operations common in backtesting, while keeping peak memory predictable. Pandas is retained for UI-facing output (DataFrames displayed by `st.dataframe`) because Streamlit's serialisation layer expects the pandas API.

**Three separate execution interfaces** — `SlippageModel`, `CommissionModel`, and `ExecutionModel` are independent protocols rather than a single monolithic execution class. This lets users swap realistic market-impact slippage into a strategy that uses a simpler fixed-commission broker without touching the order manager, and vice versa. It also makes unit testing each concern in isolation straightforward.

**WFO as a first-class feature** — Walk-Forward Optimisation is the primary defence against curve-fitting. Rather than treating it as a post-processing step, `WalkForwardOptimiser` is integrated directly into the optimiser hierarchy alongside `GridSearchOptimizer` and `OptunaOptimizer`. Every WFO fold runs a full in-sample optimisation and produces an out-of-sample performance record, so the fold results are first-class objects that can be persisted, compared, and charted.

**SQLite → PostgreSQL** — SQLite requires zero infrastructure for local development: no daemon, no Docker, no connection string. The `DATABASE_URL` environment variable defaults to a local `.db` file if unset. Switching to PostgreSQL for production requires only setting that one variable — SQLAlchemy's dialect layer handles the rest. This keeps the contributor workflow frictionless while making production deployments straightforward.

**`bcrypt<4.0` pin** — `passlib`'s `CryptContext` calls into `bcrypt`'s internal `._bcrypt` module, which was removed in `bcrypt 4.0`. Pinning `bcrypt>=3.2.0,<4.0.0` is the upstream-recommended workaround until `passlib` publishes a `4.x`-compatible release.

---

## Known limitations

- **No live trading** — this is a backtesting and research platform only. No order routing to any broker or exchange is implemented.
- **Threading, not a task queue** — background run execution uses Python's `threading` module. This is sufficient for small teams (up to ~10 concurrent users) but is not suitable for high-concurrency workloads. For larger teams, replace the thread-based dispatcher with a Celery worker backed by Redis and point `DATABASE_URL` at a connection-pooled PostgreSQL instance.
- **WFO memory usage** — Walk-Forward Optimisation with large parameter grids loads multiple full datasets into memory simultaneously (one per fold). For strategies with large universes or long histories, a machine with at least 2 GB RAM is recommended. Railway's Starter plan (512 MB) may OOM on non-trivial WFO runs; Railway Pro ($20/mo) or a dedicated VM is advised for production optimisation workloads.
