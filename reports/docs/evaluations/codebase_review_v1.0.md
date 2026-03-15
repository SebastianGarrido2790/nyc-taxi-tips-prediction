# NYC Taxi Tips Prediction — Codebase Review & Production Readiness Assessment

**Reviewer:** Antigravity (Automated Review)
**Date:** 2026-03-14
**Version:** 1.0 (First Pass)
**Scope:** Full codebase — 22 Python source files, 9 test files, 2 CI workflows, 3 YAML configs, 2 Dockerfiles + docker-compose, `pyproject.toml`, Streamlit dashboard, and 21 documentation files.

---

## Overall Verdict

The **NYC Taxi Tips Prediction System** is a **well-structured portfolio project** that demonstrates strong understanding of FTI (Feature-Training-Inference) pipeline architecture, agentic AI orchestration via LangGraph, and containerized deployment with Docker Compose. The project successfully implements a 6-stage DVC pipeline, a ReAct-based LangGraph agent with Gemini-2.5-flash, and a FastAPI inference microservice — all backed by MLflow experiment tracking and comprehensive Streamlit visualization.

The project is clearly above average for a portfolio ML project. However, several gaps remain that prevent it from reaching **elite production-grade** status. This review catalogs every finding with actionable recommendations.

**The foundation is strong. What follows are the specific weaknesses and gaps that, once addressed, will elevate this from "impressive portfolio project" to "production-grade reference architecture worthy of FAANG/elite employers."**

---

## 1. Strengths ✅

### 1.1 Architecture & Design

| Strength | Evidence |
|:---|:---|
| **FTI Pattern** | Clear 6-stage DVC pipeline (Ingestion → Validation → Transformation → Feature Engineering → Training → Evaluation) with explicit artifact handoffs between each stage |
| **Brain vs. Brawn** | Agent reasons via LangGraph (`src/agents/`); the deterministic tool (`src/tools/taxi_prediction_tool.py`) handles all HTTP execution and Pydantic validation — clean separation |
| **No Naked Prompts** | System prompt centralized and versioned in [prompts.py](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/agents/prompts.py) (`v1.2`), separated from agent execution logic |
| **Config Separation** | Three-tier YAML config (`config.yaml` for paths, `params.yaml` for hyperparameters, `schema.yaml` for data contracts) |
| **Pydantic Config Entities** | Configuration entities in [config_entity.py](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/entity/config_entity.py) use `BaseModel` instead of raw frozen dataclasses — Pydantic validates at construction time |
| **Modular Pipeline** | Each stage has its own component class, pipeline script, and configuration entity — clean Conductor/Worker separation of concerns |
| **Environment-Aware MLflow** | [mlflow_config.py](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/utils/mlflow_config.py) implements a 3-level priority chain (env var → env-based default → YAML fallback) with production runtime guard |

### 1.2 Agentic Layer

| Strength | Evidence |
|:---|:---|
| **ReAct Pattern** | LangGraph `create_react_agent` with tool-binding — agent decides autonomously when to call the prediction tool |
| **Strict Tool Validation** | `TaxiRideInput` uses Pydantic `BaseModel` with `Field(..., gt=0)` constraints — catches LLM hallucinations before they hit the network |
| **Custom Domain Exception** | `PredictionToolError` wraps all backend failures (timeout, HTTP error, network error) into a single domain-specific exception |
| **Agent Config Guard** | `AgentConfigError` raised immediately when `GOOGLE_API_KEY` is missing — fail-fast principle |
| **Gemini Response Normalization** | `app.py` handles Gemini's `list[dict]` block-based content format, normalizing it to plain strings |
| **Graceful Error UX** | `app.py` categorizes errors into Brain (quota), Brawn (API offline), and Unknown — users get contextual troubleshooting guidance |

### 1.3 MLOps & CI/CD

| Strength | Evidence |
|:---|:---|
| **DVC Pipeline** | Full DAG with `deps`, `params`, `outs`, and `metrics` — reproducible and cacheable |
| **MLflow Integration** | Experiment tracking, metric logging, model registry, multi-experiment separation (Training vs. Evaluation) |
| **Multi-Metric Champion Selection** | Weighted min-max normalization across MAE/MSE/R² with configurable weights in `params.yaml` — avoids single-metric bias |
| **Temporal Splitting** | Train (Jan–Aug) / Val (Sept–Oct) / Test (Nov–Dec) prevents look-ahead bias — critical for time-series ML |
| **CI Pipeline** | Lint + Format checks (Ruff) and unit tests (Pytest) gated on every push to `main` |
| **Containerized Deployment** | Separate `backend.Dockerfile` + `frontend.Dockerfile` with `docker-compose.yml` using health-checked service dependency |
| **Subsample Mode** | `subsample_fraction` in `params.yaml` enables fast local iteration without modifying code |

### 1.4 Testing

| Strength | Evidence |
|:---|:---|
| **9 Test Modules** | Covers ingestion, transformation, feature engineering, model training, model evaluation, predict model, API endpoints, agent tools, and agent integration |
| **Mock Strategy** | Tests use `unittest.mock.patch` to isolate MLflow, HTTP calls, and file I/O — no external dependencies during testing |
| **Edge Cases** | Pydantic validation failures (negative distance, invalid hour, excessive passengers), timeout handling, empty ride list, and missing model scenarios |
| **Polars In-Memory** | Shared `conftest.py` fixtures use in-memory Polars DataFrames with intentional anomalies for blazing-fast assertions |

### 1.5 Documentation

| Strength | Evidence |
|:---|:---|
| **Five Pillars** | Reports follow the full `architecture/`, `decisions/`, `evaluations/`, `references/`, `runbooks/`, `workflows/` taxonomy — 21 report files total |
| **Module Docstrings** | Every Python file has a module-level docstring explaining purpose and architectural context |
| **Google-style Docstrings** | Functions and classes document Args, Returns, and Raises — consistent throughout |
| **README Excellence** | Rich badges, FTI architecture diagram, dashboard screenshots, full setup instructions, tech stack justification table |
| **Launch Script** | `launch_app.bat` provides automated Windows dev experience with dependency sync, API warm-up, and clear user instructions |

### 1.6 Data Processing

| Strength | Evidence |
|:---|:---|
| **Polars for ETL** | Data ingestion, validation, and transformation use Polars for memory-efficient processing on multi-million row datasets |
| **Cyclical Feature Engineering** | Sin/Cos encoding for hour, day-of-week, and month preserves temporal proximity (e.g., hour 23 ↔ hour 0) |
| **Robust Cleaning Rules** | Negative financial amounts, impossible distances, outlier fares — all handled with explicit filters and logging |

---

## 2. Weaknesses & Gaps 🔴

### 2.1 CRITICAL: Untyped `dict` Fields in Pydantic Config Entities

> [!CAUTION]
> Five `dict` fields in [config_entity.py](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/entity/config_entity.py) are declared as bare `dict` without type parameters. This violates **Rule 2.3** (No untyped dictionaries) and completely undermines the type safety that Pydantic entities are supposed to provide.

| Entity | Field | Current | Should Be |
|:---|:---|:---|:---|
| `DataIngestionConfig` | `all_schema` | `dict` | `dict[str, Any]` or a typed `SchemaConfig` model |
| `DataValidationConfig` | `all_schema` | `dict` | `dict[str, Any]` or a typed `SchemaConfig` model |
| `ModelTrainerConfig` | `all_params` | `dict` | `dict[str, dict[str, Any]]` |
| `ModelTrainerConfig` | `selection_metrics` | `dict` | `dict[str, float]` |
| `ModelEvaluationConfig` | `all_params` | `dict` | `dict[str, dict[str, Any]]` |

**Impact:** Any typo in YAML keys (e.g., `selction_metrics` instead of `selection_metrics`) silently passes validation and produces `KeyError` at runtime instead of at construction time.

**Recommendation:**
1. At minimum, add generic type parameters: `all_schema: dict[str, Any]`.
2. Ideally, create typed Pydantic sub-models:
```python
class SelectionMetrics(BaseModel):
    mae: float = 0.0
    mse: float = 0.0
    r2: float = 0.0
```

---

### 2.2 CRITICAL: No `pyright` Configuration or CI Enforcement

> [!WARNING]
> `pyproject.toml` declares `[tool.mypy]` with `strict = true` but your GEMINI.md Standard mandates **pyright** over mypy. There is **no** `[tool.pyright]` section, no `pyright` in dependencies, and no type-checking CI step at all. The "100% type hint coverage" standard from your rules is not enforced.

**Gaps found:**
- `ConfigurationManager.__init__` parameters have no type annotations for `config_filepath`, `params_filepath`, `schema_filepath`
- `read_yaml()` returns `dict` but no generic type parameter
- `model_utils.get_feature_importances()` returns ambiguous `tuple[list[str], list[float]]` (docstring says "Returns (None, None) if unsupported" but the return type doesn't allow `None`)
- `taxi_prediction_tool.py` uses legacy `typing.List`, `typing.Dict`, `typing.Optional` instead of modern PEP 604 builtins (`list`, `dict`, `X | None`)
- `logger.py` uses `Optional[str]` instead of `str | None`

**Recommendation:**
1. Replace `[tool.mypy]` with `[tool.pyright]` in `pyproject.toml`:
```toml
[tool.pyright]
pythonVersion = "3.11"
typeCheckingMode = "standard"
```
2. Add `pyright` to dev dependencies: `"pyright>=1.1.350"` or `"basedpyright>=1.0.0"`
3. Add a CI step: `uv run pyright src/`
4. Replace all `typing.List`, `typing.Dict`, `typing.Optional` with builtins
5. Fix `ConfigurationManager.__init__` parameter annotations:
```python
def __init__(
    self,
    config_filepath: Path = CONFIG_FILE_PATH,
    params_filepath: Path = PARAMS_FILE_PATH,
    schema_filepath: Path = SCHEMA_FILE_PATH,
) -> None:
```

---

### 2.3 CRITICAL: Missing `.env.example` File

> [!CAUTION]
> No `.env.example` file exists in the repository. While `.env` is correctly gitignored, new contributors or reviewers have **no way to know** what environment variables the project requires without reading the source code or README manually.

**Impact:** Poor developer onboarding. First-time users will encounter silent failures if they miss configuring `GOOGLE_API_KEY`, `API_URL`, or `MLFLOW_TRACKING_URI`.

**Recommendation:**
Create a `.env.example` at the project root:
```env
# Define the environment (local/staging/production)
ENV=local

# Frontend → Backend communication
API_URL=http://localhost:8000

# Agentic Layer — get a key at https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=AIza...

# MLflow Tracking
MLFLOW_TRACKING_URI=file:./mlruns
```

---

### 2.4 HIGH: Hardcoded Target Column & Magic Numbers

| Location | Issue |
|:---|:---|
| [model_trainer.py:67](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/components/model_trainer.py#L67) | `target = "tip_amount"` hardcoded — should come from `schema.yaml` `TARGET_COLUMN.name` |
| [model_evaluation.py:52](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/components/model_evaluation.py#L52) | `target = "tip_amount"` duplicated hardcoded |
| [predict_model.py:55](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/components/predict_model.py#L55) | `"tip_amount"` hardcoded again |
| [data_transformation.py:98-101](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/components/data_transformation.py#L98-L101) | Filter thresholds (`0.5`, `100`, `3.70`, `1000`) are magic numbers |
| [predict_api.py:88-89](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/api/predict_api.py#L88-L89) | `PULocationID: 132.0` and `DOLocationID: 236.0` are hardcoded "default" location IDs with no explanation |
| [predict_api.py:96-98](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/api/predict_api.py#L96-L98) | `extra: 0.0`, `mta_tax: 0.5`, `improvement_surcharge: 0.3` are hardcoded NYC-specific constants |
| [predict_api.py:84](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/api/predict_api.py#L84) | `VendorID: 1.0` hardcoded default |
| [feature_engineering.py:101-103](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/components/feature_engineering.py#L101-L103) | Temporal split boundaries (1–8, 9–10, 11–12) are hardcoded |

**Impact:** `schema.yaml` already defines `TARGET_COLUMN.name: tip_amount` — but no component reads it. If the target column changes, you must grep through 3+ files. The magic numbers for data filtering make the pipeline brittle and non-auditable.

**Recommendation:**
1. Add `target_column: str` to `ModelTrainerConfig`, `ModelEvaluationConfig` entities and populate from `schema.yaml`.
2. Move filter thresholds to `params.yaml`:
```yaml
DataCleaning:
  min_trip_distance: 0.5
  max_trip_distance: 100
  min_total_amount: 3.70
  max_total_amount: 1000
```
3. Move the NYC-specific fare constants to a `nyc_fare_defaults` section in `params.yaml`.
4. Move temporal split boundaries to config:
```yaml
FeatureEngineering:
  train_months: [1, 8]
  val_months: [9, 10]
  test_months: [11, 12]
```

---

### 2.5 HIGH: Inconsistent Logger Import in `model_trainer.py`

> [!IMPORTANT]
> [model_trainer.py:24](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/components/model_trainer.py#L24) imports `from src.utils.common import logger` — using the module-level singleton logger from `common.py`. Every other component in the system uses `from src.utils.logger import get_logger; logger = get_logger(__name__)`.

**Impact:** Log messages from `ModelTrainer` appear under the logger name `src.utils.common` instead of `src.components.model_trainer`, making log filtering and debugging harder.

**Recommendation:** Replace with:
```python
from src.utils.logger import get_logger
logger = get_logger(__name__)
```

---

### 2.6 HIGH: Missing `__init__.py` in `src/api/`

> [!WARNING]
> The `src/api/` directory has no `__init__.py` file, while all other subpackages under `src/` do. This can cause import resolution issues with `pyright` and packaging tools.

**Recommendation:** Add an empty `src/api/__init__.py`.

---

### 2.7 HIGH: No `pytest-cov` and No Coverage Gate in CI

> [!WARNING]
> `pytest-cov` is not listed anywhere in `pyproject.toml`. The CI pipeline runs `uv run pytest tests/` without any coverage reporting or threshold enforcement. Test coverage can silently regress.

**Recommendation:**
1. Add `"pytest-cov>=4.1.0"` to `[project.optional-dependencies] dev`.
2. Update CI step:
```yaml
- name: Execute Deterministic Tests (Pytest)
  run: uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=50
```

---

### 2.8 HIGH: No `ruff` Configuration in `pyproject.toml`

> [!IMPORTANT]
> `ruff>=0.15.4` is declared as a dev dependency and two CI workflows run `ruff check` and `ruff format`, but there is **no `[tool.ruff]` section** in `pyproject.toml`. This means Ruff runs with default rules — no import sorting enforced, no f-string enforcement, no explicit rule selection.

**Recommendation:** Add a `[tool.ruff]` section:
```toml
[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "N", "W", "B", "SIM", "C4", "RUF"]

[tool.ruff.lint.isort]
known-first-party = ["src"]
```

---

### 2.9 MEDIUM: Cyclical Feature Encoding Mismatch Between Training and Inference

> [!WARNING]
> The cyclical encoding in [feature_engineering.py](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/components/feature_engineering.py#L64-L82) uses Day-of-Week (1–7, shifted by -1 → 0–6) divided by 7, while [predict_api.py](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/api/predict_api.py#L78-L81) uses Day-of-Month (1–31) divided by 31. These are **different features**.

Training pipeline creates:
- `pickup_day_sin = sin(2π * (weekday - 1) / 7)` ← **day of week** (Monday=0 through Sunday=6)

API preprocessing creates:
- `pickup_day_sin = sin(2π * day / 31)` ← **day of month** (1–31)

**Impact:** **Training-serving skew.** The model learned from day-of-week cyclical patterns but receives day-of-month patterns at inference time. This silently degrades prediction quality.

**Recommendation:**
1. The API preprocessing must replicate the exact same feature engineering as the training pipeline.
2. Extract the cyclical feature logic into a shared utility function (DRY principle):
```python
# src/utils/feature_utils.py
def encode_cyclical(value: float, period: float) -> tuple[float, float]:
    """Returns (sin, cos) cyclical encoding."""
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)
```
3. Both `feature_engineering.py` and `predict_api.py` should import and use this shared function.

---

### 2.10 MEDIUM: `_preprocess_request()` Hardcodes Feature Order Instead of Using Model Metadata

> [!IMPORTANT]
> [predict_api.py:137-142](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/api/predict_api.py#L137-L142) dynamically aligns columns using `model.feature_names_in_`, which is good. However, the `_preprocess_request()` function hardcodes ALL feature names and default values — if the training pipeline adds or removes a feature, `predict_api.py` must be manually updated.

**Recommendation:** Generate a feature schema artifact during training (e.g., `artifacts/model_trainer/feature_schema.json`) and load it at API startup to auto-align the preprocessing. This eliminates manual synchronization between training and serving.

---

### 2.11 MEDIUM: Legacy `typing` Imports in `taxi_prediction_tool.py`

[taxi_prediction_tool.py:8](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/tools/taxi_prediction_tool.py#L8) uses:
```python
from typing import List, Dict, Any, Optional
```

Since the project requires Python ≥3.10, these should be replaced with modern builtins:
```python
# Instead of: List[TaxiRideInput]
# Use:        list[TaxiRideInput]

# Instead of: Dict[str, Any]
# Use:        dict[str, Any]

# Instead of: Optional[str]
# Use:        str | None
```

Similarly, [logger.py](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/utils/logger.py#L14) uses `from typing import Optional` — should use `str | None`.

---

### 2.12 MEDIUM: `ModelTrainerConfig` Accepts an Undeclared Field `test_data_path`

In [test_model_trainer.py:25](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/tests/unit/test_model_trainer.py#L25), the fixture creates a `ModelTrainerConfig` with `test_data_path=tmp_path / "test.parquet"` — but this field **does not exist** in the `ModelTrainerConfig` model. This only works because Pydantic v2's default `model_config` allows extra fields. If `model_config = ConfigDict(extra="forbid")` were set (which it should be for strict validation), this test would break.

**Recommendation:**
1. Add `model_config = ConfigDict(extra="forbid")` to all config entities (strict mode).
2. Fix the test fixture to not pass `test_data_path`.

---

### 2.13 MEDIUM: `predict_model.py` Uses Hardcoded Default Paths

[predict_model.py:36-37](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/components/predict_model.py#L36-L37):
```python
predictions_dir: str = "artifacts/predictions",
output_filename: str = "inference_results.csv",
```

These default paths should come from `config.yaml` through the `ConfigurationManager`, not be passed as string arguments. This breaks the single-source-of-truth principle for paths.

---

### 2.14 MEDIUM: `read_yaml()` Returns Raw `dict` — No Validation

[common.py:21](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/utils/common.py#L21) `read_yaml()` returns a raw `dict`. The `ConfigurationManager` then accesses keys with bracket notation (`self.config["data_ingestion"]`). Any typo in a key name produces a runtime `KeyError` with no context.

**Impact:** This is the same problem originally caused by `ConfigBox` — just without the attribute-style access. The pipeline fails deep inside execution instead of at startup.

**Recommendation:** Create typed Pydantic models for the YAML structure:
```python
class AppConfig(BaseModel):
    artifacts_root: str
    data_ingestion: DataIngestionYamlConfig
    data_validation: DataValidationYamlConfig
    # ...
```
Parse `config.yaml` directly into `AppConfig` — any missing key raises a clear Pydantic `ValidationError` at startup.

---

### 2.15 MEDIUM: No `py.typed` Marker

> [!NOTE]
> No `py.typed` marker file exists in `src/`. This file signals PEP 561 compliance to downstream consumers and type checkers. Its absence means `pyright` may not fully analyze the package.

**Recommendation:** Create an empty file at `src/py.typed`.

---

### 2.16 LOW: `docker-compose.yml` Uses Deprecated `version` Key

[docker-compose.yml:1](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/docker-compose.yml#L1) declares `version: "3.8"`. This key has been [deprecated since Docker Compose v2](https://docs.docker.com/compose/releases/migrate/) and is ignored. Its presence suggests the project hasn't been updated to modern Docker Compose practices.

**Recommendation:** Remove the `version` line entirely.

---

### 2.17 LOW: Root `Dockerfile` Is a Stub

[Dockerfile](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/Dockerfile) uses `FROM scratch` — it's intentionally a no-op stub with a documentation header redirecting to the specialized Dockerfiles. While the intent is clear, having a file named `Dockerfile` that does nothing can confuse CI/CD tools and colleagues.

**Recommendation:** Either:
1. Rename to `Dockerfile.IGNORE` or delete entirely, OR
2. Convert into a multi-stage build that builds both services (more impressive for portfolio).

---

### 2.18 LOW: No Security Scanning in CI

| Gap | Impact |
|:---|:---|
| No `bandit` or `safety` step | Vulnerable dependencies or insecure code patterns ship undetected |
| No dependency audit | `pip audit` or `uv audit` should verify known CVEs |

**Recommendation:** Add a security job to CI:
```yaml
- name: Security Scan
  run: |
    uv run pip install bandit safety
    uv run bandit -r src/ -ll
    uv run safety check
```

---

### 2.19 LOW: `black` Listed as Dev Dependency Alongside `ruff`

`pyproject.toml` line 40: `"black>=23.9.0"` is listed as a dev dependency, but the project uses `ruff format` for formatting. Having both formatters creates confusion about which is authoritative.

**Recommendation:** Remove `black` from dependencies since `ruff format` replaces it entirely.

---

### 2.20 LOW: `ModelEvaluationPipeline.__init__` Has Empty Body

[stage_06_model_evaluation.py:25-27](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/pipeline/stage_06_model_evaluation.py#L25-L27):
```python
def __init__(self):
    """Initializes the ModelEvaluation Pipeline."""
    pass
```

This is the only pipeline class where `ConfigurationManager` is **not** instantiated in `__init__()` (it's created inside `main()` instead). All other pipeline classes instantiate it in `__init__()`. This inconsistency breaks the established pattern.

---

### 2.21 LOW: `app.py` Is a 481-Line Monolith

The Streamlit application `app.py` contains both pages (Dashboard & Agentic Chat), all data loading functions, custom CSS, and agent invocation logic in a single 481-line file. For a project emphasizing separation of concerns, this is an outlier.

**Recommendation:** Split into:
```
src/
  app/
    __init__.py
    main.py          # st.set_page_config, sidebar, routing
    pages/
      dashboard.py   # Page 1: Dashboard & Evaluation
      chat.py         # Page 2: Agentic Chat UI
    styles.py         # Custom CSS
    data_loaders.py   # Cached loading functions
```

---

## 3. Recommendations for Portfolio Differentiation 🚀

These are enhancements that go beyond "fixing gaps" and would make this project **stand out to elite employers**:

### 3.1 Add a `Makefile` or `justfile` for Developer Experience

Consolidate common commands into a single entry point:
```makefile
.PHONY: lint test typecheck pipeline docker serve

lint:
	uv run ruff check . && uv run ruff format --check .

test:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

typecheck:
	uv run pyright src/

pipeline:
	uv run dvc repro

docker:
	docker compose up --build

serve:
	uv run uvicorn src.api.predict_api:app --reload --port 8000
```

### 3.2 Add Pre-commit Hooks (Rule 3.3)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/RobertCraiworthy/pyright-python
    rev: v1.1.350
    hooks:
      - id: pyright
```

### 3.3 Add Great Expectations (GX) Data Validation (Rule 2.1)

The current `DataValidation` component only checks column presence and does a basic negative-distance scan. Production-grade validation should also enforce:
- Value ranges (e.g., `trip_distance` between 0.1 and 200 miles)
- Null percentage thresholds (e.g., nulls in `passenger_count` < 5%)
- Distribution drift detection (compare current batch against reference statistics)
- Data freshness checks (timestamp recency)

Replace or augment with Great Expectations suites stored as versioned artifacts.

### 3.4 Add API Versioning

[predict_api.py](file:///c:/Users/sebas/Desktop/nyc-taxi-tips-prediction/src/api/predict_api.py) mounts endpoints at `/predict` and `/health`. For production readiness:
```python
from fastapi import APIRouter
api_router = APIRouter(prefix="/v1")
# Move all endpoints to the router
app.include_router(api_router)
```

This is trivial but signals production awareness and backward compatibility planning.

### 3.5 Add Structured JSON Logging for Production

The current logger uses human-readable format (`%(asctime)s | %(levelname)s | %(name)s | %(message)s`). For observability platforms (Datadog, ELK, CloudWatch), switch to JSON:
```python
import json_log_formatter

handler = logging.StreamHandler()
handler.setFormatter(json_log_formatter.JSONFormatter())
```

### 3.6 Add OpenTelemetry Tracing (Rule 4.2)

Replace all `print()` debugging with structured traces:
```toml
# pyproject.toml
"opentelemetry-api>=1.20.0"
"opentelemetry-sdk>=1.20.0"
"opentelemetry-instrumentation-fastapi>=0.41b0"
```

This gives span-level visibility into agent decisions, tool calls, token usage, and latency — completely aligned with AgentOps rules.

### 3.7 Add LLM-as-a-Judge Evaluation Framework

Implement automated agent evaluation per your skills:
- **Relevance:** Does the tip prediction match the ride description?
- **Faithfulness:** Are cited numbers grounded in tool outputs?
- **Tool Usage Accuracy:** Did the agent call `predict_taxi_tip` with correct arguments?
- Store eval results in `reports/docs/evaluations/` and track them with MLflow.

### 3.8 Add a `CONTRIBUTING.md`

Document the development workflow, testing strategy, and code standards. This demonstrates team-readiness and engineering maturity.

### 3.9 Extract Shared Feature Engineering into a Utility Module

Create a `src/utils/feature_utils.py` containing the cyclical encoding logic shared between training and inference. This eliminates the training-serving skew identified in §2.9 and follows DRY.

### 3.10 Add Model Card (`reports/docs/evaluations/model_card.md`)

Following the [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) framework:
- Model description and intended use
- Training data characteristics
- Evaluation metrics and performance
- Limitations and ethical considerations
- Versioning and change log

---

## 4. Summary Scorecard

| Category | Score | Notes |
|:---|:---:|:---|
| **Architecture** | 9/10 | FTI pattern, clean separation, modular pipeline, conductor/worker split |
| **Agentic Design** | 8.5/10 | ReAct pattern, tool validation, centralized prompts, but no HITL or multi-agent |
| **Code Quality** | 6.5/10 | Good docstrings but 5 bare `dict` types, legacy `typing` imports, hardcoded values, inconsistent logger |
| **Type Safety** | 4/10 | `mypy` declared but not the standard (`pyright`), no type checking CI step, untyped config entities |
| **Testing** | 7.5/10 | 9 test modules with good edge cases, but no `pytest-cov`, no coverage gate, no integration tests |
| **CI/CD** | 6/10 | Lint + test only — no type checking, no coverage, no security scanning |
| **Security** | 5/10 | `.env` gitignored but no `.env.example`, no `bandit`/`safety`, duplicate `black` dependency |
| **Documentation** | 9/10 | Exemplary taxonomy, consistent docstrings, 21 report files, excellent README |
| **DevOps Maturity** | 7.5/10 | Docker Compose, health checks, but deprecated `version` key, stub Dockerfile, no `Makefile` |
| **Training-Serving Integrity** | 6/10 | Cyclical feature mismatch between training and API, hardcoded preprocessing constants |

**Overall: 6.9/10** — A strong portfolio foundation with clear MLOps understanding. The critical gaps are in type safety enforcement, training-serving parity, and CI quality gates. Addressing the CRITICAL and HIGH items in §2 (especially §2.1, §2.2, §2.4, §2.9) would immediately push this to **8.0+/10** territory.

---

## 5. Prioritized Action Plan

> [!TIP]
> Tackle these in order for maximum portfolio impact per hour invested.

### Phase 1: Quick Wins (1–2 hours)
1. ✅ Create `.env.example` (§2.3)
2. ✅ Create `src/py.typed` (§2.15)
3. ✅ Create `src/api/__init__.py` (§2.6)
4. ✅ Fix logger import in `model_trainer.py` (§2.5)
5. ✅ Replace legacy `typing` imports (§2.11)
6. ✅ Remove `black` from dependencies (§2.19)
7. ✅ Remove deprecated `version` from `docker-compose.yml` (§2.16)

### Phase 2: Type Safety & CI (2–4 hours)
8. Add `[tool.pyright]` and `[tool.ruff]` to `pyproject.toml` (§2.2, §2.8)
9. Add generic type parameters to all `dict` fields in entities (§2.1)
10. Add `pytest-cov` + coverage gate in CI (§2.7)
11. Add `pyright` CI step
12. Add `model_config = ConfigDict(extra="forbid")` to all entities (§2.12)

### Phase 3: Architecture Hardening (4–8 hours)
13. Fix cyclical feature training-serving skew with shared utility (§2.9)
14. Move all hardcoded values to config (§2.4)
15. Create typed Pydantic models for YAML config (§2.14)
16. Add API versioning (`/v1/predict`) (§3.4)
17. Add `Makefile` (§3.1)
18. Add `.pre-commit-config.yaml` (§3.2)

### Phase 4: Portfolio Differentiation (8+ hours)
19. Add Great Expectations data validation (§3.3)
20. Add structured JSON logging (§3.5)
21. Add OpenTelemetry tracing (§3.6)
22. Add LLM-as-a-Judge agent evals (§3.7)
23. Add `CONTRIBUTING.md` and Model Card (§3.8, §3.10)
24. Refactor `app.py` into modular pages (§2.21)
