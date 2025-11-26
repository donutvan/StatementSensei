Budget Sensei is a Streamlit application for importing StatementSensei CSV exports **or PDFs**, normalizing them into a unified schema, auto-categorizing transactions with rules and optional ML, and storing everything in DuckDB for dashboards and labeling workflows.

## Prerequisites
- Python 3.10+
- Recommended: a virtual environment (`python -m venv .venv` and `source .venv/bin/activate` on macOS/Linux, `.venv\\Scripts\\activate` on Windows)
- System build tools for scientific packages (on Debian/Ubuntu: `apt-get update && apt-get install -y build-essential python3-dev`)

## Installation
1. Navigate to the project root:
   ```bash
   cd budget_sensei
   ```
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Running the app
Start Streamlit from the `budget_sensei` directory:
```bash
streamlit run app.py
```
This launches the multi-page app with Dashboard, Transactions, Categorisation, and Settings sections. Upload CSV or PDF statements; encrypted PDFs can be unlocked with the optional password field on the home page.

- Transactions page: filter/search records, override categories, and reapply the rules/ML model across stored data (overrides stay intact).
- Categorisation page: label uncategorised rows, train the TF-IDF + Logistic Regression model, and reuse labeled data for future uploads.
- Settings page: review rules, add new categories, inspect overrides, and confirm ML artifact presence.

## Data and configuration
- Runtime data is stored under `data/` (DuckDB database, labeled samples, and user overrides). Missing files are created automatically on first run.
- Default categories and rule-based mappings live in `config/categories.yaml` and `config/rules.yaml`. Update these to customize the rule engine.
- ML artifacts (`vectorizer.pkl`, `model.pkl`) are written to `data/` after training from the Categorisation page or by running `python -m budget_sensei.ml.training`.

## Modular layout
- `parsers/` handles CSV normalization and PDF parsing.
- `rules/` holds rule and category loaders and rule-based assignment.
- `ml/` contains training and inference utilities for the TF-IDF + Logistic Regression model.
- `services/` orchestrates categorization using overrides, rules, and ML predictions.
- `storage/` manages DuckDB persistence, labeled samples, and user overrides.

## Development tips
- To validate imports quickly:
  ```bash
  python -m compileall .
  ```
- If you edit dependencies, re-run `pip install -r requirements.txt`.
- For a clean data slate, delete `data/transactions.db`, `data/labeled_samples.csv`, and `data/user_overrides.csv` before restarting Streamlit.
