# Sentiment Analysis (Hindi)

This repository contains a small Flask-based Hindi sentiment analysis app.

Contents
- `sentiment.py` — trains a TF-IDF + LogisticRegression model (if model files are missing) and serves a small frontend + API (`/predict`).
- `Hindi reviews.xlsx` — labeled reviews used to train the model (must be present in the project root).

Quick overview
- The app will train a model automatically if `sentiment_model.joblib` and `tfidf_vectorizer.joblib` are absent and `Hindi reviews.xlsx` is available.
- The Flask frontend is served at `http://127.0.0.1:5000` by default.

Prerequisites
- macOS / Linux / Windows with Python 3.8+ installed.
- Recommended: use the included virtual environment at `.venv` or create a new one.
- VS Code with the Python extension.

VS Code — step-by-step (recommended)
1. Open the project folder in VS Code: `File → Open Folder…` and select this repository root.
2. Select the Python interpreter (use the project's venv if present):
   - Press `Ctrl+Shift+P` (Cmd+Shift+P on macOS) → `Python: Select Interpreter` → choose `.venv/bin/python` (Linux/macOS) or `.venv\Scripts\python.exe` (Windows).
3. (Optional) If a venv is not present, create one in the project root:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   # .venv\Scripts\activate   # Windows (PowerShell)
   ```
4. Install dependencies:
   ```bash
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   # If you need scientific packages: pip install joblib pandas scikit-learn openpyxl flask-cors
   ```
5. Ensure the dataset file is present in project root and is named exactly `Hindi reviews.xlsx`.
6. Run the app from VS Code terminal:
   ```bash
   source .venv/bin/activate
   python sentiment.py
   ```
   - If model files are missing, the script will train and create `sentiment_model.joblib` and `tfidf_vectorizer.joblib`.
7. Open a browser at `http://127.0.0.1:5000` to use the frontend.

API (quick)
- POST `/predict` JSON body: `{"review":"आपका हिंदी/English review यहाँ"}`
- Response: `{"sentiment":"Positive|Negative|...","confidence":0.93}`

Files created at runtime
- `sentiment_model.joblib` — trained scikit-learn model
- `tfidf_vectorizer.joblib` — saved TF-IDF vectorizer
- `sentiment_run.log` — (if you run with redirection) runtime log

Minimal `.gitignore` suggestion
```
.venv/
__pycache__/
*.pyc
*.joblib
.DS_Store
Hindi reviews.xlsx    # OPTIONAL: keep local dataset out of git if private
```

Troubleshooting
- ModuleNotFoundError: ensure venv is activated and `pip install -r requirements.txt` ran.
- Missing Excel file: the script looks for `Hindi reviews.xlsx` in project root. Add it and restart.
- Permission denied (publickey) when pushing: add your SSH key to GitHub or use HTTPS remote (see next section).

Pushing to GitHub (brief)
- If you want to push a small number of files to a remote repo using HTTPS:
  ```bash
  git remote set-url origin https://github.com/<your-username>/Sentiment_Analysis_HindiVersion.git
  git add sentiment.py "Hindi reviews.xlsx"
  git commit -m "Add sentiment app and dataset"
  git push -u origin main
  ```
- To use SSH, add your `~/.ssh/id_ed25519.pub` to GitHub (Settings → SSH keys).

Security & privacy notes
- If the dataset contains private or PII, do not push it to a public repo. Add it to `.gitignore` and keep a local copy.
- Do not run Flask dev server in production. Use gunicorn / uWSGI and proper HTTPS.

Next steps (suggested)
- Add a small `README.md` (this file) — you’re reading it.
- Add `.gitignore` to avoid committing large or secret files.
- Add tests that validate prediction input/output.

If you want, I can:
- add the `.gitignore` and commit it for you,
- add a basic GitHub Actions workflow (lint/test), or
- add an example curl command to test `/predict` — tell me which and I will update the repo.

---
Updated: 2025-10-29
# Hindi Sentiment Analysis

This repository contains a small Flask app for Hindi sentiment analysis.

Files included:

- `sentiment.py` - trains a TF-IDF + LogisticRegression model (if needed) and serves a small frontend + API (`/predict`).
- `Hindi reviews.xlsx` - labeled sample reviews used to train the model.
- `create_sample_excel.py` - helper to generate a small sample Excel dataset (for testing).

Run locally (using the project's venv):

```bash
source .venv/bin/activate
python sentiment.py
```

Then open http://127.0.0.1:5000 in your browser.

Note: For production use, run behind a WSGI server (gunicorn) and secure the app.
