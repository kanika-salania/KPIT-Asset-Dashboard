#!/usr/bin/env bash
set -euo pipefail

# cd to the script directory so relative paths (like images/) work
cd "$(dirname "$0")"

# Prefer python3 if available; fallback to python
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "[ERROR] Python 3.12/3.13 not found. Install from https://www.python.org/downloads/ and try again."
  exit 1
fi

# Create venv if it doesn't exist (local to the project)
if [ ! -d ".venv" ]; then
  "$PYTHON" -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Upgrade pip + install deps
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run Streamlit from THIS venv (avoids global PATH issues)
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
