@echo off
REM === Create a local virtual environment if it doesn't exist ===
if not exist .venv (
  py -m venv .venv
)

REM === Activate venv ===
call .venv\Scripts\activate

REM === Install/Update dependencies ===
pip install --upgrade pip
pip install -r requirements.txt

REM === Run Streamlit app ===
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
``
