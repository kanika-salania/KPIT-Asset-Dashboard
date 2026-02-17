@echo off
setlocal

REM === Force Python 3.13 every time (fixes PATH issues) ===
py -3.13 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.13 is not installed correctly.
    echo Install Python 3.13 from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM === Create a fresh venv if it doesn't exist ===
if not exist ".venv" (
    py -3.13 -m venv .venv
)

REM === ALWAYS use the venv python (ignores system Python completely) ===
set PYEXE=.venv\Scripts\python.exe

"%PYEXE%" -m pip install --upgrade pip
"%PYEXE%" -m pip install -r requirements.txt

"%PYEXE%" -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0

endlocal
