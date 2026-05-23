@echo off
title Inflation Rate Predictor — Launcher
cd /d "%~dp0"

echo ============================================
echo  Inflation Rate Predictor — Starting Up
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Make sure Python is on your PATH.
    pause
    exit /b 1
)

:: Check that models exist
if not exist "models\ensemble.joblib" (
    echo [WARNING] Trained models not found. Running training pipeline first...
    echo This may take several minutes.
    echo.
    python -m src.train
    if errorlevel 1 (
        echo [ERROR] Training failed. Check the output above.
        pause
        exit /b 1
    )
    echo.
)

echo [1/3] Starting FastAPI server on http://localhost:8000 ...
start "FastAPI — Inflation API" cmd /k "cd /d "%~dp0" && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

:: Brief pause so API has a head start before dashboard tries to call it
timeout /t 3 /nobreak >nul

echo [2/3] Starting Streamlit dashboard on http://localhost:8501 ...
start "Streamlit — Dashboard" cmd /k "cd /d "%~dp0" && python -m streamlit run dashboard/app.py --server.port 8501"

echo [3/3] Starting MLflow UI on http://localhost:5000 ...
start "MLflow — Experiment Tracker" cmd /k "cd /d "%~dp0" && python -m mlflow ui --backend-store-uri mlruns --port 5000"

echo.
echo ============================================
echo  All services are starting in new windows.
echo.
echo  FastAPI  (REST API + docs) : http://localhost:8000/docs
echo  Dashboard (Streamlit)      : http://localhost:8501
echo  MLflow   (experiment log)  : http://localhost:5000
echo.
echo  Close this window or press any key to exit.
echo  To stop all services, close their windows.
echo ============================================
echo.
pause
