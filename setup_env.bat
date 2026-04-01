@echo off
:: ============================================================
:: setup_env.bat
:: Creates the FDS virtual environment and installs all packages
:: Virtual environment name: FDS
:: ============================================================

echo.
echo ============================================================
echo   Hybrid Fraud Detection System -- Environment Setup
echo   Virtual Environment: FDS
echo ============================================================
echo.

:: Find Python 3.11
set PYTHON_CMD=
for %%p in (
    "C:\Python311\python.exe"
    "C:\Program Files\Python311\python.exe"
    "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe"
    "C:\Python312\python.exe"
    "C:\Program Files\Python312\python.exe"
    "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe"
) do (
    if exist %%p (
        set PYTHON_CMD=%%p
        goto :found_python
    )
)

:: Try winget-installed Python (refreshed PATH)
where python >nul 2>&1
if %ERRORLEVEL% == 0 (
    set PYTHON_CMD=python
    goto :found_python
)

echo [ERROR] Python not found. Please install Python 3.11+ first.
echo         Run: winget install --id Python.Python.3.11 --source winget
pause
exit /b 1

:found_python
echo [OK] Found Python: %PYTHON_CMD%
%PYTHON_CMD% --version

:: Create virtual environment
echo.
echo [STEP 1/3] Creating virtual environment 'FDS'...
%PYTHON_CMD% -m venv FDS
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment 'FDS' created.

:: Activate and upgrade pip
echo.
echo [STEP 2/3] Upgrading pip, setuptools, wheel...
call FDS\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
echo [OK] pip upgraded.

:: Install requirements
echo.
echo [STEP 3/3] Installing requirements (this may take 10-20 minutes)...
echo            PyTorch CPU, XGBoost, LightGBM, CatBoost, SHAP, Optuna...

:: Install PyTorch CPU first (special index URL)
pip install torch==2.3.0+cpu torchvision==0.18.0+cpu --index-url https://download.pytorch.org/whl/cpu

:: Install remaining requirements
pip install ^
    numpy==1.26.4 ^
    pandas==2.2.2 ^
    scipy==1.12.0 ^
    pyarrow==16.1.0 ^
    scikit-learn==1.5.0 ^
    xgboost==2.0.3 ^
    lightgbm==4.3.0 ^
    catboost==1.2.5 ^
    imbalanced-learn==0.12.3 ^
    networkx==2.8.8 ^
    node2vec==0.4.6 ^
    python-louvain==0.16 ^
    gensim==4.3.2 ^
    "setuptools<71.0.0" ^
    optuna==3.6.1 ^
    shap==0.45.1 ^
    matplotlib==3.9.0 ^
    seaborn==0.13.2 ^
    plotly==5.22.0 ^
    tqdm==4.66.4 ^
    joblib==1.4.2 ^
    pyyaml==6.0.1 ^
    colorlog==6.8.2 ^
    psutil==5.9.8

if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some packages may have failed. Check output above.
) else (
    echo.
    echo ============================================================
    echo   [SUCCESS] FDS environment ready!
    echo.
    echo   To activate: FDS\Scripts\activate.bat
    echo   To run:      python Main_FDS.py
    echo   To skip N2V: python Main_FDS.py --skip-node2vec
    echo   To evaluate: python Main_FDS.py --only-eval
    echo ============================================================
)
pause
