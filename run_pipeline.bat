@echo off
echo ==============================================================
echo   Hybrid Graph-ML Fraud Detection System (Elliptic Dataset)
echo ==============================================================
echo.
echo Starting the full end-to-end pipeline...
echo Note: This process may take several hours on CPU due to 
echo Node2Vec embeddings and extensive hyperparameter tuning.
echo.
echo Logs are saved to: results\logs\pipeline.log
echo.

call FDS\Scripts\activate.bat
python Main_FDS.py
pause
