@echo off
setlocal

echo ================================================
echo Dynamic Traffic AI - Tambaram Setup
echo ================================================

echo.
echo Activating virtual environment...
if not exist venv\Scripts\activate.bat (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

echo.
echo Installing requirements...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Package installation failed.
    pause
    exit /b 1
)

echo.
echo Data preparation must download the Tambaram OSM network.
echo Run this command separately so any OSM connection error is visible:
echo.
echo     python prepare_data.py

echo.
echo After preparation succeeds, run:
echo     python train_models.py

echo     streamlit run app.py

echo.
pause
