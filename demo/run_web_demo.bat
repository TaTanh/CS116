@echo off
echo ======================================================================
echo   PRODUCT RECOMMENDATION SYSTEM - WEB DEMO
echo ======================================================================
echo.
echo Starting web server...
echo Please wait while loading model and data...
echo.
echo Once ready, open your browser and go to:
echo.
echo     http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ======================================================================
echo.

cd /d "%~dp0"
E:\Nam_3_HK1\PythonMayHoc\neSemi\.venv\Scripts\python.exe demo_web.py
