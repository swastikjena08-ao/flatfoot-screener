@echo off
rem Build "dist\Remi AI\Remi AI.exe" (a folder you can zip and copy to another Windows PC).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe python -m venv .venv
.venv\Scripts\python.exe -m pip install --quiet -r requirements-app.txt pyinstaller
.venv\Scripts\pyinstaller.exe --noconfirm --windowed --name "Remi AI" ^
    --icon "remi_ai\web\img\remi.ico" --add-data "remi_ai\web;remi_ai\web" --add-data "ai_models;ai_models" ^
    --hidden-import cv2 --collect-binaries onnxruntime --collect-all webview ^
    remi_ai\__main__.py
echo.
echo Done: dist\Remi AI\Remi AI.exe
pause
