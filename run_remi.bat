@echo off
rem Start Remi AI from the source folder. First run sets up a private Python environment.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Setting up Remi AI for the first time...
    python -m venv .venv || goto :nopython
)
.venv\Scripts\python.exe -m pip install --quiet --disable-pip-version-check -r requirements-app.txt || (pause & goto :eof)
.venv\Scripts\python.exe -m pip install --quiet --disable-pip-version-check -r requirements-voice.txt || echo Voice could not be installed; Remi will work without it.
.venv\Scripts\python.exe -m remi_ai %*
goto :eof
:nopython
echo Python 3.10 or newer is needed: https://www.python.org/downloads/
pause
