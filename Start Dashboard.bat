@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please follow the setup steps in README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.port=8502 --server.address=127.0.0.1
