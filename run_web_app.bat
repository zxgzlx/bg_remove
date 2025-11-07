@echo off
setlocal
cd /d "%~dp0"
python -m uvicorn web_app:app --host 0.0.0.0 --port 8000 --reload
