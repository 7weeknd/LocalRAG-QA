@echo off
cd /d "%~dp0"

rem === LLM config (local Ollama) ===
set LLM_BASE_URL=http://localhost:11434/v1
set LLM_API_KEY=ollama
set LLM_MODEL=qwen3.5:0.8b
set RAG_OPEN_BROWSER=1

rem === full-stack notes ===
rem MySQL connection (host/port/user/password/db) is read from .env in project root.
rem If MySQL is not running, login/history features are disabled but Q&A still works.
rem Easiest way to start MySQL: double-click D:\mysql\start_mysql.bat

rem === use project venv python (avoid hijacked system python) ===
set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [ERROR] venv not found: %PY%
    echo Run: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

rem === start web server, browser opens automatically. Keep this window open. ===
rem port 8010: 8000/8001 are occupied by legacy processes on this machine
"%PY%" -m uvicorn rag_qa.app:app --host 0.0.0.0 --port 8010

pause