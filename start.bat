@echo off
cd /d "%~dp0"

rem === LLM config (local Ollama) ===
set LLM_BASE_URL=http://localhost:11434/v1
set LLM_API_KEY=ollama
set LLM_MODEL=qwen3.5:0.8b

rem === use project venv python (avoid hijacked system python) ===
set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [ERROR] venv not found: %PY%
    echo Run: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

rem === launch terminal Q&A directly ===
"%PY%" -m rag_qa.cli

pause