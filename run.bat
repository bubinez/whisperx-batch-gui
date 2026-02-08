@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Сначала выполните install.bat
    pause
    exit /b 1
)
REM Кэш моделей Hugging Face внутри venv (можно удалить вместе с venv)
set HF_HUB_CACHE=%~dp0.venv\cache\huggingface\hub
.venv\Scripts\python.exe script.py
pause