@echo off
setlocal
REM Используем Python 3.13 через uv (скачивается автоматически, без установки в систему)

set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not exist "%UV%" set "UV=uv"

where %UV% >nul 2>&1
if errorlevel 1 (
    echo uv не найден. Устанавливаю uv...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    set "UV=%USERPROFILE%\.local\bin\uv.exe"
)

echo Устанавливаю Python 3.13 через uv...
"%UV%" python install 3.13

echo Создаю виртуальное окружение с Python 3.13...
"%UV%" venv .venv --python 3.13

echo Устанавливаю зависимости...
"%UV%" pip install -r req.txt --python .venv\Scripts\python.exe

echo Готово. Запуск: .venv\Scripts\python.exe script.py
pause
