@echo off
REM run.bat — اجرای ربات روی ویندوز (دابل‌کلیک کن)
REM اول باید یک بار install.bat رو اجرا کرده باشی
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [!] محیط مجازی پیدا نشد. اول install.bat رو اجرا کن.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" bot.py
pause
