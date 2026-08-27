@echo off
REM install.bat — نصب یک‌باره روی ویندوز (فقط بار اول اجرا کن)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [1/2] ساخت محیط مجازی...
    python -m venv .venv
)
echo [2/2] نصب کتابخانه‌ها...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo.
echo آماده شد! حالا فایل .env رو بساز و توکن ربات رو بذار، بعد run.bat رو اجرا کن.
pause
