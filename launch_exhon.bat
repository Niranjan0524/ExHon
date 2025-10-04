@echo off
title ExHon Proctored Exam Platform
color 0A

echo.
echo    ███████╗██╗  ██╗██╗  ██╗ ██████╗ ███╗   ██╗
echo    ██╔════╝╚██╗██╔╝██║  ██║██╔═══██╗████╗  ██║
echo    █████╗   ╚███╔╝ ███████║██║   ██║██╔██╗ ██║
echo    ██╔══╝   ██╔██╗ ██╔══██║██║   ██║██║╚██╗██║
echo    ███████╗██╔╝ ██╗██║  ██║╚██████╔╝██║ ╚████║
echo    ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
echo.
echo    Proctored Exam Platform - Easy Launcher
echo    =======================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [INFO] Python found - checking version...
for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [INFO] Python version: %PYTHON_VERSION%

REM Check if required files exist
if not exist "app.py" (
    echo [ERROR] app.py not found. Make sure you're in the ExHon directory.
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found.
    pause
    exit /b 1
)

echo [INFO] Checking dependencies...

REM Install/check dependencies
python -c "import flask, flask_socketio, psutil, requests" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing missing dependencies...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        echo Try running: pip install -r requirements.txt
        pause
        exit /b 1
    )
) else (
    echo [INFO] All dependencies are installed
)

REM Create scripts directory if it doesn't exist
if not exist "scripts" mkdir scripts

REM Create Chrome debugging script if it doesn't exist
if not exist "scripts\start_chrome_debug.bat" (
    echo [INFO] Creating Chrome debugging script...
    echo @echo off > "scripts\start_chrome_debug.bat"
    echo echo Starting Chrome with debugging enabled... >> "scripts\start_chrome_debug.bat"
    echo echo Chrome will open with monitoring capabilities. >> "scripts\start_chrome_debug.bat"
    echo echo You can use this browser normally for your exam. >> "scripts\start_chrome_debug.bat"
    echo. >> "scripts\start_chrome_debug.bat"
    echo REM Try common Chrome installation paths >> "scripts\start_chrome_debug.bat"
    echo if exist "%%USERPROFILE%%\AppData\Local\Google\Chrome\Application\chrome.exe" ^( >> "scripts\start_chrome_debug.bat"
    echo     "%%USERPROFILE%%\AppData\Local\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%%TEMP%%\chrome_debug" --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows >> "scripts\start_chrome_debug.bat"
    echo ^) else if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" ^( >> "scripts\start_chrome_debug.bat"
    echo     "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%%TEMP%%\chrome_debug" --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows >> "scripts\start_chrome_debug.bat"
    echo ^) else if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" ^( >> "scripts\start_chrome_debug.bat"
    echo     "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%%TEMP%%\chrome_debug" --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows >> "scripts\start_chrome_debug.bat"
    echo ^) else ^( >> "scripts\start_chrome_debug.bat"
    echo     echo [ERROR] Chrome not found. Please install Google Chrome. >> "scripts\start_chrome_debug.bat"
    echo     pause >> "scripts\start_chrome_debug.bat"
    echo ^) >> "scripts\start_chrome_debug.bat"
)

REM Test system functionality
echo [INFO] Running system test...
python test_system.py >nul 2>&1
if errorlevel 1 (
    echo [WARNING] System test reported issues. Continuing anyway...
) else (
    echo [INFO] System test passed
)

echo.
echo [INFO] Starting Chrome with debugging support...
echo [INFO] A new Chrome window will open - use this for your exam
echo.

REM Start Chrome in background
start /b "Chrome Debug" "scripts\start_chrome_debug.bat"

REM Wait a moment for Chrome to start
timeout /t 3 /nobreak >nul

echo [INFO] Starting ExHon monitoring server...
echo [INFO] Dashboard will be available at: http://localhost:5000
echo.
echo =======================================
echo  IMPORTANT INSTRUCTIONS:
echo =======================================
echo  1. Use the Chrome window that just opened for your exam
echo  2. The dashboard will open automatically in a few seconds
echo  3. Click "Close All" to terminate AI apps and other tabs
echo  4. Your exam tab will be protected from closure
echo  5. Press Ctrl+C to stop the server when exam is complete
echo.
echo [INFO] Server starting in 2 seconds...
timeout /t 2 /nobreak >nul

REM Start the Flask application
echo [INFO] Launching ExHon dashboard...

REM Open the dashboard in the debug-enabled Chrome
timeout /t 3 /nobreak >nul
start "" "http://localhost:5000"

REM Start the Python application
python start.py

echo.
echo [INFO] ExHon session ended.
echo Thank you for using ExHon Proctored Exam Platform!
pause
