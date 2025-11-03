@echo off
TITLE Video Analysis Platform - Ultimate Launcher (v9.0 FINAL)

REM --- 使用 %~dp0 确保我们总是在脚本所在的目录中操作 ---
echo Changing directory to the script's location...
cd /d "%~dp0"

REM --- 检查Python是否已安装 ---
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo ====================================================================
    echo ERROR: Python is not found in your system's PATH.
    echo Please install Python 3.8+ and ensure it's added to the PATH.
    echo ====================================================================
    echo.
    pause
    exit /b
)

REM --- [终极改动] 自动检测并创建/激活虚拟环境 ---
IF EXIST ".\.venv\Scripts\activate.bat" (
    echo Found .venv virtual environment. Activating...
) ELSE (
    echo.
    echo ====================================================================
    echo Virtual environment not found. Creating it now...
    echo This may take a moment.
    echo ====================================================================
    echo.
    python -m venv .venv
    IF %ERRORLEVEL% NEQ 0 (
        echo.
        echo ====================================================================
        echo ERROR: Failed to create the virtual environment.
        echo Please check your Python installation.
        echo ====================================================================
        echo.
        pause
        exit /b
    )
    echo Virtual environment created successfully. Activating...
)
call .\.venv\Scripts\activate.bat

REM --- [新增] 自动更新pip到最新版本 ---
echo.
echo ====================================================================
echo  Updating pip to the latest version...
echo ====================================================================
python.exe -m pip install --upgrade pip

REM --- [终极改动] 自动读取 requirements.txt 并安装所有依赖 ---
echo.
echo ====================================================================
echo  Checking and installing all required dependencies from requirements.txt...
echo  This might take some time on the first run.
echo ====================================================================
echo.
pip install -r requirements.txt
echo.
echo ====================================================================
echo  Dependency check complete.
echo ====================================================================
echo.


REM --- 启动Gradio Web应用 ---
echo.
echo Starting the Gradio Web Platform...
echo The Python script will now automatically open your browser to http://127.0.0.1:8001 once the server is ready.
echo.

python app.py

echo.
echo The application has been closed.
pause
