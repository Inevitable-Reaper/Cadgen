@echo off
echo 🔧 Setting up AI CAD Generator Frontend...

REM Check if we're in the frontend directory
if not exist package.json (
    echo ❌ Please run this script from the frontend directory
    pause
    exit /b 1
)

REM Install dependencies
echo 📦 Installing dependencies...
call npm install

REM Check if installation was successful
if errorlevel 1 (
    echo ❌ npm install failed
    pause
    exit /b 1
)

echo ✅ Frontend setup complete!
echo 🚀 You can now run: npm start
pause
