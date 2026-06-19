@echo off
cd /d "%~dp0"
echo 서버를 시작합니다...
start "beat-metronome-server" cmd /k venv\Scripts\python.exe server.py
timeout /t 3 /nobreak >nul
start index.html
