@echo off
setlocal
cd /d "%~dp0"
"%~dp0python\python.exe" -I -X utf8 -u "%~dp0runtime.py" script server.py %*
exit /b %errorlevel%
