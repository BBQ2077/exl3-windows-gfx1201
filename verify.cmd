@echo off
setlocal
cd /d "%~dp0"
"%~dp0python\python.exe" -I -X utf8 -u "%~dp0tests\verify_manifest.py"
if errorlevel 1 exit /b %errorlevel%
"%~dp0python\python.exe" -I -X utf8 -u "%~dp0runtime.py" verify %*
exit /b %errorlevel%
