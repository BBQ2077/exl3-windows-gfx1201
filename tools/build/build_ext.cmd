@echo off
rem Compile native\exllamav3_ext.pyd for gfx1201. Run tools\build\setup_env.cmd once first.
rem Arguments go to build_ext.py: --clean (full rebuild), --no-install (build only, leave native\ alone).
setlocal
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "PY=%ROOT%\build\.venv\Scripts\python.exe"
if not exist "%PY%" (echo [ERROR] build\.venv missing. Run tools\build\setup_env.cmd first. & exit /b 1)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSDIR="
if exist "%VSWHERE%" for /f "usebackq delims=" %%V in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%V"
if not defined VSDIR (echo [ERROR] Visual Studio 2022 with "Desktop development with C++" not found. & exit /b 1)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

for /f "usebackq delims=" %%R in (`"%PY%" -m rocm_sdk path --root`) do set "ROCM_DEVEL=%%R"
if not exist "%ROCM_DEVEL%\lib\llvm\bin\clang-cl.exe" (echo [ERROR] ROCm development tree incomplete: %ROCM_DEVEL% & exit /b 1)
set "CXX=%ROCM_DEVEL%\lib\llvm\bin\clang-cl.exe"
set "PYTORCH_ROCM_ARCH=gfx1201"
set "DISTUTILS_USE_SDK=1"
if not defined MAX_JOBS set "MAX_JOBS=%NUMBER_OF_PROCESSORS%"
set "PATH=%ROOT%\build\.venv\Scripts;%PATH%"

"%PY%" "%~dp0build_ext.py" %*
exit /b %ERRORLEVEL%
