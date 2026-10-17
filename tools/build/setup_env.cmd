@echo off
rem Create the venv used to compile exllamav3_ext.pyd: build\.venv (run once; downloads about 4 GB).
rem Needs uv (https://docs.astral.sh/uv/) and network access. Versions are pinned to the combination
rem that built the kernels shipped in native\.
setlocal
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "VENV=%ROOT%\build\.venv"
where uv >nul 2>nul || (echo [ERROR] uv not found. Install it first: https://docs.astral.sh/uv/ & exit /b 1)
if not exist "%VENV%\Scripts\python.exe" (
  uv venv -p 3.12 "%VENV%" || exit /b 1
)
uv pip install --python "%VENV%\Scripts\python.exe" ^
  --index-url https://rocm.prereleases.amd.com/whl-multi-arch/ ^
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match ^
  "torch[device-gfx1201]==2.13.0+rocm10.0.0rc4" "rocm[devel]==10.0.0rc4" ^
  ninja setuptools wheel numpy pybind11 || exit /b 1
rem First run unpacks the ROCm development tree (clang, HIP headers, hipBLAS) into site-packages\_rocm_sdk_devel
"%VENV%\Scripts\python.exe" -m rocm_sdk path --root || exit /b 1
echo BUILD_ENV_READY
