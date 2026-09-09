@echo off
setlocal

rem =====================================================================
rem  ninfer-serve launcher - NInfer-4080S-32G, ET27B DFlash2 K=7 profile
rem
rem  Windows equivalent of the Linux command:
rem    ./build-sm89/apps/ninfer-serve out/et27b_dflash2.ninfer \
rem      --host 127.0.0.1 --port 9527 \
rem      --max-context 245760 --kv-capacity 245760 --kv-dtype int8 \
rem      --spec dflash2 --draft-tokens 7 --lm-head-draft \
rem      --preserve-thinking --max-concurrency 4 --vision \
rem      --slot-save-path /home/raymond/ninfer-sessions \
rem      --session-auto-restore --auto-save-on-stop \
rem      --max-snapshot-disk-gib 100 --checkpoint-interval 16384
rem
rem  Usage:    run-et27b-dflash2.bat [path\to\model.ninfer]
rem  Override: NINFER_SERVER=path\to\ninfer-serve.exe
rem =====================================================================

set "ROOT=%~dp0"

rem ---- locate a native Windows ninfer-serve.exe ------------------------
set "SERVER=%NINFER_SERVER%"
if not defined SERVER if exist "%ROOT%build-sm89-win\apps\ninfer-serve.exe" set "SERVER=%ROOT%build-sm89-win\apps\ninfer-serve.exe"
if not defined SERVER if exist "%ROOT%build-sm89\apps\ninfer-serve.exe" set "SERVER=%ROOT%build-sm89\apps\ninfer-serve.exe"
if not defined SERVER if exist "%ROOT%build-sm89\apps\Release\ninfer-serve.exe" set "SERVER=%ROOT%build-sm89\apps\Release\ninfer-serve.exe"
if not defined SERVER if exist "%ROOT%build-sm89\apps\Debug\ninfer-serve.exe" set "SERVER=%ROOT%build-sm89\apps\Debug\ninfer-serve.exe"
if not defined SERVER if exist "%ROOT%ninfer-serve.exe" set "SERVER=%ROOT%ninfer-serve.exe"
if not exist "%SERVER%" (
  echo ninfer-serve.exe not found at: %SERVER%
  echo A native Windows build is required - build-sm89 holds Linux binaries
  echo and cannot run on Windows. Build one here with CMake
  echo -DCMAKE_CUDA_ARCHITECTURES=89 ^(Visual Studio 2022+, CUDA 12.8+^), or
  echo set NINFER_SERVER to the full path of a Windows ninfer-serve.exe.
  exit /b 1
)

rem ---- model artifact --------------------------------------------------
set "MODEL=%~1"
if not defined MODEL set "MODEL=%ROOT%out\et27b_dflash2.ninfer"
if not exist "%MODEL%" (
  echo Model not found: %MODEL%
  echo Generate out\et27b_dflash2.ninfer with the tools/reference converter
  echo ^(recipe qwen3_8_27b-v2^) or drag a .ninfer file onto this launcher.
  exit /b 1
)

rem ---- on-disk session store ------------------------------------------
set "SESSIONS=%ROOT%ninfer-sessions"
if not exist "%SESSIONS%" mkdir "%SESSIONS%"

echo Starting ninfer-serve - ET27B DFlash2 K=7 at http://127.0.0.1:9527/v1
echo   Server : %SERVER%
echo   Model  : %MODEL%
echo   Sessions: %SESSIONS%
echo Press Ctrl+C to stop - sessions are flushed on clean shutdown.

"%SERVER%" "%MODEL%" ^
  --host 127.0.0.1 --port 9527 ^
  --max-context 245760 --kv-capacity 245760 --kv-dtype int8 ^
  --spec dflash2 --draft-tokens 7 --lm-head-draft ^
  --preserve-thinking --max-concurrency 4 --vision ^
  --slot-save-path "%SESSIONS%" ^
  --session-auto-restore --auto-save-on-stop ^
  --max-snapshot-disk-gib 100 --checkpoint-interval 16384

set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" echo ninfer-serve exited with code %EXITCODE%.
exit /b %EXITCODE%
