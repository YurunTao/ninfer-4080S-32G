@echo off
setlocal

rem =====================================================================
rem  ninfer-serve launcher - NInfer-4080S-32G, dau_jailbroken artifact
rem  240K context (245760), 4-bit KV cache (rk4v4-e8), 2 concurrent requests
rem
rem  Dedicated launcher for out\dau_jailbroken.ninfer (Qwen3.8-27B
rem  groupwise-int artifact carrying its own MTP head). Speculative decoding
rem  therefore uses MTP3 - no DFlash2 companion weights are required - and
rem  --session-auto-restore can rewind mid-history.
rem
rem  Equivalent Linux command:
rem    ./build-sm89/apps/ninfer-serve out/dau_jailbroken.ninfer \
rem      --host 127.0.0.1 --port 9527 \
rem      --max-context 245760 --kv-capacity 245760 --kv-dtype rk4v4-e8 \
rem      --spec mtp --draft-tokens 3 --lm-head-draft \
rem      --preserve-thinking --max-concurrency 2 --vision \
rem      --vision-max-tokens 32768 \
rem      --slot-save-path /home/raymond/ninfer-sessions \
rem      --session-auto-restore --auto-save-on-stop \
rem      --max-snapshot-disk-gib 100 --checkpoint-interval 16384
rem
rem  Usage:    run-dau-jailbroken.bat [path\to\model.ninfer]
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
if not defined MODEL set "MODEL=%ROOT%out\dau_jailbroken.ninfer"
if not exist "%MODEL%" (
  echo Model not found: %MODEL%
  echo Generate out\dau_jailbroken.ninfer with the tools/reference converter, or
  echo drag a .ninfer file onto this launcher.
  exit /b 1
)

rem ---- on-disk session store ------------------------------------------
set "SESSIONS=%ROOT%ninfer-sessions"
if not exist "%SESSIONS%" mkdir "%SESSIONS%"

echo Starting ninfer-serve - dau_jailbroken (Qwen3.8-27B MTP3), 240K ctx, rk4v4-e8 KV, 2 concurrent at http://127.0.0.1:9527/v1
echo   Server : %SERVER%
echo   Model  : %MODEL%
echo   Sessions: %SESSIONS%
echo Press Ctrl+C to stop - sessions are flushed on clean shutdown.

"%SERVER%" "%MODEL%" ^
  --host 127.0.0.1 --port 9527 ^
  --max-context 245760 --kv-capacity 245760 --kv-dtype rk4v4-e8 ^
  --spec mtp --draft-tokens 3 --lm-head-draft ^
  --preserve-thinking --max-concurrency 2 --vision ^
  --vision-max-tokens 32768 ^
  --slot-save-path "%SESSIONS%" ^
  --session-auto-restore --auto-save-on-stop ^
  --max-snapshot-disk-gib 100 --checkpoint-interval 16384

set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" echo ninfer-serve exited with code %EXITCODE%.
exit /b %EXITCODE%
