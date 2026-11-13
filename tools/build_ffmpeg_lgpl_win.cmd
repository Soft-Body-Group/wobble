@echo off
rem MSVC environment + MSYS2 bash for build_ffmpeg_lgpl_win.sh.
rem MSYS2 is found at %WOBBLE_MSYS% (default target\tools\msys64, a
rem no-admin extract of msys2-base); it needs make diffutils nasm pkgconf.
setlocal
cd /d "%~dp0.."
if "%WOBBLE_MSYS%"=="" set WOBBLE_MSYS=%CD%\target\tools\msys64
rem vswhere goes on PATH: a quoted "Program Files (x86)" path inside the
rem for's command loses its quotes and splits.
set "PATH=%PATH%;%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
for /f "usebackq delims=" %%i in (`vswhere -latest -property installationPath`) do set "VS=%%i"
if not defined VS (echo no Visual Studio found & exit /b 1)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
rem MSYS2's coreutils `link` would shadow MSVC's linker.
if exist "%WOBBLE_MSYS%\usr\bin\link.exe" ren "%WOBBLE_MSYS%\usr\bin\link.exe" link-coreutils.exe
set MSYS2_PATH_TYPE=inherit
set CHERE_INVOKING=1
"%WOBBLE_MSYS%\usr\bin\bash.exe" -lc "sh tools/build_ffmpeg_lgpl_win.sh %*"
