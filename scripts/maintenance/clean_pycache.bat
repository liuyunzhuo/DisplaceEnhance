@echo off
setlocal

for %%i in ("%~dp0..\..") do set "ROOT_DIR=%%~fi"
echo Cleaning Python cache under %ROOT_DIR%

for /d /r "%ROOT_DIR%" %%d in (__pycache__) do (
    if exist "%%d" (
        echo Removing %%d
        rmdir /s /q "%%d"
    )
)

for /r "%ROOT_DIR%" %%f in (*.pyc) do (
    if exist "%%f" (
        echo Removing %%f
        del /f /q "%%f"
    )
)

echo Done.
endlocal
