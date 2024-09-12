@echo off
setlocal enabledelayedexpansion

:: Remove all __pycache__ directories
for /d /r %%d in (__pycache__) do (
    if exist "%%d" (
        echo Removing directory: %%d
        rmdir /s /q "%%d"
    )
)

:: Collect all Python files
set "PYPATH="
for /r %%f in (*.py) do (
    set "PYPATH=!PYPATH! "%%f""
)

:: Check if there are any Python files to process
if not defined PYPATH (
    echo No Python files found.
    goto :EOF
)

:: Run Black to format the code
python -m black %PYPATH% 

:: Run isort to sort imports
python -m isort %PYPATH%

endlocal
