@echo off
setlocal

echo Installing PyInstaller...
python -m pip install pyinstaller
if errorlevel 1 (
  echo Failed to install PyInstaller.
  exit /b 1
)

echo Building executable...
python -m PyInstaller --noconfirm --onefile --windowed --name MaJoT-GUI app_gui.py
if errorlevel 1 (
  echo Build failed.
  exit /b 1
)

echo.
echo Done. Executable is in dist\MaJoT-GUI.exe
endlocal
