@echo off
REM Compile l'application en un seul fichier : dist\AudioTranscriber.exe
REM Prerequis : pip install -r requirements.txt pyinstaller
cd /d "%~dp0"

python -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name AudioTranscriber ^
  --collect-all faster_whisper ^
  --collect-all ctranslate2 ^
  --collect-all av ^
  --collect-all onnxruntime ^
  --exclude-module tkinter --exclude-module tkinterdnd2 ^
  --exclude-module torch --exclude-module torchvision --exclude-module torchaudio ^
  --exclude-module matplotlib --exclude-module scipy --exclude-module pandas ^
  app.py

if errorlevel 1 (
  echo.
  echo ECHEC de la compilation.
) else (
  echo.
  echo OK : dist\AudioTranscriber.exe
)
pause
