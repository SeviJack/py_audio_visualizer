@echo off
c:\PersonalProjects\Esp32_equalizer\.venv\Scripts\python.exe -m PyInstaller --noconsole --onefile --icon=C:\icon\eq.ico desktopeq.py
move /Y C:\PersonalProjects\Esp32_equalizer\dist\desktopeq.exe "C:\Tools\desktopeq\desktopeq.exe"
echo Build complete.
