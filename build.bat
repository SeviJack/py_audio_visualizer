@echo off
c:\PersonalProjects\Esp32_equalizer\.venv\Scripts\python.exe -m PyInstaller --noconsole --onefile --icon=C:\icon\eq.ico desktopeq.py
move /Y C:\PersonalProjects\py_audio_visualizer\dist "C:\Tools\desktopeq\desktopeq.exe"
echo Build complete.
