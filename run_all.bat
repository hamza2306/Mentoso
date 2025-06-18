@echo off
start "" python -m http.server 5501
start "" python chatbot.py
pause
