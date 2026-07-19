@echo off
cd /d D:\Capstone\MechSage
start "MechSage Backend" /min C:\Python314\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
