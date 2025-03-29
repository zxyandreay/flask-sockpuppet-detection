@echo off  
:: Prevents command output from being displayed in the console  

cd /d %~dp0  
:: Changes directory to the location of the batch file (ensures script runs from the correct path)  

call venv\Scripts\activate  
:: Activates the Python virtual environment located in the "venv" folder  

python app.py  
:: Runs the Python application (app.py)  

pause  
:: Keeps the command window open after execution, so you can see any output or errors  
