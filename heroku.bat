@echo off
cd /d %~dp0

echo Pushing to Heroku...
git push heroku main

echo Opening app in Heroku...
heroku open

pause