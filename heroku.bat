@echo off
echo Changing directory to the Git repository...
cd /d %~dp0

echo Pushing to Heroku...
git push heroku main

echo Opening app in Heroku...
heroku open

echo Checking logs for details...
heroku logs --tail

pause