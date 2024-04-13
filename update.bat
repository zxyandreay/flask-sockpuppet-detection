@echo off
echo Changing directory to the Git repository...
cd /d %~dp0

echo Adding all changes to staging...
git add .

echo Committing the changes...
git commit -m "Updated"

echo Pushing changes to the remote repository...
git push origin main

echo Update complete.
