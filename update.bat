@echo off
cd /d %~dp0

echo Staging changes...
git add .

echo Committing the changes...
git commit -m "Updated"
if errorlevel 1 (
    echo Commit failed, possibly no changes to commit.
) else (
    echo Commit successful.
)

echo Pushing changes to the remote repository...
git push origin main
if errorlevel 1 (
    echo Push failed, check your connection or permissions.
) else (
    echo Push successful.
)

echo Update complete.
pause