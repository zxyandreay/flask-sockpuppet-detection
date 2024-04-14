@echo off
cd /d %~dp0

echo Adding all changes to staging...
git add .

echo Starting to remove files listed in .gitignore...
if not exist .gitignore (
    echo .gitignore file not found.
    exit /b 1
)

for /f "tokens=* usebackq" %%a in (`.gitignore`) do (
    set "line=%%a"
    
    if not "!line!"=="" if not "!line:~0,1!"=="#" (
        echo Removing %%a from Git...
        git rm -r --cached %%a
        if errorlevel 1 (
            echo Failed to remove %%a from Git.
        ) else (
            echo Successfully removed %%a from Git.
        )
    )
)

echo Committing the changes...
git commit -m "Updated"

echo Pushing changes to the remote repository...
git push origin main

echo Update complete.
