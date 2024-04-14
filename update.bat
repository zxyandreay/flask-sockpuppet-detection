@echo off
cd /d %~dp0

echo Checking for .gitignore...
if not exist .gitignore (
    echo .gitignore file not found.
    exit /b 1
)

echo Starting to remove files listed in .gitignore from Git tracking...
for /f "usebackq delims=" %%a in (`.gitignore`) do (
    set "line=%%a"
    
    rem Strip out leading spaces for more accurate comparison
    set "trimmed=!line: =!"

    if not "!trimmed!"=="" (
        if not "!trimmed:~0,1!"=="#" (
            echo Removing %%a from Git tracking...
            git rm -r --cached "%%a"
            if errorlevel 1 (
                echo Failed to remove %%a from Git tracking.
            ) else (
                echo Successfully removed %%a from Git tracking.
            )
        )
    )
)

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