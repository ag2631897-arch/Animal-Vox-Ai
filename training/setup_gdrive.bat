@echo off
echo.
echo ========================================
echo  AnimalVox AI - rclone Google Drive Setup
echo ========================================
echo.
echo This will configure rclone to access your
echo Google Drive (5TB storage).
echo.
echo A browser window will open for Google login.
echo.
pause

:: Check if rclone is available
where rclone >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: rclone not found!
    echo Install it with: winget install Rclone.Rclone
    echo Then restart this terminal and try again.
    pause
    exit /b 1
)

echo.
echo Step 1: Creating Google Drive remote named "gdrive"...
echo.
echo Follow the prompts below:
echo   - Choose "n" for new remote
echo   - Name: gdrive
echo   - Storage: Google Drive (usually option 18)
echo   - Client ID: (leave blank, press Enter)
echo   - Client Secret: (leave blank, press Enter)
echo   - Scope: 1 (Full access)
echo   - Root folder: (leave blank, press Enter)
echo   - Service account: (leave blank, press Enter)
echo   - Advanced config: n
echo   - Auto config: y
echo   - Team drive: n
echo.
rclone config

echo.
echo Step 2: Testing connection...
rclone lsd gdrive: 2>nul
if %errorlevel% equ 0 (
    echo.
    echo SUCCESS! Google Drive is connected.
    echo.
    echo Step 3: Creating AnimalVox folder structure...
    rclone mkdir "gdrive:AnimalVoxDatasets/shards/week1"
    rclone mkdir "gdrive:AnimalVoxDatasets/shards/week2"
    rclone mkdir "gdrive:AnimalVoxDatasets/shards/week3"
    rclone mkdir "gdrive:AnimalVoxDatasets/features"
    rclone mkdir "gdrive:AnimalVoxDatasets/checkpoints"
    echo Done!
    echo.
    echo You can now mount Google Drive as a virtual drive:
    echo   rclone mount gdrive: X: --vfs-cache-mode full --vfs-cache-max-size 15G
    echo.
    echo Or use the shard manager:
    echo   py training/gdrive_shard_manager.py --list
    echo   py training/gdrive_shard_manager.py --pull dogspeak
    echo.
    echo Or run the full pipeline:
    echo   py training/run_training.py --full
) else (
    echo.
    echo WARNING: Could not connect to Google Drive.
    echo Run 'rclone config' manually to troubleshoot.
)

pause
