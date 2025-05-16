# This script installs TensorFlow and other Python packages required by the Chess Neural Network game.
Write-Host "Installing TensorFlow and required packages..." -ForegroundColor Green

# Install specific version of TensorFlow, Pygame for graphics, and python-chess for chess logic.
pip install tensorflow==2.12.0
pip install pygame
pip install python-chess

Write-Host "`nInstallation complete!" -ForegroundColor Green
Write-Host "Now you can run the chess game with: python chess_game.py" -ForegroundColor Yellow

Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
