# Chess Game Documentation

## Setting Up the Environment

If you get an error like "Import tensorflow.keras could not be resolved", you need to install TensorFlow:

1. Run the provided setup script:
   ```
   powershell -ExecutionPolicy Bypass -File setup_tensorflow.ps1
   ```

2. This will install:
   - TensorFlow
   - Pygame
   - python-chess

3. After installation, you can run the chess game:
   ```
   python chess_game.py
   ```