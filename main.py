# main.py
"""
This file serves as the entry point for the Stratified Frame Selector application.
It initializes a QApplication (the main event loop for a Qt-based GUI), creates
the main window, and starts the application's event loop.

Steps:
1. Create a QApplication instance, which is necessary for any PySide6/Qt GUI.
2. Instantiate the MainWindow class (defined in ui.py) which sets up the entire UI.
3. Show the main window.
4. Start the event loop by calling app.exec(). This loop continues until the user
   closes the application or another exit condition is met.
5. sys.exit() ensures the program ends cleanly when the event loop finishes.
"""

import sys
from PySide6.QtWidgets import QApplication
from ui import MainWindow  # Import the MainWindow class from ui.py, where the UI is defined

def main():
    """
    The main() function is the entry point of the application.
    It:
    - Creates the QApplication instance
    - Sets up and shows the main window
    - Starts the Qt event loop
    """
    # Create a QApplication instance, which provides the event loop and other GUI essentials.
    app = QApplication(sys.argv)
    
    # Create an instance of the main window. This will set up all UI elements and logic.
    window = MainWindow()
    
    # Show the main window on the screen.
    window.show()
    
    # Execute the event loop. This call blocks until the application is closed.
    # When app.exec() returns, the application is about to exit.
    sys.exit(app.exec())

if __name__ == "__main__":
    # If main.py is run as a script (not imported as a module), call the main() function.
    # This ensures that if main.py is imported elsewhere, it doesn't immediately start the UI.
    main()
