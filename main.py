import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QLockFile
from PySide6.QtWidgets import QApplication, QMessageBox

from core.logger import logger
from core.config import config_manager
from gui.main_window import MainWindow


def main():
    logger.info("Starting Toki-STT Application...")

    # Load Config
    config = config_manager.config
    logger.info(f"Loaded configuration. Model: {config['models']['selected_model']}")

    app = QApplication(sys.argv)

    # Single instance guard: a second launch would register its own global PTT
    # hook and OSC sender, doubling VRChat chat messages.
    lock_file = QLockFile(str(Path(tempfile.gettempdir()) / "Toki-STT.lock"))
    if not lock_file.tryLock(100):
        QMessageBox.information(None, "Toki-STT", "Toki-STT is already running.")
        logger.info("Another instance is already running - exiting.")
        sys.exit(0)

    window = MainWindow()
    window.show()

    logger.info("Application initialized.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()