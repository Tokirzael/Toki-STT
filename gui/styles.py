
DARK_THEME = """
QMainWindow {
    background-color: #121212;
    color: #E0E0E0;
}

QWidget {
    background-color: #121212;
    color: #E0E0E0;
    font-family: "Segoe UI", sans-serif;
    font-size: 14px;
}

QGroupBox {
    border: 1px solid #330044;
    border-radius: 5px;
    margin-top: 20px;
    font-weight: bold;
    color: #BB86FC;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 3px;
    color: #BB86FC;
}

QLabel {
    color: #E0E0E0;
}

QLineEdit {
    background-color: #1E1E1E;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 5px;
    color: #FFF;
    selection-background-color: #BB86FC;
}

QComboBox {
    background-color: #1E1E1E;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 5px;
    color: #FFF;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 1px;
    border-left-color: darkgray;
    border-left-style: solid;
}

QPushButton {
    background-color: #3700B3;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #6200EE;
}

QPushButton:pressed {
    background-color: #3700B3;
}

QProgressBar {
    border: 1px solid #333;
    border-radius: 5px;
    text-align: center;
    background-color: #1E1E1E;
}

QProgressBar::chunk {
    background-color: #BB86FC;
    width: 20px;
}

QTextEdit {
    background-color: #1E1E1E;
    border: 1px solid #333;
    border-radius: 4px;
    color: #03DAC6; /* Cyan for text preview */
}

QRadioButton {
    color: #E0E0E0;
    spacing: 8px;
}

/* Explicit indicator styling: the default Qt checked dot inherits a dark
   palette color and disappears on this background. Draw it as a filled
   purple disc (checked) / hollow ring (unchecked). */
QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border-radius: 8px;
    border: 2px solid #5A5A5A;
    background-color: transparent;
}

QRadioButton::indicator:hover {
    border-color: #BB86FC;
}

QRadioButton::indicator:checked {
    border-color: #BB86FC;
    background-color: #BB86FC;
}

QStatusBar {
    background-color: #121212;
    color: #BB86FC;
}
"""
