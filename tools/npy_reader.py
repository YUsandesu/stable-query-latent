import csv
import sys
from pathlib import Path

try:
    import numpy as np
    from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import (
        QApplication,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QSplitter,
        QTableView,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    missing = exc.name
    print(
        f"Missing dependency: {missing}\n"
        "Install dependencies with: py -m pip install -r requirements.txt"
    )
    sys.exit(1)


MAX_PREVIEW_ROWS = 500
MAX_PREVIEW_COLUMNS = 200


class NpyTableModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self.array = None
        self.view = np.empty((0, 0))
        self.row_offset = 0
        self.col_offset = 0

    def set_array(self, array, row_start=0, row_count=100, col_start=0, col_count=30):
        self.beginResetModel()
        self.array = array
        self.row_offset = row_start
        self.col_offset = col_start
        matrix = self._as_matrix(array)
        row_end = min(row_start + row_count, matrix.shape[0])
        col_end = min(col_start + col_count, matrix.shape[1])
        self.view = matrix[row_start:row_end, col_start:col_end]
        self.endResetModel()

    def _as_matrix(self, array):
        if array is None:
            return np.empty((0, 0))
        if array.ndim == 0:
            return array.reshape(1, 1)
        if array.ndim == 1:
            return array.reshape(array.shape[0], 1)
        if array.ndim == 2:
            return array
        return array.reshape(array.shape[0], -1)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.view.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.view.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.ToolTipRole,
        ):
            return None

        value = self.view[index.row(), index.column()]
        if isinstance(value, np.generic):
            value = value.item()
        return str(value)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self.col_offset + section)
        return str(self.row_offset + section)


class NpyReaderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NPY Reader")
        self.resize(1100, 720)

        self.file_path = None
        self.array = None
        self.model = NpyTableModel()

        self.path_input = QLineEdit()
        self.path_input.setReadOnly(True)

        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self.open_file_dialog)

        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload_file)
        self.reload_button.setEnabled(False)

        self.export_button = QPushButton("Export View CSV")
        self.export_button.clicked.connect(self.export_current_view)
        self.export_button.setEnabled(False)

        self.row_start = self._spinbox()
        self.row_count = self._spinbox(100, 1, MAX_PREVIEW_ROWS)
        self.col_start = self._spinbox()
        self.col_count = self._spinbox(30, 1, MAX_PREVIEW_COLUMNS)

        self.apply_button = QPushButton("Apply Slice")
        self.apply_button.clicked.connect(self.apply_slice)
        self.apply_button.setEnabled(False)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)

        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)

        self._build_ui()
        self._build_menu()

    def _spinbox(self, value=0, minimum=0, maximum=1_000_000_000):
        box = QSpinBox()
        box.setRange(minimum, maximum)
        box.setValue(value)
        return box

    def _build_ui(self):
        file_bar = QHBoxLayout()
        file_bar.addWidget(QLabel("File"))
        file_bar.addWidget(self.path_input, 1)
        file_bar.addWidget(self.open_button)
        file_bar.addWidget(self.reload_button)
        file_bar.addWidget(self.export_button)

        controls = QFormLayout()
        controls.addRow("Row Start", self.row_start)
        controls.addRow("Row Count", self.row_count)
        controls.addRow("Column Start", self.col_start)
        controls.addRow("Column Count", self.col_count)
        controls.addRow(self.apply_button)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.addLayout(controls)
        side_layout.addWidget(QLabel("Array Info"))
        side_layout.addWidget(self.info_text, 1)

        splitter = QSplitter()
        splitter.addWidget(side)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addLayout(file_bar)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def _build_menu(self):
        open_action = QAction("Open NPY", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file_dialog)

        export_action = QAction("Export Current View", self)
        export_action.setShortcut("Ctrl+S")
        export_action.triggered.connect(self.export_current_view)

        menu = self.menuBar().addMenu("File")
        menu.addAction(open_action)
        menu.addAction(export_action)

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open NPY File",
            str(Path.cwd()),
            "NumPy files (*.npy *.npz);;All files (*.*)",
        )
        if path:
            self.load_file(Path(path))

    def reload_file(self):
        if self.file_path:
            self.load_file(self.file_path)

    def load_file(self, path):
        try:
            loaded = np.load(path, mmap_mode="r", allow_pickle=False)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                names = loaded.files
                if not names:
                    raise ValueError("NPZ file does not contain arrays.")
                self.array = loaded[names[0]]
                npz_note = f"\nNPZ keys: {', '.join(names)}\nShowing key: {names[0]}"
            else:
                self.array = loaded
                npz_note = ""
        except Exception as exc:
            QMessageBox.critical(self, "Open Failed", str(exc))
            return

        self.file_path = path
        self.path_input.setText(str(path))
        self._sync_controls()
        self.apply_slice()
        self._update_info(npz_note)
        self.reload_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.apply_button.setEnabled(True)

    def _sync_controls(self):
        matrix = self.model._as_matrix(self.array)
        max_row = max(matrix.shape[0] - 1, 0)
        max_col = max(matrix.shape[1] - 1, 0)
        self.row_start.setRange(0, max_row)
        self.col_start.setRange(0, max_col)
        self.row_count.setRange(1, min(MAX_PREVIEW_ROWS, max(matrix.shape[0], 1)))
        self.col_count.setRange(1, min(MAX_PREVIEW_COLUMNS, max(matrix.shape[1], 1)))
        self.row_count.setValue(min(100, max(matrix.shape[0], 1)))
        self.col_count.setValue(min(30, max(matrix.shape[1], 1)))

    def apply_slice(self):
        if self.array is None:
            return
        self.model.set_array(
            self.array,
            self.row_start.value(),
            self.row_count.value(),
            self.col_start.value(),
            self.col_count.value(),
        )
        self.table.resizeColumnsToContents()

    def _update_info(self, note=""):
        matrix = self.model._as_matrix(self.array)
        file_size = self.file_path.stat().st_size if self.file_path else 0
        details = [
            f"Path: {self.file_path}",
            f"File size: {file_size:,} bytes",
            f"Shape: {self.array.shape}",
            f"Display shape: {matrix.shape}",
            f"Dtype: {self.array.dtype}",
            f"Dimensions: {self.array.ndim}",
            f"Memory mapped: {isinstance(self.array, np.memmap)}",
        ]
        if self.array.ndim > 2:
            details.append("Display mode: first axis by flattened trailing axes")
        if note:
            details.append(note.strip())
        self.info_text.setPlainText("\n".join(details))

    def export_current_view(self):
        if self.array is None or self.model.view.size == 0:
            return

        default_name = "npy_view.csv"
        if self.file_path:
            default_name = f"{self.file_path.stem}_view.csv"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Current View",
            str(Path.cwd() / default_name),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerows(self.model.view.tolist())
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return

        QMessageBox.information(self, "Export Complete", f"Saved to {path}")


def main():
    app = QApplication(sys.argv)
    window = NpyReaderWindow()
    if len(sys.argv) > 1:
        window.load_file(Path(sys.argv[1]))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
