import sys
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QObject, QThread, Qt, pyqtSignal
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSplitter,
        QTableView,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    print(
        f"Missing dependency: {exc.name}\n"
        "Install dependencies with: py -m pip install -r requirements.txt"
    )
    sys.exit(1)

from visualize_backprop_attribution import (
    DEFAULT_CHECKPOINT,
    compute_attribution,
    embed_sentences,
    get_score_columns,
    load_manifest,
    load_model,
    make_embed_args,
    normalize_scores,
    resolve_script_path,
    split_text,
)


DEFAULT_MANIFEST = "bench_data/pseudo_text_sentence_embeddings_multi/manifest.json"


class ProbabilityTableModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self.rows = []

    def set_rows(self, rows):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 2

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        score, probability = self.rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                return str(score)
            return f"{probability:.6f}"
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        return ["Score", "Probability"][section]


class AttributionTableModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self.rows = []

    def set_rows(self, rows):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 5

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row = self.rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            values = [
                row["rank"],
                row["original_index"] + 1,
                f"{row['importance']:.8g}",
                f"{row['signed']:.8g}",
                row["sentence"],
            ]
            return str(values[index.column()])

        if role == Qt.ItemDataRole.BackgroundRole:
            alpha = int(35 + 180 * row["normalized"])
            if row["signed"] >= 0:
                return QColor(219, 70, 38, alpha)
            return QColor(37, 99, 235, alpha)

        if role == Qt.ItemDataRole.ToolTipRole:
            direction = "supports" if row["signed"] >= 0 else "opposes"
            return (
                f"{direction} target score\n"
                f"importance={row['importance']:.8g}\n"
                f"signed={row['signed']:.8g}"
            )

        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() < 4:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        return ["Rank", "Sentence #", "Importance", "Signed", "Sentence"][section]


class AttributionWorker(QObject):
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, options):
        super().__init__()
        self.options = options

    def run(self):
        try:
            self.status.emit("Loading manifest and splitting text...")
            manifest = load_manifest(self.options.manifest)
            embed_args = make_embed_args(self.options, manifest)
            sentences = split_text(self.options.text, embed_args)

            self.status.emit(f"Embedding {len(sentences)} sentence(s)...")
            embeddings = embed_sentences(sentences, embed_args)

            self.status.emit("Loading classifier checkpoint...")
            device = torch.device(
                self.options.device or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            model, checkpoint = load_model(self.options.checkpoint, device)

            score_columns = get_score_columns(checkpoint)
            score_index = self.options.score_index
            score_dim = int(checkpoint["score_dim"])
            score_class_count = int(checkpoint.get("score_class_count", 5))

            self.status.emit("Running prediction...")
            with torch.no_grad():
                preview_inputs = torch.from_numpy(embeddings).float().unsqueeze(0).to(device)
                preview_logits = model(preview_inputs).view(1, score_dim, score_class_count)[
                    0, score_index
                ]
            predicted_class = int(preview_logits.argmax().item())
            class_index = predicted_class if self.options.score_value is None else self.options.score_value - 1

            self.status.emit("Computing backprop attribution...")
            attribution = compute_attribution(
                model,
                embeddings,
                score_index,
                class_index,
                score_dim,
                score_class_count,
                device,
                self.options.method,
            )

            normalized = normalize_scores(attribution["importance"])
            rows = []
            for index, (importance, signed, norm) in enumerate(
                zip(attribution["importance"], attribution["signed"], normalized)
            ):
                rows.append(
                    {
                        "original_index": index,
                        "sentence": sentences[index],
                        "importance": float(importance.item()),
                        "signed": float(signed.item()),
                        "normalized": norm,
                    }
                )
            rows.sort(key=lambda item: item["importance"], reverse=True)
            for rank, row in enumerate(rows, start=1):
                row["rank"] = rank

            probabilities = [
                (index + 1, float(probability.item()))
                for index, probability in enumerate(attribution["probabilities"][score_index])
            ]
            self.finished.emit(
                {
                    "rows": rows,
                    "probabilities": probabilities,
                    "score_column": score_columns[score_index],
                    "target_score": class_index + 1,
                    "predicted_score": predicted_class + 1,
                    "sentence_count": len(sentences),
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class AttributionViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Backprop Attribution Viewer")
        self.resize(1280, 780)

        self.probability_model = ProbabilityTableModel()
        self.attribution_model = AttributionTableModel()
        self.thread = None
        self.worker = None

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text to explain...")

        self.checkpoint_input = QLineEdit(str(resolve_script_path(DEFAULT_CHECKPOINT)))
        self.manifest_input = QLineEdit(str(resolve_script_path(DEFAULT_MANIFEST)))

        self.score_combo = QComboBox()
        self.score_value_combo = QComboBox()
        self.score_value_combo.addItem("Predicted", None)
        for score in range(1, 6):
            self.score_value_combo.addItem(str(score), score)

        self.method_combo = QComboBox()
        self.method_combo.addItem("grad-times-input", "grad-times-input")
        self.method_combo.addItem("grad-norm", "grad-norm")

        self.device_input = QLineEdit()
        self.device_input.setPlaceholderText("auto")
        self.embedding_device_input = QLineEdit()
        self.embedding_device_input.setPlaceholderText("same as model")

        self.run_button = QPushButton("Run Attribution")
        self.run_button.clicked.connect(self.run_attribution)
        self.reload_scores_button = QPushButton("Reload Dimensions")
        self.reload_scores_button.clicked.connect(self.load_score_columns)

        self.status_label = QLabel("Ready")
        self.summary_label = QLabel("")

        self.probability_table = QTableView()
        self.probability_table.setModel(self.probability_model)
        self.probability_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.probability_table.verticalHeader().setVisible(False)

        self.attribution_table = QTableView()
        self.attribution_table.setModel(self.attribution_model)
        self.attribution_table.setWordWrap(True)
        self.attribution_table.setAlternatingRowColors(True)
        self.attribution_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        for column in range(4):
            self.attribution_table.horizontalHeader().setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )

        self._build_ui()
        self.load_score_columns()

    def _build_ui(self):
        checkpoint_button = QPushButton("Browse")
        checkpoint_button.clicked.connect(self.choose_checkpoint)
        manifest_button = QPushButton("Browse")
        manifest_button.clicked.connect(self.choose_manifest)

        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(self.checkpoint_input, 1)
        checkpoint_row.addWidget(checkpoint_button)

        manifest_row = QHBoxLayout()
        manifest_row.addWidget(self.manifest_input, 1)
        manifest_row.addWidget(manifest_button)

        controls = QFormLayout()
        controls.addRow("Checkpoint", checkpoint_row)
        controls.addRow("Manifest", manifest_row)
        controls.addRow("Score Dimension", self.score_combo)
        controls.addRow("Target Score", self.score_value_combo)
        controls.addRow("Method", self.method_combo)
        controls.addRow("Model Device", self.device_input)
        controls.addRow("Embedding Device", self.embedding_device_input)

        button_row = QHBoxLayout()
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.reload_scores_button)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Input Text"))
        left_layout.addWidget(self.text_input, 1)
        left_layout.addLayout(controls)
        left_layout.addLayout(button_row)
        left_layout.addWidget(self.status_label)

        result_top = QWidget()
        result_top_layout = QHBoxLayout(result_top)
        result_top_layout.addWidget(self.summary_label, 1)
        result_top_layout.addWidget(self.probability_table, 0)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(result_top)
        right_layout.addWidget(self.attribution_table, 1)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def choose_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Checkpoint",
            str(resolve_script_path(".")),
            "PyTorch checkpoint (*.pt *.pth);;All files (*.*)",
        )
        if path:
            self.checkpoint_input.setText(path)
            self.load_score_columns()

    def choose_manifest(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Manifest",
            str(resolve_script_path(".")),
            "JSON files (*.json);;All files (*.*)",
        )
        if path:
            self.manifest_input.setText(path)

    def load_score_columns(self):
        try:
            checkpoint = torch.load(
                resolve_script_path(self.checkpoint_input.text()),
                map_location="cpu",
                weights_only=True,
            )
            score_columns = get_score_columns(checkpoint)
        except Exception as exc:
            self.status_label.setText(f"Could not load score dimensions: {exc}")
            return

        self.score_combo.clear()
        for index, column in enumerate(score_columns):
            self.score_combo.addItem(f"{index}: {column}", index)
        self.status_label.setText(f"Loaded {len(score_columns)} score dimension(s).")

    def run_attribution(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Missing Text", "Enter text before running attribution.")
            return
        if self.score_combo.currentData() is None:
            QMessageBox.warning(self, "Missing Dimension", "Load and choose a score dimension first.")
            return

        self.run_button.setEnabled(False)
        self.reload_scores_button.setEnabled(False)
        self.status_label.setText("Starting...")
        self.summary_label.setText("")
        self.probability_model.set_rows([])
        self.attribution_model.set_rows([])

        options = SimpleNamespace(
            text=text,
            checkpoint=self.checkpoint_input.text(),
            manifest=self.manifest_input.text(),
            score_index=int(self.score_combo.currentData()),
            score_value=self.score_value_combo.currentData(),
            method=self.method_combo.currentData(),
            device=self.device_input.text().strip() or None,
            embedding_device=self.embedding_device_input.text().strip() or None,
            embedding_model=None,
            embedding_backend=None,
            embedding_batch_size=16,
            sentence_model_name=None,
            sentence_device=None,
            max_length=None,
            normalize_embeddings=False,
        )

        self.thread = QThread()
        self.worker = AttributionWorker(options)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_attribution_finished)
        self.worker.failed.connect(self.on_attribution_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.on_worker_stopped)
        self.thread.start()

    def on_attribution_finished(self, result):
        self.probability_model.set_rows(result["probabilities"])
        self.attribution_model.set_rows(result["rows"])
        self.attribution_table.resizeRowsToContents()
        self.summary_label.setText(
            "Dimension: {score_column}\n"
            "Target score: {target_score}\n"
            "Predicted score: {predicted_score}\n"
            "Sentences: {sentence_count}".format(**result)
        )
        self.status_label.setText("Done")

    def on_attribution_failed(self, message):
        self.status_label.setText("Failed")
        QMessageBox.critical(self, "Attribution Failed", message)

    def on_worker_stopped(self):
        self.thread = None
        self.worker = None
        self.run_button.setEnabled(True)
        self.reload_scores_button.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    window = AttributionViewerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
