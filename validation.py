"""PyQt6 validation UI for VICReg review tag prediction.

Pipeline (aligned with the current train_tag_probe.py method):

    input text -> local Qwen embedding -> frozen VICReg encoder -> (num_latents,
    output_dim) code -> pool (flatten/stats) -> normalize -> per-tag linear
    logistic probe -> per-tag probabilities sorted high to low.

The probe is the portable linear artifact produced by:

    train_tag_probe.py --export-head VICReg_review/heads/tag_probe_linear.pt

(normalizer + per-tag logistic weights; same method as the cross-validation, fit on
all games). The artifact stores which encoder checkpoint it was fit on, so this UI
loads that exact encoder by default to keep features consistent.

Run from the repository root:

    C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe validation.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON_EXE = Path("C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe")
GAME_REVIEW_DATA = ROOT / "game_review_data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GAME_REVIEW_DATA) not in sys.path:
    sys.path.insert(0, str(GAME_REVIEW_DATA))

from PyQt6 import QtCore, QtGui, QtWidgets

from game_review_data.embedding_data import DEFAULT_LOCAL_MODEL, LocalEmbedder  # noqa: E402
from VICReg_review.train_tag_probe import load_frozen_encoder, pool_features  # noqa: E402
try:
    from VICReg_review.coarse_tags import coarsen_tag_dict, keyword_scores
except ImportError:  # pragma: no cover
    from coarse_tags import coarsen_tag_dict, keyword_scores


DEFAULT_HEADS_DIR = ROOT / "VICReg_review" / "heads"
DEFAULT_GUI_RUN_DIR = DEFAULT_HEADS_DIR / "gui_run"
DEFAULT_TAGS_DIR = ROOT / "VICReg_review" / "tags"
DEFAULT_GAMES_JSON = ROOT / "game_review_data" / "Steam Games Metadata and Player Reviews (2020–2024" / "games.json"


def newest_existing(patterns: list[str]) -> Path | None:
    paths = []
    for pattern in patterns:
        paths.extend(ROOT.glob(pattern))
    paths = [path for path in paths if path.is_file()]
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def resolve_optional_path(value: str | None, patterns: list[str], label: str) -> Path:
    if value:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path
    path = newest_existing(patterns)
    if path is None:
        joined = ", ".join(patterns)
        raise FileNotFoundError(f"No {label} found. Looked for: {joined}")
    return path


def split_text(text: str, max_sentences: int) -> list[str]:
    parts = re.split(r"(?:\r?\n)+|(?<=[.!?。！？；;])\s*", text.strip())
    sentences = [part.strip() for part in parts if part.strip()]
    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences[:max_sentences]


def load_tags_fallback(tags_dir: Path) -> list[str]:
    vocab_path = tags_dir / "tag_vocab.json"
    if not vocab_path.exists():
        return []
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    return list(payload.get("tags") or [])


def game_tag_dict(record: dict) -> dict[str, float]:
    tags = record.get("tags") or {}
    if isinstance(tags, dict):
        return {str(name): float(value) for name, value in tags.items()}
    return {str(name): 1.0 for name in tags}


def build_game_index(games_json: Path, tags: list[str], coarse: bool = False):
    import numpy as np

    payload = json.loads(Path(games_json).read_text(encoding="utf-8"))
    items = payload.items() if isinstance(payload, dict) else enumerate(payload)
    tag_to_id = {tag: index for index, tag in enumerate(tags)}

    rows = []
    names = []
    appids = []
    for key, record in items:
        if not isinstance(record, dict):
            continue
        vector = np.zeros(len(tags), dtype=np.float32)
        raw_tags = game_tag_dict(record)
        if coarse:
            raw_tags = coarsen_tag_dict(raw_tags)
        if not raw_tags:
            continue
        max_weight = max(raw_tags.values()) if raw_tags else 1.0
        max_weight = max(max_weight, 1.0)
        for tag, weight in raw_tags.items():
            tag_id = tag_to_id.get(tag)
            if tag_id is not None:
                vector[tag_id] = min(float(weight) / max_weight, 1.0)
        if vector.any():
            appid = str(record.get("steam_appid") or record.get("appid") or key)
            rows.append(vector)
            names.append(str(record.get("name") or appid))
            appids.append(appid)

    if not rows:
        return [], [], np.zeros((0, len(tags)), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    matrix = np.stack(rows, axis=0)
    norms = np.linalg.norm(matrix, axis=1).astype(np.float32)
    return appids, names, matrix, norms


class PredictorWorker(QtCore.QObject):
    status = QtCore.pyqtSignal(str)
    ready = QtCore.pyqtSignal(str, str, str)
    result = QtCore.pyqtSignal(list, list, int)
    error = QtCore.pyqtSignal(str)

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.device = None
        self.embedder = None
        self.encoder = None
        self.probe = None          # portable linear artifact (dict)
        self.keep_mask = None      # which tags to predict/match (per --tag-filter)
        self.tags = []
        self.encoder_path = None
        self.head_path = None
        self.games_json_path = None
        self.game_appids = []
        self.game_names = []
        self.game_matrix = None
        self.game_norms = None

    @QtCore.pyqtSlot(str, str, str)
    def load(self, encoder_value: str = "", head_value: str = "", games_value: str = "") -> None:
        try:
            import h5py
            import torch

            # Resolve the deployable linear probe first; it records the encoder
            # checkpoint it was fit on, which we use unless one is given explicitly.
            head_path = resolve_optional_path(
                head_value or self.args.tag_head,
                [
                    "VICReg_review/heads/tag_probe_linear*.pt",
                    "VICReg_review/heads/**/tag_probe_linear*.pt",
                ],
                "linear tag probe artifact (train_tag_probe.py --export-head)",
            )
            probe = torch.load(head_path, map_location="cpu", weights_only=False)
            if not isinstance(probe, dict) or probe.get("kind") != "linear_tag_probe":
                raise ValueError(
                    f"{head_path} is not a linear_tag_probe artifact. Re-export with "
                    "train_tag_probe.py --export-head."
                )

            encoder_request = encoder_value or self.args.encoder_checkpoint or probe.get("encoder_checkpoint")
            encoder_path = resolve_optional_path(
                encoder_request,
                [
                    "VICReg_review/heads/sweep_adv/vicreg_adv*_best*.pt",
                    "VICReg_review/heads/gui_run/vicreg_review_h5_best*.pt",
                    "VICReg_review/heads/vicreg_review_h5_best*.pt",
                ],
                "VICReg encoder checkpoint",
            )
            # Candidate game pool must be the in-domain intersection (the 293
            # training games), not all 65k games.json entries. test_games.json is
            # that intersection with emotional tags already filtered out.
            games_json = resolve_optional_path(
                games_value or self.args.games_json,
                [
                    "VICReg_review/tags/test_games.json",
                    "game_review_data/**/games.json",
                ],
                "test_games.json (run VICReg_review/build_test_games.py)",
            )

            if self.embedder is None:
                self.status.emit("loading local embedding model")
                self.embedder = LocalEmbedder(
                    self.args.local_model,
                    device=self.args.device,
                    batch_size=self.args.batch_size,
                )
            self.device = torch.device(self.embedder.device)

            input_dim = self.args.input_dim
            if self.args.h5:
                with h5py.File(self.args.h5, "r") as h5:
                    input_dim = int(h5.attrs["input_dim"])

            self.status.emit("loading VICReg encoder")
            self.encoder, _, _, _ = load_frozen_encoder(encoder_path, input_dim, self.device)
            self.encoder.float().eval()

            self.status.emit("loading linear tag probe")
            self.probe = probe
            self.tags = list(probe.get("tags") or load_tags_fallback(Path(self.args.tags_dir)))
            if not self.tags:
                self.tags = [f"tag_{index}" for index in range(len(probe["intercept"]))]
            self.keep_mask = self._build_tag_keep_mask()

            self.status.emit("loading game table")
            self.game_appids, self.game_names, self.game_matrix, self.game_norms = build_game_index(
                games_json, self.tags, bool(probe.get("coarse_aliases"))
            )
            self.encoder_path = encoder_path
            self.head_path = head_path
            self.games_json_path = games_json

            self.ready.emit(str(encoder_path), str(head_path), str(games_json))
            self.status.emit("ready")
        except BaseException as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")

    def _build_tag_keep_mask(self):
        """Which tags to predict/match, per --tag-filter.

        non_emotional (default): drop the `subjective` affect/quality group.
        content: keep only mechanics+story (from the probe's content_mask).
        all: keep everything.
        """
        import numpy as np

        n = len(self.tags)
        mode = self.args.tag_filter
        if mode == "all":
            return np.ones(n, dtype=bool)
        if mode == "content":
            mask = self.probe.get("content_mask") if self.probe else None
            if mask is not None and np.asarray(mask).any():
                return np.asarray(mask, dtype=bool)
        # non_emotional: drop subjective group read from tag_groups.json.
        groups_path = Path(self.args.tags_dir) / "tag_groups.json"
        if groups_path.exists():
            subjective = set(json.loads(groups_path.read_text(encoding="utf-8")).get("subjective", []))
            return np.array([t not in subjective for t in self.tags], dtype=bool)
        return np.ones(n, dtype=bool)

    @QtCore.pyqtSlot(str)
    def predict(self, text: str) -> None:
        try:
            import numpy as np
            import torch

            if self.embedder is None or self.encoder is None or self.probe is None:
                self.error.emit("Models are not ready yet.")
                return

            sentences = split_text(text, self.args.max_sentences)
            if not sentences:
                self.error.emit("Please enter some text first.")
                return

            self.status.emit(f"embedding {len(sentences)} sentence(s)")
            vectors = self.embedder.embed(sentences)
            vt = torch.tensor(vectors, dtype=torch.float32, device=self.device)
            n_sent = vt.shape[0]

            # Match how the probe's features were built: average feature_views
            # sub-sampled views (sample_fraction of sentences), not one full pass.
            # A single full forward is out-of-distribution vs the training feature
            # and makes the standardized code blow up (saturated probabilities).
            views = max(1, int(self.probe.get("feature_views") or 4))
            frac = float(self.probe.get("sample_fraction") or 0.6)
            rng = np.random.default_rng(0)
            self.status.emit("running encoder and linear probe")
            codes = []
            with torch.no_grad():
                for _ in range(views):
                    if n_sent > 2:
                        k = max(1, int(np.ceil(n_sent * frac)))
                        idx = np.sort(rng.choice(n_sent, size=k, replace=False))
                        sub = vt[idx]
                    else:
                        sub = vt
                    code = self.encoder(sub.unsqueeze(0), key_padding_mask=None)  # (1, L, D)
                    codes.append(code.squeeze(0).float())
            feats = torch.stack(codes, dim=0).mean(dim=0).cpu().numpy()  # (num_latents, output_dim)

            # Pool + normalize + per-tag logistic exactly as train_tag_probe does.
            pooled = pool_features(feats[None, ...], self.probe["pool"])[0]
            if self.probe.get("normalizer", "standard") == "l2":
                x_probe = pooled / (np.linalg.norm(pooled) + float(self.probe.get("norm_eps", 1e-8)))
            else:
                x_probe = (pooled - self.probe["scaler_mean"]) / self.probe["scaler_scale"]
                x_probe = np.clip(x_probe, -10.0, 10.0)
            logits = x_probe @ self.probe["coef"].T + self.probe["intercept"]
            probs = 1.0 / (1.0 + np.exp(-logits))
            probs = np.where(self.probe["trained_mask"], probs, 0.0).astype(np.float32)
            keyword_weight = float(self.probe.get("keyword_weight", 0.0) or 0.0)
            if keyword_weight > 0 and self.probe.get("coarse_aliases"):
                prior = keyword_scores(text, self.tags)
                keyword_weight = min(max(keyword_weight, 0.0), 1.0)
                probs = ((1.0 - keyword_weight) * probs + keyword_weight * prior).astype(np.float32)
            presence_scores = probs.tolist()

            keep = self.keep_mask if self.keep_mask is not None else np.ones(len(self.tags), dtype=bool)
            rows = sorted(
                (tag, score) for tag, score, k in zip(self.tags, presence_scores, keep) if k
            )
            rows.sort(key=lambda item: item[1], reverse=True)
            game_rows = self.match_games(presence_scores)
            if self.args.top_k > 0:
                rows = rows[: self.args.top_k]
            self.result.emit(rows, game_rows, len(sentences))
            self.status.emit("ready")
        except BaseException as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")

    def match_games(self, scores: list[float]) -> list[tuple[str, str, float, str]]:
        import numpy as np

        if self.game_matrix is None or self.game_matrix.shape[0] == 0:
            return []
        pred = np.asarray(scores, dtype=np.float32)
        # Match on the kept (non-emotional / content) tags only; emotional tags are
        # filtered from test_games.json anyway and only add cosine noise here.
        if self.keep_mask is not None and not self.args.match_all_tags:
            pred = pred * np.asarray(self.keep_mask, dtype=np.float32)
        pred_norm = float(np.linalg.norm(pred))
        if pred_norm <= 1e-8:
            return []
        numerators = self.game_matrix @ pred
        similarities = numerators / ((self.game_norms + 1e-8) * pred_norm)
        top_count = min(self.args.game_top_k, similarities.shape[0])
        if top_count <= 0:
            return []
        top_indices = np.argpartition(-similarities, top_count - 1)[:top_count]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        rows = []
        for game_index in top_indices:
            tag_weights = self.game_matrix[game_index] * pred
            tag_ids = np.argsort(-tag_weights)[:5]
            matched = [
                self.tags[tag_id]
                for tag_id in tag_ids
                if tag_weights[tag_id] > 0
            ]
            rows.append(
                (
                    self.game_appids[game_index],
                    self.game_names[game_index],
                    float(similarities[game_index]),
                    ", ".join(matched),
                )
            )
        return rows


class ValidationWindow(QtWidgets.QMainWindow):
    predict_requested = QtCore.pyqtSignal(str)
    load_requested = QtCore.pyqtSignal(str, str, str)

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.setWindowTitle("VICReg Review Validation")
        self.resize(980, 760)
        self._build_ui()
        self._build_worker()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.status_label = QtWidgets.QLabel("loading")
        self.encoder_label = QtWidgets.QLabel("encoder: auto")
        self.head_label = QtWidgets.QLabel("tag head: auto")
        self.games_label = QtWidgets.QLabel("games: auto")
        self.encoder_label.setWordWrap(True)
        self.head_label.setWordWrap(True)
        self.games_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addWidget(self.encoder_label)
        layout.addWidget(self.head_label)
        layout.addWidget(self.games_label)

        self.text_edit = QtWidgets.QPlainTextEdit()
        self.text_edit.setPlaceholderText("输入游戏评论文本。可以是一段长文本，也可以多行输入。")
        self.text_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.text_edit.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout.addWidget(self.text_edit, stretch=2)

        controls = QtWidgets.QHBoxLayout()
        self.load_button = QtWidgets.QPushButton("加载")
        self.predict_button = QtWidgets.QPushButton("预测标签分数")
        self.predict_button.setEnabled(False)
        self.load_button.clicked.connect(self.on_load_clicked)
        self.predict_button.clicked.connect(self.on_predict_clicked)
        self.count_label = QtWidgets.QLabel("sentences: 0")
        controls.addWidget(self.load_button)
        controls.addWidget(self.predict_button)
        controls.addWidget(self.count_label)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.tabs = QtWidgets.QTabWidget()
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Rank", "Tag", "Probability"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

        self.game_table = QtWidgets.QTableWidget(0, 5)
        self.game_table.setHorizontalHeaderLabels(["Rank", "AppID", "Game", "Score", "Matched tags"])
        self.game_table.verticalHeader().setVisible(False)
        self.game_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.game_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.game_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.game_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.game_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.game_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.game_table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.tabs.addTab(self.table, "标签分数")
        self.tabs.addTab(self.game_table, "最可能游戏")
        layout.addWidget(self.tabs, stretch=3)

    def _build_worker(self) -> None:
        self.thread = QtCore.QThread(self)
        self.worker = PredictorWorker(self.args)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.request_auto_load)
        self.worker.status.connect(self.set_status)
        self.worker.ready.connect(self.on_ready)
        self.worker.result.connect(self.on_result)
        self.worker.error.connect(self.on_error)
        self.predict_requested.connect(self.worker.predict)
        self.load_requested.connect(self.worker.load)
        self.thread.start()

    @QtCore.pyqtSlot()
    def request_auto_load(self) -> None:
        self.load_requested.emit("", "", "")

    @QtCore.pyqtSlot(str)
    def set_status(self, text: str) -> None:
        self.status_label.setText(f"status: {text}")

    @QtCore.pyqtSlot(str, str, str)
    def on_ready(self, encoder_path: str, head_path: str, games_path: str) -> None:
        self.encoder_label.setText(f"encoder: {encoder_path}")
        self.head_label.setText(f"tag head: {head_path}")
        self.games_label.setText(f"games: {games_path}")
        self.load_button.setEnabled(True)
        self.predict_button.setEnabled(True)

    def on_load_clicked(self) -> None:
        encoder_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择 VICReg encoder checkpoint",
            str(DEFAULT_GUI_RUN_DIR),
            "PyTorch checkpoints (*.pt);;All files (*)",
        )
        if not encoder_path:
            return
        head_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择 tag probe head checkpoint",
            str(DEFAULT_GUI_RUN_DIR),
            "PyTorch checkpoints (*.pt);;All files (*)",
        )
        if not head_path:
            return
        games_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择 games.json，可取消使用自动路径",
            str(DEFAULT_GAMES_JSON.parent),
            "JSON files (*.json);;All files (*)",
        )
        self.load_button.setEnabled(False)
        self.predict_button.setEnabled(False)
        self.table.setRowCount(0)
        self.game_table.setRowCount(0)
        self.set_status("loading selected model")
        self.load_requested.emit(encoder_path, head_path, games_path)

    def on_predict_clicked(self) -> None:
        self.predict_button.setEnabled(False)
        self.table.setRowCount(0)
        self.game_table.setRowCount(0)
        self.set_status("queued")
        self.predict_requested.emit(self.text_edit.toPlainText())

    @QtCore.pyqtSlot(list, list, int)
    def on_result(self, rows: list, game_rows: list, sentence_count: int) -> None:
        self.count_label.setText(f"sentences: {sentence_count}")
        self.table.setRowCount(len(rows))
        for row_index, (tag, score) in enumerate(rows):
            rank_item = QtWidgets.QTableWidgetItem(str(row_index + 1))
            tag_item = QtWidgets.QTableWidgetItem(str(tag))
            score_item = QtWidgets.QTableWidgetItem(f"{float(score):.6f}")
            score_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row_index, 0, rank_item)
            self.table.setItem(row_index, 1, tag_item)
            self.table.setItem(row_index, 2, score_item)
        self.game_table.setRowCount(len(game_rows))
        for row_index, (appid, name, score, matched) in enumerate(game_rows):
            rank_item = QtWidgets.QTableWidgetItem(str(row_index + 1))
            appid_item = QtWidgets.QTableWidgetItem(str(appid))
            name_item = QtWidgets.QTableWidgetItem(str(name))
            score_item = QtWidgets.QTableWidgetItem(f"{float(score):.6f}")
            matched_item = QtWidgets.QTableWidgetItem(str(matched))
            score_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.game_table.setItem(row_index, 0, rank_item)
            self.game_table.setItem(row_index, 1, appid_item)
            self.game_table.setItem(row_index, 2, name_item)
            self.game_table.setItem(row_index, 3, score_item)
            self.game_table.setItem(row_index, 4, matched_item)
        self.predict_button.setEnabled(True)

    @QtCore.pyqtSlot(str)
    def on_error(self, message: str) -> None:
        self.set_status(f"error: {message}")
        self.load_button.setEnabled(True)
        self.predict_button.setEnabled(self.worker.encoder is not None and self.worker.probe is not None)
        QtWidgets.QMessageBox.warning(self, "Validation error", message)

    def closeEvent(self, event) -> None:
        self.thread.quit()
        self.thread.wait(3000)
        event.accept()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyQt6 VICReg tag validation UI.")
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--tag-head", default=None)
    parser.add_argument("--games-json", default=None)
    parser.add_argument("--tags-dir", default=str(DEFAULT_TAGS_DIR))
    parser.add_argument("--h5", default=None, help="Optional H5 path to read input_dim from.")
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-sentences", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=0, help="0 shows every tag.")
    parser.add_argument("--game-top-k", type=int, default=20)
    parser.add_argument("--tag-filter", choices=["non_emotional", "content", "all"], default="non_emotional",
                        help="Which tags to predict and match on. non_emotional drops the "
                             "subjective affect group; content keeps mechanics+story only.")
    parser.add_argument("--match-all-tags", action="store_true",
                        help="Override --tag-filter for game matching and use every tag.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    app = QtWidgets.QApplication(sys.argv)
    window = ValidationWindow(args)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
