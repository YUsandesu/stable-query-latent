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

from backheads.model import RecommendationRateHead  # noqa: E402
from game_review_data.embedding_data import DEFAULT_LOCAL_MODEL, LocalEmbedder  # noqa: E402
from VICReg_review.train_tag_probe import load_frozen_encoder, pool_features  # noqa: E402
try:
    from VICReg_review.tap_mapping import load_tap_mapping, map_tag_dict, keyword_scores
except ImportError:  # pragma: no cover
    from tap_mapping import load_tap_mapping, map_tag_dict, keyword_scores


DEFAULT_HEADS_DIR = ROOT / "VICReg_review" / "heads"
DEFAULT_GUI_RUN_DIR = DEFAULT_HEADS_DIR / "gui_run"
DEFAULT_TAGS_DIR = ROOT / "VICReg_review" / "tags"
DEFAULT_H5 = ROOT / "VICReg_review" / "h5" / "game_review_cleaned_3_sentences.h5"
DEFAULT_GAMES_JSON = ROOT / "game_review_data" / "Steam Games Metadata and Player Reviews (2020–2024" / "games.json"
DEFAULT_RECOMMENDATION_HEAD = ROOT / "backheads" / "heads" / "recommendation_head.pt"
DEFAULT_LINEAR_RECOMMENDATION_PROBE = ROOT / "backheads" / "heads" / "recommendation_linear_probe.pt"
DEFAULT_VICREG_RECOMMENDATION_PROBE = ROOT / "backheads" / "heads" / "recommendation_vicreg_linear_probe.pt"
DEFAULT_RECOMMENDATION_CACHE = ROOT / "backheads" / "heads" / "recommendation_features_mean_std.npz"
DEFAULT_VICREG_RECOMMENDATION_CACHE = ROOT / "backheads" / "heads" / "recommendation_vicreg_features.npz"


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


def summarize_recommendation_vectors(vectors, mode: str):
    import numpy as np

    vectors = np.asarray(vectors, dtype=np.float32)
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    if mode == "mean":
        return mean
    if mode == "mean_std":
        return np.concatenate([mean, std], axis=0)
    if mode == "mean_std_extrema":
        return np.concatenate([mean, std, vectors.min(axis=0), vectors.max(axis=0)], axis=0)
    raise ValueError(f"Unknown recommendation feature mode: {mode}")


def game_tag_dict(record: dict) -> dict[str, float]:
    tags = record.get("tags") or {}
    if isinstance(tags, dict):
        return {str(name): float(value) for name, value in tags.items()}
    return {str(name): 1.0 for name in tags}


def _decode_h5_string(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_training_appids(h5_path: Path) -> set[str]:
    import h5py

    with h5py.File(h5_path, "r") as h5:
        if "appids" in h5:
            return {_decode_h5_string(x) for x in h5["appids"][:]}
        return {_decode_h5_string(x).split("_")[0] for x in h5["game_names"][:]}


def build_game_index(games_json: Path, tags: list[str], training_h5: Path | None = None):
    import numpy as np

    games_json = Path(games_json)
    if games_json.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(games_json, "r") as h5:
            if "tap_names" not in h5 or "tap_raw_counts" not in h5:
                raise ValueError(f"{games_json} has no tap_names/tap_raw_counts datasets.")
            h5_tags = [_decode_h5_string(x) for x in h5["tap_names"][:]]
            h5_index = {tag: index for index, tag in enumerate(h5_tags)}
            cols = [h5_index[tag] for tag in tags if tag in h5_index]
            matrix = h5["tap_raw_counts"][:, cols].astype(np.float32)
            out_tags = [h5_tags[col] for col in cols]
            if out_tags != tags:
                aligned = np.zeros((matrix.shape[0], len(tags)), dtype=np.float32)
                for out_col, tag in enumerate(out_tags):
                    aligned[:, tags.index(tag)] = matrix[:, out_col]
                matrix = aligned
            appids = [_decode_h5_string(x) for x in h5.get("appids", h5["game_names"])[:]]
            if "game_titles" in h5:
                names = [_decode_h5_string(x) for x in h5["game_titles"][:]]
            else:
                names = [_decode_h5_string(x) for x in h5["game_names"][:]]
        norms = np.linalg.norm(matrix, axis=1).astype(np.float32)
        return appids, names, matrix, norms

    allowed_appids = read_training_appids(training_h5) if training_h5 else None
    payload = json.loads(games_json.read_text(encoding="utf-8"))
    items = payload.items() if isinstance(payload, dict) else enumerate(payload)
    tag_to_id = {tag: index for index, tag in enumerate(tags)}
    spec = load_tap_mapping()

    rows = []
    names = []
    appids = []
    for key, record in items:
        if not isinstance(record, dict):
            continue
        appid = str(record.get("steam_appid") or record.get("appid") or key)
        if allowed_appids is not None and appid not in allowed_appids:
            continue
        vector = np.zeros(len(tags), dtype=np.float32)
        raw_tags = map_tag_dict(game_tag_dict(record), spec)
        if not raw_tags:
            continue
        max_weight = max(raw_tags.values()) if raw_tags else 1.0
        max_weight = max(max_weight, 1.0)
        for tag, weight in raw_tags.items():
            tag_id = tag_to_id.get(tag)
            if tag_id is not None:
                vector[tag_id] = min(float(weight) / max_weight, 1.0)
        if vector.any():
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
    ready = QtCore.pyqtSignal(str, str, str, str, str)
    result = QtCore.pyqtSignal(list, list, list, list, int)
    error = QtCore.pyqtSignal(str)

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.device = None
        self.embedder = None
        self.encoder = None
        self.probe = None          # portable linear artifact (dict)
        self.pxi_probe = None      # portable PXI artifact (dict), optional
        self.recommendation_head = None
        self.recommendation_checkpoint = None
        self.recommendation_cache = None
        self.recommendation_head_path = None
        self.keep_mask = None      # which tags to predict/match (per --tag-filter)
        self.tags = []
        self.encoder_path = None
        self.head_path = None
        self.pxi_head_path = None
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

            if self.args.pxi_head:
                candidate = Path(self.args.pxi_head)
                pxi_head_path = candidate if candidate.is_absolute() else ROOT / candidate
            else:
                pxi_head_path = newest_existing([
                    "VICReg_review/heads/pxi_probe_linear*.pt",
                    "VICReg_review/heads/**/pxi_probe_linear*.pt",
                ])

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
            # Candidate game pool is restricted to the VICReg training H5. If a
            # JSON metadata file is passed, build_game_index still filters it by
            # the H5 appid set.
            games_path = resolve_optional_path(
                games_value or self.args.games_json or self.args.h5,
                [
                    "VICReg_review/h5/game_review_cleaned_3_sentences.h5",
                ],
                "TAP-labeled H5 (run VICReg_review/build_review_h5.py)",
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
            h5_path = Path(self.args.h5)
            if self.args.h5:
                with h5py.File(h5_path, "r") as h5:
                    input_dim = int(h5.attrs["input_dim"])

            self.status.emit("loading VICReg encoder")
            self.encoder, _, _, _ = load_frozen_encoder(encoder_path, input_dim, self.device)
            self.encoder.float().eval()

            self.status.emit("loading linear tag probe")
            self.probe = probe
            self.pxi_probe = None
            self.pxi_head_path = None
            if pxi_head_path and Path(pxi_head_path).exists():
                self.status.emit("loading PXI probe")
                pxi_probe = torch.load(pxi_head_path, map_location="cpu", weights_only=False)
                if isinstance(pxi_probe, dict) and pxi_probe.get("kind") == "linear_pxi_probe":
                    self.pxi_probe = pxi_probe
                    self.pxi_head_path = Path(pxi_head_path)

            self.recommendation_head = None
            self.recommendation_checkpoint = None
            self.recommendation_cache = None
            self.recommendation_head_path = None
            recommendation_head_path = Path(self.args.recommendation_head)
            if not recommendation_head_path.is_absolute():
                recommendation_head_path = ROOT / recommendation_head_path
            recommendation_cache_path = Path(self.args.recommendation_cache)
            if not recommendation_cache_path.is_absolute():
                recommendation_cache_path = ROOT / recommendation_cache_path
            if recommendation_head_path.exists() and recommendation_cache_path.exists():
                self.status.emit("loading recommendation head")
                rec_ckpt = torch.load(recommendation_head_path, map_location="cpu", weights_only=False)
                if rec_ckpt.get("kind") in ("linear_recommendation_probe", "vicreg_linear_recommendation_probe"):
                    self.recommendation_head = "linear"
                else:
                    rec_head = RecommendationRateHead(
                        input_dim=int(rec_ckpt["input_dim"]),
                        hidden_dims=tuple(rec_ckpt["hidden_dims"]),
                        dropout=float(rec_ckpt["dropout"]),
                    )
                    rec_head.load_state_dict(rec_ckpt["state_dict"])
                    rec_head.eval()
                    self.recommendation_head = rec_head
                self.recommendation_checkpoint = rec_ckpt
                self.recommendation_cache = self._load_recommendation_cache(recommendation_cache_path)
                self.recommendation_head_path = recommendation_head_path
            self.tags = list(probe.get("tags") or [])
            if not self.tags:
                self.tags = [f"tag_{index}" for index in range(len(probe["intercept"]))]
            self.keep_mask = self._build_tag_keep_mask()

            self.status.emit("loading game table")
            self.game_appids, self.game_names, self.game_matrix, self.game_norms = build_game_index(
                games_path, self.tags, h5_path
            )
            self.encoder_path = encoder_path
            self.head_path = head_path
            self.games_json_path = games_path

            self.ready.emit(
                str(encoder_path),
                str(head_path),
                f"{games_path} ({len(self.game_appids)} VICReg-training games)",
                str(self.pxi_head_path) if self.pxi_head_path else "",
                str(self.recommendation_head_path) if self.recommendation_head_path else "",
            )
            self.status.emit("ready")
        except BaseException as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")

    def _load_recommendation_cache(self, path: Path) -> dict:
        import numpy as np

        data = np.load(path, allow_pickle=True)
        cache = {key: data[key] for key in data.files}
        cache["appid_to_index"] = {
            str(appid): index for index, appid in enumerate(cache.get("appids", []))
        }
        return cache

    def _build_tag_keep_mask(self):
        """Which tags to predict/match, per --tag-filter.

        non_emotional/content: same as all for TAP labels.
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
            if keyword_weight > 0 and self.probe.get("tap_mapping_json"):
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
            pxi_rows = self.predict_pxi(feats)
            recommendation_rows = self.predict_recommendation(vectors, feats, game_rows)
            if self.args.top_k > 0:
                rows = rows[: self.args.top_k]
            self.result.emit(rows, game_rows, pxi_rows, recommendation_rows, len(sentences))
            self.status.emit("ready")
        except BaseException as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")

    def predict_recommendation(
        self,
        vectors,
        feats,
        game_rows: list[tuple[str, str, float, str]],
    ) -> list[tuple[str, str, str, float | None, float, float, float | None, float | None]]:
        import numpy as np
        import torch

        if not self.recommendation_head or not self.recommendation_checkpoint:
            return []
        ckpt = self.recommendation_checkpoint
        if ckpt.get("kind") == "vicreg_linear_recommendation_probe":
            feature = pool_features(feats[None, ...], ckpt.get("pool", "stats"))[0].astype(np.float32)
        else:
            feature_mode = str(ckpt.get("feature_mode", "mean_std"))
            feature = summarize_recommendation_vectors(vectors, feature_mode)
        text_rates = self._predict_recommendation_rates(feature)

        rows = [
            (
                "当前输入",
                "",
                "输入文本",
                None,
                float(text_rates[0]),
                float(text_rates[1]),
                None,
                None,
            )
        ]
        cache = self.recommendation_cache or {}
        appid_to_index = cache.get("appid_to_index") or {}
        if not appid_to_index:
            return rows

        max_rows = max(1, int(self.args.recommendation_top_k))
        for appid, name, similarity, _ in game_rows[:max_rows]:
            index = appid_to_index.get(str(appid))
            if index is None:
                continue
            review_feature = cache["X"][index].astype(np.float32)
            review_rates = self._predict_recommendation_rates(review_feature)
            true_rates = cache["y"][index].astype(np.float32)
            rows.append(
                (
                    "相似游戏",
                    str(appid),
                    str(name),
                    float(similarity),
                    float(review_rates[0]),
                    float(review_rates[1]),
                    float(true_rates[0]),
                    float(true_rates[1]),
                )
            )
        return rows

    def _predict_recommendation_rates(self, feature):
        import numpy as np
        import torch

        ckpt = self.recommendation_checkpoint
        feature = np.asarray(feature, dtype=np.float32)
        feature_mean = ckpt["feature_mean"].astype(np.float32)
        feature_std = np.maximum(ckpt["feature_std"].astype(np.float32), 1e-6)
        normalized = (feature - feature_mean) / feature_std
        if ckpt.get("kind") in ("linear_recommendation_probe", "vicreg_linear_recommendation_probe"):
            value = float(normalized @ ckpt["coef"].astype(np.float32) + float(ckpt["intercept"]))
            if ckpt.get("target_transform") == "logit":
                positive = 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))
            else:
                positive = np.clip(value, 0.0, 1.0)
            return np.asarray([positive, 1.0 - positive], dtype=np.float32)
        with torch.no_grad():
            return (
                self.recommendation_head
                .predict_rates(torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0))
                .numpy()[0]
            )

    def predict_pxi(self, feats) -> list[tuple[str, str, float]]:
        import numpy as np

        probe = self.pxi_probe
        if not probe:
            return []
        pooled = pool_features(feats[None, ...], probe.get("pool", "stats"))[0].astype(np.float32)
        if probe.get("normalizer", "standard") == "l2":
            x = pooled / (np.linalg.norm(pooled) + float(probe.get("norm_eps", 1e-8)))
        else:
            x = (pooled - probe["scaler_mean"]) / probe["scaler_scale"]
        comps = probe.get("pca_components")
        if comps is not None:
            x = (x - probe["pca_mean"]) @ np.asarray(comps).T
        values = x @ probe["ridge_coef"].T + probe["ridge_intercept"]
        target_min = probe.get("target_min")
        target_max = probe.get("target_max")
        if target_min is not None and target_max is not None:
            values = np.clip(values, np.asarray(target_min), np.asarray(target_max))
        functional = set(probe.get("functional_dims") or [])
        rows = []
        for dim, value in zip(probe["dims"], values):
            group = "functional" if dim in functional else "psychological"
            rows.append((group, dim, float(value)))
        rows.sort(key=lambda item: (item[0] != "functional", item[1]))
        return rows

    def match_games(self, scores: list[float]) -> list[tuple[str, str, float, str]]:
        import numpy as np

        if self.game_matrix is None or self.game_matrix.shape[0] == 0:
            return []
        pred = np.asarray(scores, dtype=np.float32)
        # Match on the kept TAP tags only.
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
        self.pxi_label = QtWidgets.QLabel("pxi head: auto")
        self.recommendation_label = QtWidgets.QLabel("recommendation head: auto")
        self.games_label = QtWidgets.QLabel("games: auto")
        self.encoder_label.setWordWrap(True)
        self.head_label.setWordWrap(True)
        self.pxi_label.setWordWrap(True)
        self.recommendation_label.setWordWrap(True)
        self.games_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addWidget(self.encoder_label)
        layout.addWidget(self.head_label)
        layout.addWidget(self.pxi_label)
        layout.addWidget(self.recommendation_label)
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

        self.pxi_table = QtWidgets.QTableWidget(0, 3)
        self.pxi_table.setHorizontalHeaderLabels(["Group", "PXI dimension", "Predicted mean"])
        self.pxi_table.verticalHeader().setVisible(False)
        self.pxi_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.pxi_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.pxi_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.pxi_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.pxi_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

        self.recommendation_table = QtWidgets.QTableWidget(0, 8)
        self.recommendation_table.setHorizontalHeaderLabels(
            [
                "Type",
                "AppID",
                "Name",
                "Similarity",
                "Pred positive",
                "Pred negative",
                "True positive",
                "True negative",
            ]
        )
        self.recommendation_table.verticalHeader().setVisible(False)
        self.recommendation_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.recommendation_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.recommendation_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.recommendation_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for col in range(2, 8):
            self.recommendation_table.horizontalHeader().setSectionResizeMode(
                col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )

        self.tabs.addTab(self.table, "标签分数")
        self.tabs.addTab(self.game_table, "最可能游戏")
        self.tabs.addTab(self.pxi_table, "PXI")
        self.tabs.addTab(self.recommendation_table, "好评率")
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

    @QtCore.pyqtSlot(str, str, str, str, str)
    def on_ready(
        self,
        encoder_path: str,
        head_path: str,
        games_path: str,
        pxi_path: str,
        recommendation_path: str,
    ) -> None:
        self.encoder_label.setText(f"encoder: {encoder_path}")
        self.head_label.setText(f"tag head: {head_path}")
        self.pxi_label.setText(f"pxi head: {pxi_path or 'not loaded'}")
        self.recommendation_label.setText(
            f"recommendation head: {recommendation_path or 'not loaded'}"
        )
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
            "选择候选池（默认使用 VICReg 训练 H5；JSON 会按训练 appid 过滤）",
            str(DEFAULT_H5.parent),
            "H5/JSON files (*.h5 *.json);;All files (*)",
        )
        self.load_button.setEnabled(False)
        self.predict_button.setEnabled(False)
        self.table.setRowCount(0)
        self.game_table.setRowCount(0)
        self.pxi_table.setRowCount(0)
        self.recommendation_table.setRowCount(0)
        self.set_status("loading selected model")
        self.load_requested.emit(encoder_path, head_path, games_path)

    def on_predict_clicked(self) -> None:
        self.predict_button.setEnabled(False)
        self.table.setRowCount(0)
        self.game_table.setRowCount(0)
        self.pxi_table.setRowCount(0)
        self.recommendation_table.setRowCount(0)
        self.set_status("queued")
        self.predict_requested.emit(self.text_edit.toPlainText())

    @QtCore.pyqtSlot(list, list, list, list, int)
    def on_result(
        self,
        rows: list,
        game_rows: list,
        pxi_rows: list,
        recommendation_rows: list,
        sentence_count: int,
    ) -> None:
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
        self.pxi_table.setRowCount(len(pxi_rows))
        for row_index, (group, dim, value) in enumerate(pxi_rows):
            group_item = QtWidgets.QTableWidgetItem(str(group))
            dim_item = QtWidgets.QTableWidgetItem(str(dim))
            value_item = QtWidgets.QTableWidgetItem(f"{float(value):.6f}")
            value_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.pxi_table.setItem(row_index, 0, group_item)
            self.pxi_table.setItem(row_index, 1, dim_item)
            self.pxi_table.setItem(row_index, 2, value_item)
        self.recommendation_table.setRowCount(len(recommendation_rows))
        for row_index, values in enumerate(recommendation_rows):
            kind, appid, name, similarity, pred_pos, pred_neg, true_pos, true_neg = values
            display_values = [
                str(kind),
                str(appid),
                str(name),
                "" if similarity is None else f"{float(similarity):.6f}",
                f"{float(pred_pos):.6f}",
                f"{float(pred_neg):.6f}",
                "" if true_pos is None else f"{float(true_pos):.6f}",
                "" if true_neg is None else f"{float(true_neg):.6f}",
            ]
            for col, value in enumerate(display_values):
                item = QtWidgets.QTableWidgetItem(value)
                if col >= 3:
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                self.recommendation_table.setItem(row_index, col, item)
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
    parser.add_argument("--pxi-head", default=None)
    parser.add_argument("--recommendation-head", default=str(DEFAULT_VICREG_RECOMMENDATION_PROBE))
    parser.add_argument("--recommendation-cache", default=str(DEFAULT_VICREG_RECOMMENDATION_CACHE))
    parser.add_argument("--recommendation-top-k", type=int, default=3,
                        help="How many similar games to compare in the recommendation-rate tab.")
    parser.add_argument("--games-json", default=None,
                        help="Optional candidate metadata. JSON inputs are filtered to appids present in --h5.")
    parser.add_argument("--tags-dir", default=str(DEFAULT_TAGS_DIR))
    parser.add_argument("--h5", default=str(DEFAULT_H5), help="H5 path with vectors and TAP labels.")
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
