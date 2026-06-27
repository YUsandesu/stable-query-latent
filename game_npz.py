"""Compact per-game vector storage for VICReg review training.

Each game is one ``.npz`` holding just two arrays:

    vectors        float16  (total_sentences, dim)   all sentence vectors,
                                                      concatenated review by review
    review_offsets int64    (num_reviews + 1)         offsets into ``vectors``;
                                                      review r = vectors[off[r]:off[r+1]]

This is ~10x smaller than the equivalent embedded JSON (which stores every float
as text) and still preserves the review boundaries VICReg needs for review-level
view sampling. It is the same ragged layout the H5 uses (``review_offsets``), but
one file per game so the corpus stays prefetch/stream friendly.

Pure numpy, no torch, so both the embedding pipeline (game_review_data) and the
trainer (VICReg_review) can import it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

VECTORS_KEY = "vectors"
REVIEW_OFFSETS_KEY = "review_offsets"


def save_game_npz(path, review_vectors, dtype="float16") -> int:
    """Save a list of per-review 2D arrays (each ``n_sent_i x dim``) to one .npz.

    Empty reviews are dropped. Writes atomically (tmp + replace). Returns the
    number of reviews actually written.
    """
    arrays = []
    for review in review_vectors:
        arr = np.asarray(review, dtype=dtype)
        if arr.ndim == 2 and arr.shape[0] > 0:
            arrays.append(arr)

    if arrays:
        vectors = np.concatenate(arrays, axis=0).astype(dtype, copy=False)
        lengths = np.fromiter((a.shape[0] for a in arrays), dtype=np.int64, count=len(arrays))
        review_offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        np.cumsum(lengths, out=review_offsets[1:])
    else:
        # Valid-but-empty file so resume logic can still skip it.
        vectors = np.zeros((0, 0), dtype=dtype)
        review_offsets = np.zeros((1,), dtype=np.int64)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        # Save to an explicit handle so numpy doesn't append a second ".npz".
        with open(tmp, "wb") as handle:
            np.savez(handle, **{VECTORS_KEY: vectors, REVIEW_OFFSETS_KEY: review_offsets})
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return len(arrays)


def load_game_review_arrays(path):
    """Return a list of per-review float32 arrays (``n_sent_i x dim``)."""
    with np.load(path) as data:
        vectors = data[VECTORS_KEY]
        offsets = data[REVIEW_OFFSETS_KEY]
    if vectors.size == 0:
        return []
    vectors = vectors.astype(np.float32, copy=False)
    return [
        vectors[int(offsets[i]) : int(offsets[i + 1])]
        for i in range(len(offsets) - 1)
        if int(offsets[i + 1]) > int(offsets[i])
    ]


def load_game_flat(path, dtype, input_dim):
    """Return ``(flat_vectors[dtype], per_review_lengths[int64])`` for the H5 builder.

    Mirrors build_review_h5.load_game_as_arrays' JSON return contract.
    """
    with np.load(path) as data:
        vectors = data[VECTORS_KEY]
        offsets = data[REVIEW_OFFSETS_KEY].astype(np.int64, copy=False)
    if vectors.size == 0:
        raise ValueError(f"{path} contains no vectors.")
    if vectors.shape[1] != input_dim:
        raise ValueError(f"{path}: vector dim {vectors.shape[1]} != expected {input_dim}")
    flat = np.ascontiguousarray(vectors, dtype=dtype)
    lengths = np.diff(offsets)
    # Drop any zero-length reviews so downstream offsets stay strictly increasing.
    lengths = lengths[lengths > 0]
    return flat, lengths.astype(np.int64, copy=False)
