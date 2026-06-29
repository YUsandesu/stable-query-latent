from types import SimpleNamespace

import numpy as np

from VICReg_review import train_vicreg_review_h5 as trainer


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _RecordingExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, fn, *args):
        self.calls.append((fn, args))
        return _ImmediateFuture([])


def test_full_cache_prefetch_keeps_view_cap_and_train_indices_separate():
    args = SimpleNamespace(
        cache_mode="full",
        epochs=2,
        input_h5="embedding_h5.h5",
        batch_size=128,
        steps_per_epoch=4,
        sample_fraction=0.8,
        seed=42,
        pin_cache=True,
        game_order="random",
        max_batch_sentences=0,
        max_view_sentences=4096,
        train_game_indices=np.asarray([1, 2, 3], dtype=np.int64),
    )
    executor = _RecordingExecutor()

    _batches, _future = trainer.iter_epoch(args, 1, _ImmediateFuture([]), executor, np.dtype("float16"))

    assert len(executor.calls) == 1
    fn, submitted = executor.calls[0]
    assert fn is trainer.prepare_epoch_batches
    assert submitted[10] == args.max_view_sentences
    np.testing.assert_array_equal(submitted[11], args.train_game_indices)
