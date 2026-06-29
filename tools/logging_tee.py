"""File-descriptor level stdout/stderr tee for long notebook-launched jobs."""

from __future__ import annotations

import os
import shlex
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from pathlib import Path


class StdoutStderrTee:
    """Mirror fd 1/2 to a log file while keeping live console output.

    Redirecting the file descriptors, rather than only replacing sys.stdout,
    lets child processes inherited by subprocess.run write into the same log.
    """

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path).expanduser()
        self._saved_stdout: int | None = None
        self._saved_stderr: int | None = None
        self._pipe_read: int | None = None
        self._thread: threading.Thread | None = None
        self._log_file = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._saved_stdout = os.dup(1)
        self._saved_stderr = os.dup(2)
        self._pipe_read, pipe_write = os.pipe()
        self._log_file = self.log_path.open("ab", buffering=0)

        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

        os.dup2(pipe_write, 1)
        os.dup2(pipe_write, 2)
        os.close(pipe_write)

        cmd = " ".join(shlex.quote(part) for part in sys.argv)
        header = (
            f"\n\n=== log start {time.strftime('%Y-%m-%d %H:%M:%S %z')} ===\n"
            f"cwd: {Path.cwd()}\n"
            f"cmd: {cmd}\n"
            f"log: {self.log_path.resolve()}\n\n"
        )
        os.write(1, header.encode("utf-8", errors="replace"))
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        if self._saved_stdout is not None:
            os.dup2(self._saved_stdout, 1)
        if self._saved_stderr is not None:
            os.dup2(self._saved_stderr, 2)

        if self._thread is not None:
            self._thread.join(timeout=5)

        for fd in (self._saved_stdout, self._saved_stderr):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if self._log_file is not None:
            self._log_file.close()
        return False

    def _pump(self) -> None:
        assert self._pipe_read is not None
        assert self._saved_stdout is not None
        assert self._log_file is not None
        with os.fdopen(self._pipe_read, "rb", buffering=0) as reader:
            while True:
                chunk = reader.read(65536)
                if not chunk:
                    break
                try:
                    os.write(self._saved_stdout, chunk)
                except OSError:
                    pass
                self._log_file.write(chunk)


def maybe_tee(logout_address: str | Path | None):
    if logout_address is None or str(logout_address).strip() == "":
        return nullcontext()
    return StdoutStderrTee(logout_address)


def run_with_optional_tee(logout_address: str | Path | None, func, *args, **kwargs):
    with maybe_tee(logout_address):
        try:
            return func(*args, **kwargs)
        except SystemExit as exc:
            if exc.code not in (None, 0):
                if isinstance(exc.code, int):
                    print(f"SystemExit: {exc.code}", file=sys.stderr, flush=True)
                    raise
                print(exc.code, file=sys.stderr, flush=True)
                raise SystemExit(1)
            raise
        except KeyboardInterrupt:
            print("KeyboardInterrupt", file=sys.stderr, flush=True)
            raise
        except BaseException:
            traceback.print_exc()
            raise
