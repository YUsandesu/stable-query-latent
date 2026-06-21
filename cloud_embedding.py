"""Embed the split game-review sentences with a remote HuggingFace TEI endpoint.

Reads each JSON list of sentences in ``game_review_cleaned_3_sentences/``, sends
them to the (Nvidia L4, max-concurrency 512) text-embeddings endpoint in batched
concurrent requests, and writes one big JSON list of sentence vectors per file to
``game_review_cleaned_3_sentence_embeddings/`` -- the same layout the local
pipeline produced.

The API token is read from the local ``tokenAPI.txt`` file (gitignored) instead
of being hardcoded.
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent

# Endpoint URL and token both come from the gitignored credentials file at the
# project root (this script's own directory). No URL is hardcoded — change the
# endpoint by editing that file.
# Format (one KEY=VALUE per line, '#' for comments, blank lines allowed):
#     url=https://<your-endpoint>.huggingface.cloud
#     token=hf_xxx...
DEFAULT_TOKEN_FILE = str(SCRIPT_DIR / "tokenAPI.txt")
# When this script is used as a standalone CLI for the game-review flat-vector
# pipeline, point at the data dirs under game_review_data/.
DEFAULT_INPUT_DIR = str(SCRIPT_DIR / "game_review_data" / "game_review_cleaned_3_sentences")
DEFAULT_OUTPUT_DIR = str(SCRIPT_DIR / "game_review_data" / "game_review_cleaned_3_sentence_embeddings")

# The endpoint is configured for an Nvidia L4 with a maximum concurrency of 512,
# so keep at most this many requests in flight at once.
DEFAULT_CONCURRENCY = 512
# Sentences per request. TEI batches server-side; a small batch cuts the number
# of HTTP round-trips while staying well under the per-request token budget.
DEFAULT_BATCH_SIZE = 32


def resolve_script_relative(path):
    path = Path(path)
    return path if path.is_absolute() else SCRIPT_DIR / path


def load_credentials(token_file):
    """Parse the gitignored credentials file. Returns {'url': ..., 'token': ...}.
    Supports KEY=VALUE lines (CRLF or LF), '#' comments, and blank lines."""
    token_path = resolve_script_relative(token_file)
    if not token_path.exists():
        raise FileNotFoundError(
            f"Credentials file not found: {token_path}. Create it with two lines:\n"
            f"  url=https://<your-endpoint>.huggingface.cloud\n"
            f"  token=hf_xxx..."
        )
    creds = {}
    for raw_line in token_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{token_path}: invalid line (expected KEY=VALUE): {raw_line!r}")
        key, _, value = line.partition("=")
        creds[key.strip().lower()] = value.strip()

    missing = [k for k in ("url", "token") if not creds.get(k)]
    if missing:
        raise ValueError(
            f"{token_path}: missing required key(s) {missing}. "
            f"Need both 'url=...' and 'token=...' lines."
        )
    return creds


def load_token(token_file):
    """Backward-compat shim: return only the token string from the creds file."""
    return load_credentials(token_file)["token"]


def load_base_url(token_file):
    """Convenience: return only the endpoint URL from the creds file."""
    return load_credentials(token_file)["url"]


def load_sentences(input_path):
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{input_path} does not contain a JSON list.")
    return data


class EndpointPausedError(RuntimeError):
    """Raised when the inference endpoint is paused / scaled down and won't recover."""


class EmbeddingClient:
    """Thread-safe batched client for a HuggingFace TEI ``/embed`` endpoint."""

    def __init__(self, base_url, token, normalize=False, timeout=120, max_retries=15):
        self.embed_url = base_url.rstrip("/") + "/embed"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self.normalize = normalize
        self.timeout = timeout
        self.max_retries = max_retries
        self._local = threading.local()

    def _session(self):
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def embed_batch(self, sentences):
        payload = {"inputs": sentences, "normalize": self.normalize, "truncate": True}
        backoff = 1.0
        last_error = None
        last_was_paused = False
        for attempt in range(self.max_retries):
            try:
                response = self._session().post(
                    self.embed_url, json=payload, headers=self.headers, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(self._jitter(backoff))
                backoff = min(backoff * 2, 60)
                continue

            if response.status_code == 200:
                return response.json()

            text = response.text
            lowered = text.lower()

            # Transient states to ride out by backing off and retrying:
            #   503                       -> scaling from zero / initializing
            #   429 "Model is overloaded" -> server request queue full while replicas catch up
            #   400 "workload is not stopped"/"initializing"/"starting"/"scaling"/"loading"/"paused"
            #                             -> endpoint is mid-transition. This endpoint scales to
            #      zero and auto-wakes, briefly surfacing a "paused" message during wake-up, so we
            #      retry it too rather than bailing; if it never recovers, the final raise below
            #      reports it as a (possibly manually) paused endpoint.
            paused = response.status_code == 400 and "paused" in lowered
            transient_400 = response.status_code == 400 and any(
                kw in lowered
                for kw in ("not stopped", "initializing", "starting", "scaling", "loading", "paused")
            )
            if response.status_code in (429, 503) or transient_400:
                last_error = RuntimeError(f"{response.status_code}: {text[:200]}")
                last_was_paused = paused
                time.sleep(self._jitter(backoff))
                backoff = min(backoff * 2, 60)
                continue

            raise RuntimeError(f"Embedding request failed [{response.status_code}]: {text[:300]}")

        if last_was_paused:
            raise EndpointPausedError(
                "The inference endpoint stayed paused across all retries. Resume it in the "
                "HuggingFace dashboard, then rerun."
            )
        raise RuntimeError(f"Embedding request failed after {self.max_retries} retries: {last_error}")

    @staticmethod
    def _jitter(backoff):
        import random

        return backoff * (0.5 + random.random())


def chunked(items, size):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def embed_file_streaming(sentences, client, executor, batch_size, out_file, max_in_flight):
    """Embed all sentences of one file and stream the vectors into ``out_file`` as
    a JSON array, in order, as soon as each contiguous batch is ready.

    Memory is bounded to roughly ``max_in_flight`` batches: we never hold the whole
    file's vectors at once, and we keep at most ``max_in_flight`` requests
    outstanding so completed-but-not-yet-written results can't pile up unbounded.
    Returns (vector_count, embedding_dim).
    """
    batches = [batch for _, batch in chunked(sentences, batch_size)]
    n = len(batches)

    in_flight = {}          # future -> batch index
    completed = {}          # batch index -> vectors (only the out-of-order lookahead)
    next_submit = 0
    next_write = 0
    count = 0
    dim = 0
    separator = ""
    out_file.write("[")

    def fill_window():
        nonlocal next_submit
        while next_submit < n and len(in_flight) < max_in_flight:
            future = executor.submit(client.embed_batch, batches[next_submit])
            in_flight[future] = next_submit
            next_submit += 1

    fill_window()
    while in_flight:
        done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
        for future in done:
            completed[in_flight.pop(future)] = future.result()

        # Flush the contiguous completed prefix to disk and drop it from memory.
        while next_write in completed:
            for vector in completed.pop(next_write):
                out_file.write(separator)
                out_file.write(json.dumps(vector))
                separator = ","
                count += 1
                if dim == 0 and vector:
                    dim = len(vector)
            next_write += 1

        fill_window()

    out_file.write("]")
    return count, dim


def write_manifest(output_dir, args, input_files, embedded, skipped, empty,
                   total_vectors, dims, run_started, status, error=None):
    """Write a small, durable status record so 'is it done / what happened?' is
    answerable by reading one file -- no need to watch the process. Written on
    every exit path, including unexpected aborts, so progress is never lost."""
    manifest = {
        "status": status,
        "error": error,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(time.time() - run_started, 1),
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "batch_size": args.batch_size,
        "normalize": args.normalize,
        "input_files": len(input_files),
        "embedded_this_run": embedded,
        "skipped_existing": skipped,
        "empty": empty,
        "output_files_total": len(list(output_dir.glob("*.json"))),
        "total_vectors_this_run": total_vectors,
        "embedding_dims_seen": sorted(dims),
    }
    try:
        (SCRIPT_DIR / "cloud_embedding_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        print(f"Warning: could not write manifest: {exc}", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--base-url", default=None,
                        help="Override endpoint URL. Default: read 'url=' from --token-file.")
    parser.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--concurrency", default=DEFAULT_CONCURRENCY, type=int)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument(
        "--max-in-flight",
        default=None,
        type=int,
        help=(
            "Max requests outstanding at once. Bounds memory of the streaming "
            "writer (completed-but-unwritten batches). Defaults to concurrency."
        ),
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Ask the endpoint to L2-normalize embeddings.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-embed files even if an output JSON already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_in_flight is None:
        args.max_in_flight = args.concurrency
    input_dir = resolve_script_relative(args.input_dir).resolve()
    output_dir = resolve_script_relative(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.json"))
    if not input_files:
        raise ValueError(f"No JSON files were found in {input_dir}.")

    creds = load_credentials(args.token_file)
    # Resolve effective URL once and store back on args so write_manifest (which
    # records the run's configuration) reports the URL actually used, not None.
    args.base_url = args.base_url or creds["url"]
    client = EmbeddingClient(args.base_url, creds["token"], normalize=args.normalize)

    print(
        f"Embedding {len(input_files)} files via {args.base_url} "
        f"(concurrency={args.concurrency}, batch={args.batch_size})"
    )

    run_started = time.time()
    embedded = skipped = empty = 0
    total_vectors = 0
    dims = set()

    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        for file_index, input_path in enumerate(input_files, start=1):
            output_path = output_dir / input_path.name
            if output_path.exists() and not args.overwrite:
                skipped += 1
                print(f"[{file_index}/{len(input_files)}] Skipping {input_path.name} (already embedded)")
                continue

            sentences = load_sentences(input_path)
            if not sentences:
                empty += 1
                output_path.write_text("[]", encoding="utf-8")
                print(f"[{file_index}/{len(input_files)}] {input_path.name}: empty, wrote []")
                continue

            started = time.time()

            # Stream vectors straight into the tmp file as batches complete (in
            # order) so we never hold a whole file's embeddings in memory and the
            # disk write is spread out instead of one big flush at the end. Only on
            # success do we atomically rename to the final name, so a crash leaves
            # a partial .tmp (ignored by the resume check), never a bogus output.
            tmp_path = output_path.with_suffix(".json.tmp")
            try:
                with tmp_path.open("w", encoding="utf-8") as file:
                    vec_count, dim = embed_file_streaming(
                        sentences, client, executor, args.batch_size, file, args.max_in_flight
                    )
                tmp_path.replace(output_path)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise
            elapsed = time.time() - started

            embedded += 1
            total_vectors += vec_count
            dims.add(dim)
            print(
                f"[{file_index}/{len(input_files)}] {input_path.name}: "
                f"{len(sentences)} sentences -> {vec_count} vectors (dim {dim}) "
                f"in {elapsed:.1f}s"
            )
    except EndpointPausedError as exc:
        write_manifest(output_dir, args, input_files, embedded, skipped, empty,
                       total_vectors, dims, run_started, status="endpoint_paused", error=str(exc))
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        write_manifest(output_dir, args, input_files, embedded, skipped, empty,
                       total_vectors, dims, run_started, status="interrupted")
        print(f"\nInterrupted after {embedded} file(s). Progress saved; rerun to resume.", file=sys.stderr)
        sys.exit(130)
    except BaseException as exc:
        # Catch-all so an unexpected abort still records what was done before dying.
        write_manifest(output_dir, args, input_files, embedded, skipped, empty,
                       total_vectors, dims, run_started, status="error",
                       error=f"{type(exc).__name__}: {exc}")
        print(f"\nERROR after {embedded} file(s): {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    finally:
        executor.shutdown(wait=True)

    write_manifest(output_dir, args, input_files, embedded, skipped, empty,
                   total_vectors, dims, run_started, status="done")
    print(f"Done. Sentence-embedding JSON files written to {output_dir}")


if __name__ == "__main__":
    main()
