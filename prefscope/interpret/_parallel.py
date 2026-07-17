"""Order-preserving map that runs per-feature work concurrently when asked.

Features are independent, so their LLM calls can fire in parallel against an
OpenRouter (or any OpenAI-compatible) endpoint. concurrency=1 keeps the simple
sequential path; >1 uses a thread pool (LLM calls are I/O-bound, so threads are
the right tool) while preserving input order in the output.

Progress is reported to stderr as work completes (a tqdm bar when available,
otherwise periodic plain-text updates that stay readable in captured logs), so a
long interpret/verify run is no longer silent until it finishes.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def _make_bar(total: int, desc: str):
    """A tqdm bar if tqdm is installed, else a tiny periodic stderr printer.

    Both expose .update(1) and .close(); the fallback prints `desc: done/total`
    roughly every 5% (and on the final item) using newlines, so it doesn't rely
    on a TTY and reads cleanly when captured to a log file."""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, file=sys.stderr)
    except Exception:
        step = max(1, total // 20)

        class _Plain:
            def __init__(self):
                self.n = 0

            def update(self, k: int = 1):
                self.n += k
                if self.n == total or self.n % step == 0:
                    print(f"{desc}: {self.n}/{total}", file=sys.stderr, flush=True)

            def close(self):
                pass

        return _Plain()


def run(fn, items, concurrency: int = 1, desc: str = "working") -> list:
    items = list(items)
    total = len(items)
    bar = _make_bar(total, desc)
    try:
        if concurrency and concurrency > 1 and total > 1:
            results: list = [None] * total
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = {ex.submit(fn, x): i for i, x in enumerate(items)}
                for fut in as_completed(futures):
                    results[futures[fut]] = fut.result()  # keep input order
                    bar.update(1)
            return results
        out = []
        for x in items:
            out.append(fn(x))
            bar.update(1)
        return out
    finally:
        bar.close()
