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
                self.postfix = ""

            def set_postfix_str(self, text: str, refresh: bool = False):
                self.postfix = text

            def update(self, k: int = 1):
                self.n += k
                if self.n == total or self.n % step == 0:
                    suffix = f" | {self.postfix}" if self.postfix else ""
                    print(f"{desc}: {self.n}/{total}{suffix}", file=sys.stderr, flush=True)

            def close(self):
                pass

        return _Plain()


def run(fn, items, concurrency: int = 1, desc: str = "working", usage=None,
        on_result=None) -> list:
    """Apply ``fn`` and optionally checkpoint each successful result immediately.

    ``on_result`` runs in the caller thread, including in concurrent mode.  Long LLM
    stages use it to durably save a feature as soon as it completes, while the returned
    list remains ordered like ``items`` for backward compatibility.
    """
    items = list(items)
    total = len(items)
    bar = _make_bar(total, desc)

    def advance() -> None:
        progress = getattr(usage, "usage_progress", None)
        if callable(progress) and hasattr(bar, "set_postfix_str"):
            bar.set_postfix_str(progress(), refresh=False)
        bar.update(1)

    try:
        if concurrency and concurrency > 1 and total > 1:
            results: list = [None] * total
            ex = ThreadPoolExecutor(max_workers=concurrency)
            try:
                futures = {ex.submit(fn, x): i for i, x in enumerate(items)}
                first_error = None
                for fut in as_completed(futures):
                    # Once one feature fails we cancel work that has not started. A
                    # canceled future is not completed work and must not advance the bar;
                    # otherwise an interrupted 2/215 run misleadingly prints 215/215.
                    if fut.cancelled():
                        continue
                    try:
                        result = fut.result()
                    except BaseException as exc:  # preserve other completed checkpoints
                        if first_error is None:
                            first_error = exc
                            # Do not turn one broken credential/configuration into thousands
                            # of doomed API calls. Already-running work may still checkpoint.
                            for pending in futures:
                                pending.cancel()
                    else:
                        results[futures[fut]] = result  # keep input order
                        if on_result is not None:
                            on_result(result)
                    advance()
                if first_error is not None:
                    raise first_error
            except BaseException:
                ex.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                ex.shutdown(wait=True)
            return results
        out = []
        for x in items:
            result = fn(x)
            out.append(result)
            if on_result is not None:
                on_result(result)
            advance()
        return out
    finally:
        bar.close()
