"""Regenerate the tracked synthetic corpus used by source-checkout tutorials."""

from pathlib import Path

from prefscope.data.demo import make_demo_corpus


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "sample_corpus.parquet"
    frame = make_demo_corpus()
    frame.to_parquet(out, index=False)
    print(f"wrote {len(frame)} synthetic battles -> {out}")
