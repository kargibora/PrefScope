#!/usr/bin/env python
"""Label each named concept with a TYPE — a cheap second interpreter pass.

Adds a coarse ``type`` per feature so the viewer can show what a model *lacks* only
for concepts where absence is meaningful: capability + format (a real deficiency),
while dropping topic (set by the prompt, not the model) and de-emphasising style.

This classifies the CONCEPT STRING only (no activations), so it's one small LLM pass
over feature_names.csv — not a re-interpret. Writes feature_types.csv (feature_id,
concept, type). Surface it by merging into the viewer's feature export.

    OPENROUTER_API_KEY=... python scripts/label_feature_types.py \
        --names "$LENS/feature_names.csv" --out "$LENS/feature_types.csv" \
        --model deepseek/deepseek-v3.2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.interpret.llm import LLMClient      # noqa: E402

TYPES = ["capability", "format", "style", "topic", "safety"]
_SYS = (
    "You classify short descriptions of things an AI answer can do into ONE type:\n"
    "- capability: a substantive thing the answer does to fulfil the request "
    "(provides code, gives a proof, translates, summarises, gives a worked example)\n"
    "- format: how the answer is structured/presented (uses a table, step-by-step, "
    "bullet list, shows its working)\n"
    "- style: tone/voice/stance, optional and not request-fulfilling (narrative framing, "
    "humor, hedging, disclaimers, verbosity)\n"
    "- topic: the subject matter, set by the prompt not the model (cryptocurrency, "
    "video games, biography, geopolitics)\n"
    "- safety: refusal or safety/ethical concern\n"
    "Return JSON {\"labels\":[{\"id\":<int>,\"type\":<one of the five>}, ...]} for every id."
)
_SCHEMA = {
    "type": "object", "required": ["labels"],
    "properties": {"labels": {"type": "array", "items": {
        "type": "object", "required": ["id", "type"],
        "properties": {"id": {"type": "integer"},
                       "type": {"type": "string", "enum": TYPES}}}}},
}


def main() -> None:
    ap = argparse.ArgumentParser(description="label feature concepts with a coarse type")
    ap.add_argument("--names", required=True, help="feature_names.csv (feature_id, concept)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--model", default="deepseek/deepseek-v3.2")
    ap.add_argument("--api-base", default=None, dest="api_base")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY", dest="api_key_env")
    ap.add_argument("--chunk", type=int, default=60)
    args = ap.parse_args()

    names = pd.read_csv(args.names)
    named = names[names["concept"].notna() & (names["concept"].astype(str).str.strip() != "")]
    items = list(zip(named["feature_id"].astype(int), named["concept"].astype(str)))
    print(f"labelling {len(items)} concepts in chunks of {args.chunk}", flush=True)

    kw = {"backend": args.backend, "model": args.model, "api_key_env": args.api_key_env}
    if args.api_base:
        kw["api_base"] = args.api_base
    client = LLMClient(**kw)

    label: dict[int, str] = {}
    for i in range(0, len(items), args.chunk):
        chunk = items[i:i + args.chunk]
        listing = "\n".join(f'{fid}: {c}' for fid, c in chunk)
        raw = client.raw(
            [{"role": "system", "content": _SYS},
             {"role": "user", "content": listing}],
            json_mode=True, response_schema=_SCHEMA, max_tokens=4000)
        try:
            for r in json.loads(raw).get("labels", []):
                t = str(r.get("type", "")).lower()
                if int(r["id"]) in dict(chunk) and t in TYPES:
                    label[int(r["id"])] = t
        except Exception as e:
            print(f"  chunk {i}: parse error ({e}); skipped", file=sys.stderr)
        print(f"  {min(i + args.chunk, len(items))}/{len(items)} done", flush=True)

    out = named[["feature_id", "concept"]].copy()
    out["type"] = out["feature_id"].astype(int).map(label).fillna("uncategorized")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    counts = out["type"].value_counts().to_dict()
    print(f"\nwrote {args.out} — {counts}")


if __name__ == "__main__":
    main()
