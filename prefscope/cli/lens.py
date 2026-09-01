"""CLI handlers for reusable lens packaging and single-text inference."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _display_concepts(title: str, rows: list[dict], *, policy: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")
    if not rows:
        print("No named concepts passed the requested filters.")
        if policy == "calibrated":
            print(
                "The lens may lack passing semantic calibration; use --presence-policy "
                "mixed only for explicitly exploratory activations."
            )
        return
    display = pd.DataFrame(rows)
    display.insert(0, "rank", np.arange(1, len(display) + 1))
    columns = [
        "rank",
        "feature_id",
        "activation",
        "presence_basis",
        "concept",
        *[
            name
            for name in ("correlation", "agreement", "semantic_role")
            if name in display.columns
        ],
    ]
    print(display[columns].to_string(index=False, max_colwidth=72))
    if display["presence_basis"].eq("positive_nonzero").any():
        print(
            "\nNote: positive_nonzero is exploratory SAE activity, not a calibrated "
            "semantic-presence claim."
        )


def _cmd_extract_concepts(args) -> int:
    from prefscope.pipeline.text_concepts import extract_text_concepts

    top = None if args.top == 0 else args.top
    result = extract_text_concepts(
        args.prompt,
        args.completion,
        repo_id=args.repo,
        prompt_lens=args.prompt_lens,
        completion_lens=args.completion_lens,
        prompt_subfolder=args.prompt_subfolder,
        completion_subfolder=args.completion_subfolder,
        revision=args.revision,
        device=args.device,
        presence_policy=args.presence_policy,
        fidelity_only=not args.include_unverified,
        top=top,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if "prompt" in result:
            _display_concepts(
                "Prompt concepts", result["prompt"], policy=args.presence_policy
            )
        if "completion" in result:
            _display_concepts(
                "Completion concepts", result["completion"], policy=args.presence_policy
            )
    return 0


def _cmd_package_lens(args) -> int:
    from prefscope.api.loaded_lens import Lens
    from prefscope.core.manifest import LensManifest

    # Corpus-aligned z arrays are intentionally absent from many existing inference
    # bundles; checkpoint/manifest integrity is still validated by the loader and the
    # newly packaged artifact declares no such arrays.
    lens = Lens.load(args.lens_dir, device=args.device, validate_arrays=False)
    out = lens.save(
        args.out,
        overwrite=args.overwrite,
        annotations=args.annotations,
        inference_only=True,
    )
    if args.model_card:
        shutil.copy2(args.model_card, Path(out) / "README.md")
    manifest_path = Path(out) / "manifest.json"
    manifest = LensManifest.from_dict(
        json.loads(manifest_path.read_text()), strict=True
    )
    manifest.validate_arrays(out)
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))
    print(f"wrote validated inference-only lens to {out}")
    return 0


__all__ = ["_cmd_extract_concepts", "_cmd_package_lens"]
