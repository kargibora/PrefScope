"""CLI handler for reusable lens packaging."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

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


__all__ = ["_cmd_package_lens"]
