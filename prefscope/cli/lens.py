"""CLI handler for reusable lens packaging."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

def _cmd_package_bundle(args) -> int:
    from prefscope.api._bundle_publication import package_bundle

    members = {}
    for value in args.member:
        if "=" not in value:
            raise ValueError("--member must use NAME=DIR")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError("--member must use NAME=DIR")
        if name in members:
            raise ValueError(f"duplicate bundle member: {name}")
        members[name] = path
    out = package_bundle(
        members, args.out, readme=args.readme, overwrite=args.overwrite,
    )
    print(f"wrote lens bundle to {out}")
    return 0

def _cmd_package_lens(args) -> int:
    from prefscope.api.loaded_lens import Lens
    from prefscope.core.manifest import LensManifest

    # Corpus-aligned z arrays are intentionally absent from many existing inference
    # bundles; checkpoint/manifest integrity is still validated by the loader and the
    # newly packaged artifact declares no such arrays.
    lens = Lens.load(
        args.lens_dir,
        device=args.device,
        validate_arrays=False,
        derived_catalog=args.derived_catalog,
    )
    out = lens.save(
        args.out,
        overwrite=args.overwrite,
        annotations=args.annotations,
        derived_catalog=args.derived_catalog,
        inference_only=True,
        derived_view=args.derived_view,
        derived_transform=args.derived_transform,
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
