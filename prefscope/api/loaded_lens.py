"""The Lens: an SAE encoder + interpreted concept names + manifest, as one object.

A ``Lens`` turns a trained lens directory into a reusable inference artifact, or
trains a fresh one from preference data. Lifecycle: ``train -> save -> load ->
encode -> analyze``.

``load`` (alias ``from_dir``) builds the real (torch) projector + embedder; the
constructor takes them as objects so the orchestration is testable with fakes.
``encode_items(dataset)`` accepts homogeneous paired or single-response data;
``encode_pairs(dataset)`` (alias ``project``) embeds each PairItem's two
responses, forms the self-minus-other contrast the lens was trained on, and
projects it through the SAE to signed codes. ``encode`` projects single
(prompt, completion) responses (individual / prompt lenses only). The analysis
methods delegate to ``prefscope.analysis`` (the format-agnostic cores).

Convention: ``y_a`` is "self" (the model under study), ``y_b`` is "other"; codes
are self-minus-other and ``meta['pref']`` = P(self preferred), matching the
analysis contract.

``LoadedLens`` remains a back-compat alias for ``Lens``.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# absolute submodule import (not ``from prefscope import analysis``) so this
# module survives partial package init — prefscope/__init__ imports this module,
# so a ``from prefscope import analysis`` here would hit a half-initialized package.
import prefscope.analysis as analysis
# module-level so the lens_rep strategies register even in library use (Lens
# otherwise imports nothing from pipeline -> the registry bucket would be empty).
from prefscope.pipeline.lens_rep import get_lens_rep

_REQUIRED_ITEM_COLS = ["prompt", "completion_a", "instruction_id"]


def pairs_to_battles(data, columns=None) -> pd.DataFrame:
    """Normalize preference data into a ``build_lens`` battles DataFrame.

    Accepts (a) a ``Dataset`` / iterable of ``PairItem`` (mapped
    ``x->prompt, y_a->completion_a, y_b->completion_b, id->instruction_id`` plus
    ``pref->human_pref`` / ``model_a`` / ``model_b`` when present), (b) a
    ``pd.DataFrame`` (the ``columns`` rename map is applied first, then required
    columns are validated), or (c) a ``str`` / ``Path`` parquet file (read, then
    treated as a DataFrame). ``completion_b`` is optional for homogeneous
    single-response data. Pure: no embedding, no torch.
    """
    if isinstance(data, (str, Path)):
        data = pd.read_parquet(data)

    if isinstance(data, pd.DataFrame):
        df = data.rename(columns=dict(columns)) if columns else data.copy()
        missing = [c for c in _REQUIRED_ITEM_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"battles missing required columns: {missing}")
        return df.reset_index(drop=True)

    # iterable of PairItem-like objects
    rows = []
    for it in data:
        rows.append({
            "instruction_id": it.id,
            "prompt": it.x,
            "completion_a": it.y_a,
            "completion_b": it.y_b,
            "human_pref": it.pref,
            "model_a": it.model_a,
            "model_b": it.model_b,
        })
    # The columns list above guarantees every required column exists, so no
    # missing-column check is needed here (it can only fire on the DataFrame path).
    return pd.DataFrame(rows, columns=["instruction_id", "prompt", "completion_a",
                                       "completion_b", "human_pref", "model_a",
                                       "model_b"])


class Lens:
    def __init__(self, projector, embedder, *, names=None, manifest=None) -> None:
        """``projector``/``embedder`` are duck-typed (injected so tests can use fakes)."""
        self.projector = projector
        self.embedder = embedder
        self.names = names
        self.manifest = dict(manifest or {})
        if self.manifest:
            # Parse through the versioned manifest so a real artifact's representation is
            # migrated/inferred — NEVER silently defaulted to "difference" (the old
            # `.get("input_rep", "difference")` corrupted every code when a lens omitted
            # it). from_dict raises rather than guess an undeterminable representation.
            from prefscope.core.manifest import LensManifest
            self.manifest_obj = LensManifest.from_dict(self.manifest)
            self.input_rep = self.manifest_obj.input_rep
        else:
            # in-memory Lens with no backing artifact — nothing to be wrong about
            self.manifest_obj = None
            self.input_rep = "difference"
        self.granularity = self.manifest.get("granularity", "response")
        self.lens_dir = None     # set by from_dir/load; None when constructed directly

    @classmethod
    def from_dir(cls, lens_dir, *, device: str = "cpu") -> "Lens":
        """Load a trained lens dir into a ``Lens`` (``device`` is "cpu"/"cuda")."""
        from prefscope.encode.embed import Embedder
        from prefscope.encode.sae import SAEProjector

        lens_dir = Path(lens_dir)
        projector = SAEProjector(lens_dir, device=device)
        names_path = lens_dir / "feature_names.csv"
        names = pd.read_csv(names_path) if names_path.exists() else None
        manifest_path = lens_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        mid = manifest.get("embed_model_id")
        embedder = Embedder(None, device=device, **({"model_id": mid} if mid else {}))
        lens = cls(projector, embedder, names=names, manifest=manifest)
        lens.lens_dir = lens_dir
        return lens

    # public name for from_dir; both work
    load = from_dir

    @classmethod
    def train(cls, data, config=None, *, out, columns=None) -> "Lens":
        """Train + save a fresh lens from preference data, then load it.

        ``data`` is anything ``pairs_to_battles`` accepts. ``config`` is a
        ``TrainConfig`` (defaults if omitted). Trains via ``build_lens`` and
        returns the loaded ``Lens``. Heavy imports (Embedder / build_lens) are
        lazy so ``import prefscope`` stays torch-free.
        """
        from prefscope.api.config import TrainConfig
        from prefscope.encode.embed import Embedder
        from prefscope.pipeline.build_lens import build_lens

        if config is None:
            config = TrainConfig()

        forbidden = {"m_total", "k", "matryoshka_prefix", "input_rep", "val_frac",
                     "device", "embed_model_id", "max_train_rows", "dump_embeddings"}
        overlap = forbidden & set(config.train_kwargs)
        if overlap:
            raise ValueError(
                f"train_kwargs may not override {sorted(overlap)}; set them via "
                f"SAEConfig/TrainConfig fields")

        battles = pairs_to_battles(data, columns=columns)
        embedder = Embedder(
            None, device=config.device,
            **({"model_id": config.embed_model_id} if config.embed_model_id else {}))
        build_lens(
            battles, embedder, out,
            m_total=config.sae.m, k=config.sae.k,
            matryoshka_prefix=config.sae.matryoshka_prefix,
            input_rep=config.sae.input_rep,
            val_frac=config.val_frac, device=config.device,
            embed_model_id=config.embed_model_id,
            max_train_rows=config.max_train_rows,
            **config.train_kwargs)
        return cls.load(out, device=config.device)

    @property
    def fidelity_feature_ids(self):
        if self.names is not None and "fidelity_pass" in self.names.columns:
            return self.names.loc[self.names["fidelity_pass"].astype(bool),
                                  "feature_id"].astype(int).tolist()
        return None

    @property
    def concept_names(self):
        """Series mapping feature_id -> concept name, or None if unnamed.

        De-dups on ``feature_id`` so the index is unique (a duplicated id would
        make ``names.loc[fid]`` a Series and break ``top_concepts``).
        """
        if self.names is not None and "concept" in self.names.columns:
            return (self.names.drop_duplicates("feature_id")
                    .set_index("feature_id")["concept"])
        return None

    def encode(self, prompts, completions=None) -> np.ndarray:
        """Per-response concept codes for (prompt, completion) lists -> (N, M).

        Individual lens: embeds prompt+completion. Prompt lens: embeds the prompt
        alone (completions ignored). A difference lens is contrast-only and
        raises — use ``encode_pairs`` instead. A single ``str`` is accepted for
        either argument and wrapped to length 1 (still returns a 2-D array).

        Returns a bare ``(N, M)`` ndarray (no meta) — unlike ``encode_pairs``,
        which returns ``(codes, meta)``.
        """
        if self.input_rep == "difference":
            raise ValueError(
                "encode() needs an individual/prompt lens; a difference lens is "
                "contrast-only — use encode_pairs(pairs)")
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(completions, str):
            completions = [completions]
        if self.input_rep == "prompt":
            e = self.embedder.encode_prompts(list(prompts))
        else:  # individual
            if completions is None:
                raise ValueError(
                    "individual lens needs completions; pass completion text(s) "
                    "aligned with prompts")
            prompts, completions = list(prompts), list(completions)
            if len(prompts) != len(completions):
                raise ValueError(
                    f"prompts/completions length mismatch: "
                    f"{len(prompts)} vs {len(completions)}")
            e = self.embedder.encode(prompts, completions)
        return self.projector.project(np.asarray(e, dtype=np.float32))

    def encode_one(self, prompt, completion=None) -> np.ndarray:
        """Concept codes for a single response -> (M,)."""
        return self.encode([prompt],
                           [completion] if completion is not None else None)[0]

    def top_concepts(self, codes, k: int = 5):
        """Per row, the k named features with the largest |code|.

        Returns a list (one per row) of ``(concept, signed_value)`` pairs sorted
        by ``|value|`` descending. Unnamed features are skipped; rows shorter than
        k named features return fewer pairs. Empty lists if the lens is unnamed.
        """
        codes = np.atleast_2d(np.asarray(codes, dtype=np.float32))
        if k <= 0:
            return [[] for _ in range(len(codes))]
        names = self.concept_names
        out = []
        for row in codes:
            picks: list[tuple] = []
            if names is not None:
                for fid in np.argsort(-np.abs(row)):
                    fid = int(fid)
                    if np.isnan(row[fid]):
                        continue
                    if fid in names.index:
                        name = names.loc[fid]
                        if pd.notna(name):
                            picks.append((name, float(row[fid])))
                            if len(picks) == k:
                                break
            out.append(picks)
        return out

    def encode_pairs(self, dataset, *, return_meta: bool = True):
        """Dataset -> (codes (N, M) self-minus-other, meta DataFrame).

        Returns ``(codes, meta)`` (not a bare array like ``encode``): pair codes
        need ``pref``/``model_*`` in ``meta`` for diagnosis. ``return_meta=False``
        returns just the codes array.
        """
        if self.granularity == "token":
            raise ValueError(
                "token-granularity lens does not support encode_pairs()/diagnose() in v0")
        items = list(dataset)
        if not items:
            codes = np.empty((0, self.projector.m_total), np.float32)
            return (codes, pd.DataFrame()) if return_meta else codes
        if any(it.is_single for it in items):
            raise ValueError(
                "encode_pairs() requires y_b on every item; use encode_items() "
                "with an individual lens for single-response data")
        prompts = [it.x for it in items]
        e_a = np.asarray(self.embedder.encode(prompts, [it.y_a for it in items]), np.float32)
        e_b = np.asarray(self.embedder.encode(prompts, [it.y_b for it in items]), np.float32)
        codes = get_lens_rep(self.input_rep).contrast_codes(self.projector, e_a, e_b)
        if not return_meta:
            return codes
        meta = pd.DataFrame({
            "id": [it.id for it in items],
            "pref": [it.pref for it in items],
            "model_a": [it.model_a for it in items],
            "model_b": [it.model_b for it in items],
        })
        return codes, meta

    def encode_items(self, dataset, *, return_meta: bool = True):
        """Encode a homogeneous iterable of paired or single-response items.

        Paired input delegates to :meth:`encode_pairs` and returns contrast codes.
        Single-response input is supported by an ``individual`` lens and returns
        absolute per-response codes. Mixing the two modes in one call is rejected so
        one matrix never silently combines quantities with different meanings.
        Preference-based analyses still require paired contrast codes.
        """
        if self.granularity == "token":
            raise ValueError(
                "token-granularity lens does not support encode_items() in v0")
        items = list(dataset)
        meta_cols = ["id", "pref", "model_a", "model_b"]
        if not items:
            codes = np.empty((0, self.projector.m_total), np.float32)
            meta = pd.DataFrame(columns=meta_cols)
            return (codes, meta) if return_meta else codes
        single = np.array([it.is_single for it in items], dtype=bool)
        if bool(single.any()) and not bool(single.all()):
            raise ValueError(
                "encode_items() needs homogeneous data: either every item has y_b "
                "or no item has y_b")
        if not bool(single.all()):
            return self.encode_pairs(items, return_meta=return_meta)
        if self.input_rep != "individual":
            raise ValueError(
                "single-response items need an individual lens; a difference lens "
                "only represents A/B contrasts")
        codes = self.encode([it.x for it in items], [it.y_a for it in items])
        if not return_meta:
            return codes
        meta = pd.DataFrame({
            "id": [it.id for it in items],
            "pref": [it.pref for it in items],
            "model_a": [it.model_a for it in items],
            "model_b": [it.model_b for it in items],
        }, columns=meta_cols)
        return codes, meta

    # back-compat name; encode_pairs is the canonical method
    project = encode_pairs

    def save(self, dest, *, overwrite: bool = False):
        """Copy the backing lens dir to ``dest`` **atomically** (no-op if dest == src).

        The old behaviour merged into ``dest`` (``copytree(dirs_exist_ok=True)``), which
        could leave a *hybrid* lens: new ``sae_model.pt`` beside a stale ``feature_names``
        or ``manifest`` from a previous artifact. Instead we stage into a temp sibling and
        swap it in with an atomic rename, so ``dest`` always contains **exactly one**
        artifact. A non-empty ``dest`` is refused unless ``overwrite=True``.
        """
        if dest is None:
            raise ValueError("save() requires a destination path")
        if self.lens_dir is None:
            raise ValueError("this Lens has no backing directory to save")
        src = Path(self.lens_dir)
        dest = Path(dest)
        if src.resolve() == dest.resolve():
            return dest
        if dest.exists() and any(dest.iterdir()) and not overwrite:
            raise FileExistsError(
                f"{dest} exists and is not empty; pass overwrite=True to replace it "
                "(save() never merges — that would risk a hybrid lens)")
        dest.parent.mkdir(parents=True, exist_ok=True)
        # stage on the SAME filesystem as dest so the final swap is an atomic rename
        staging = dest.parent / f".{dest.name}.tmp-{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        try:
            shutil.copytree(src, staging)          # fresh dir → no dirs_exist_ok merge
            if dest.exists():
                shutil.rmtree(dest)
            os.replace(staging, dest)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        return dest

    def diagnose(self, codes, meta, *, fidelity_only: bool = False):
        """See ``prefscope.analysis.diagnose``."""
        return analysis.diagnose(codes, meta, names=self.names, fidelity_only=fidelity_only)

    def feature_preference_relevance(self, codes, meta):
        """See ``prefscope.analysis.feature_preference_relevance``."""
        return analysis.feature_preference_relevance(codes, meta, names=self.names)

    def evaluate_preference(self, codes, meta, **kwargs):
        """See ``prefscope.analysis.evaluate_preference``."""
        return analysis.evaluate_preference(codes, meta, names=self.names, **kwargs)


# Back-compat alias: the class was formerly named LoadedLens.
LoadedLens = Lens
