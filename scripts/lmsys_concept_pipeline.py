#!/usr/bin/env python
"""End-to-end concept pipeline on LMSYS arena data — prompts AND completions.

NOTE: standalone legacy recipe (hard-coded constants, not wired into the CLI). It
builds a two-lens (prompt + response) concept view from the framework primitives.
The current, maintained prompt→response analysis is
``prefscope elicit`` (co-activation lift). Kept as a reference
example of the full two-lens pipeline; not part of the core framework story.

Hard-coded constants (no argparser): edit the CONFIG block below. Drives the
PrefScope tools, no reimplementation:

  (a) load LMSYS/arena prompts + completions          prefscope.data.arenas.load_arena
  (b) embed + train two single-text concept SAEs:      build_lens / build_prompt_lens
      a prompt-concept lens, and a response-concept
      lens — ONE SAE over the pooled individual
      responses (e_a and e_b merged; no A/B contrast)
  (c) interpret every feature from the texts that      name_prompt_features
      activate it, then verify each concept on held-out  verify_single_text_features
      texts ("does this text exhibit the concept?")
  (d) cluster features by mutual-information + Leiden    cluster_features / summarize_clusters
      and LLM-name each cluster -> a hierarchical view   / name_clusters
      (cluster behavior -> member feature concepts),
      adapted from *Anatomy of Post-Training* (App. B)
  (e) save all artifacts under OUT/

Run on a box with the embedding model (GPU) and an OpenRouter key:
    OPENROUTER_API_KEY=... python scripts/lmsys_concept_pipeline.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

# scripts/ runs as a loose file (the package isn't installed); add the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefscope.config import CONFIG  # noqa: E402
from prefscope.data.arenas import load_arena  # noqa: E402
from prefscope.encode.cache import NpyCache  # noqa: E402
from prefscope.encode.embed import Embedder  # noqa: E402
from prefscope.interpret.llm import LLMClient  # noqa: E402
from prefscope.interpret.prompt_name import name_prompt_features  # noqa: E402
from prefscope.interpret.verify import verify_single_text_features  # noqa: E402
from prefscope.pipeline.build_lens import build_lens, build_prompt_lens  # noqa: E402
from prefscope.pipeline.cluster import (  # noqa: E402
    cluster_features, name_clusters, summarize_clusters,
)

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# CONFIG — hard-fix the model / dataset / hyperparameters here.
# ----------------------------------------------------------------------------
SOURCE = "lmarena-100k"          # LMSYS/lmarena arena dump (see arenas.SOURCES)
SPLIT = "train"
LIMIT = None                     # set an int for a quick trial; None = full split

EMBED_MODEL = CONFIG.embed_model_id   # Qwen3-Embedding (the project default)
DEVICE = "cuda"

PROMPT_M, PROMPT_K = 64, 8        # prompt SAE: dictionary size / active-per-row
COMP_M, COMP_K = 128, 16         # completion SAE (trained on completions individually)
CLUSTER_METHOD = "mi-leiden"     # MI graph + Leiden (Anatomy of Post-Training); needs igraph + leidenalg
CLUSTER_RESOLUTION = 1.0         # Leiden resolution: higher -> more, smaller clusters
N_CLUSTERS = 12                  # used only by the spherical-kmeans / agglomerative fallbacks

INTERPRETER_MODEL = "deepseek/deepseek-v3.2"   # OpenRouter; needs OPENROUTER_API_KEY
INTERPRET_CONCURRENCY = 8        # parallel LLM calls during naming/verification

OUT = Path("results/lmsys_concepts")
CACHE_DIR = OUT / "embed_cache"  # per-completion embedding cache (resumable)


def _interpret_and_cluster(tag, lens_dir, z, names, client, *, fidelity=None):
    """Cluster a lens's features by co-activation and LLM-name each cluster.

    Produces the hierarchical view: clusters.csv (feature -> cluster) and
    cluster_summary.csv (one row per behavior: LLM label + member concepts).
    ``names`` carries the per-feature concepts (item-level explanations);
    ``fidelity`` (optional) adds a ``correlation`` column so the representative
    member is the most faithful one.
    """
    log.info(f"[{tag}] clustering {z.shape[1]} features by co-activation ({CLUSTER_METHOD})")
    clusters = cluster_features(z, n_clusters=N_CLUSTERS, method=CLUSTER_METHOD,
                                resolution=CLUSTER_RESOLUTION)
    clusters.to_csv(lens_dir / "clusters.csv", index=False)

    # merge fidelity (correlation / fidelity_pass) into the names table so the
    # cluster representative + verified counts are populated when available.
    named = names
    if fidelity is not None:
        named = names.merge(
            fidelity[[c for c in ("feature_id", "correlation", "fidelity_pass")
                      if c in fidelity.columns]],
            on="feature_id", how="left")

    summary = summarize_clusters(clusters, named)
    log.info(f"[{tag}] LLM-naming {len(summary)} clusters")
    labels = name_clusters(summary, client, concurrency=INTERPRET_CONCURRENCY)
    summary["behavior_llm"] = summary["cluster_id"].map(labels)
    summary.to_csv(lens_dir / "cluster_summary.csv", index=False)
    return clusters, summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, datefmt="%H:%M:%S",
        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    comp_dir = OUT / "response_lens"
    prompt_dir = OUT / "prompt_lens"
    prompt_emb_dir = OUT / "prompt_embeddings"

    # ---- (a) load LMSYS / arena prompts + completions ----------------------
    log.info(f"[load] {SOURCE} (split={SPLIT}, limit={LIMIT})")
    battles = load_arena(SOURCE, split=SPLIT, limit=LIMIT, keep_labels=True)
    # build_lens requires an `instruction_id`; the corpus keys on `battle_id`.
    battles = battles.reset_index(drop=True)
    battles["instruction_id"] = battles["battle_id"].astype(str)
    log.info(f"[load] {len(battles)} battles, {battles['prompt'].nunique()} unique prompts")

    embedder = Embedder(NpyCache(CACHE_DIR), model_id=EMBED_MODEL, device=DEVICE)

    # ---- (b1) response-concept SAE — one SAE over pooled individual responses
    # input_rep="individual" embeds each (prompt, response) and pools e_a + e_b
    # into a (2N, D) matrix, then trains ONE SAE over all responses (no A/B
    # contrast). It writes the per-response codes z_a and z_b.
    log.info("[build] response-concept lens (pooled individual responses)")
    build_lens(battles, embedder, comp_dir,
               m_total=COMP_M, k=COMP_K, input_rep="individual",
               device=DEVICE, embed_model_id=EMBED_MODEL)

    # ---- (b2) prompt SAE — trained on prompt-only embeddings ---------------
    prompt_df = battles.drop_duplicates("prompt").reset_index(drop=True)
    prompts = prompt_df["prompt"].tolist()
    log.info(f"[build] embedding {len(prompts)} unique prompts")
    e_prompt = embedder.encode_prompts(prompts)          # reloads the embed model
    if hasattr(embedder, "unload"):
        embedder.unload()                                # free GPU before SAE training
    prompt_emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(prompt_emb_dir / "e_prompt.npy", np.asarray(e_prompt, dtype=np.float32))
    prompt_df[["instruction_id", "prompt"]].to_parquet(prompt_emb_dir / "meta.parquet")
    log.info("[build] prompt lens")
    build_prompt_lens(prompt_emb_dir, prompt_dir,
                      m_total=PROMPT_M, k=PROMPT_K,
                      device=DEVICE, embed_model_id=EMBED_MODEL)

    # ---- (c) interpret + verify each lens (single-text) --------------------
    # Both lenses are single-text concept SAEs: each feature = a concept that
    # appears in the texts (prompts for one lens, responses for the other). Each
    # feature is named from the texts that most activate it, then verified on
    # held-out texts ("does this text exhibit the concept?" -> Yes/No), correlated
    # with the SAE's active/silent split.
    # NOTE: name_prompt_features' template is phrased for prompts ("what they ask
    # for"); reused for responses here — a response-specific template would
    # sharpen the wording but the mechanism (top-activating texts) is the same.
    client = LLMClient(model=INTERPRETER_MODEL)

    # response-concept lens: codes for the pooled individual responses (z_a, z_b).
    z_a = np.load(comp_dir / "z_a.npy")
    z_b = np.load(comp_dir / "z_b.npy")
    z_resp = np.vstack([z_a, z_b]).astype(np.float32)
    responses = battles["completion_a"].tolist() + battles["completion_b"].tolist()
    resp_ids = ([f"{i}_a" for i in battles["instruction_id"]]
                + [f"{i}_b" for i in battles["instruction_id"]])
    log.info(f"[interpret] naming {z_resp.shape[1]} response-concept features")
    resp_names = name_prompt_features(
        responses, z_resp, client, instruction_ids=resp_ids,
        concurrency=INTERPRET_CONCURRENCY)
    resp_names.to_csv(comp_dir / "feature_names.csv", index=False)
    log.info("[verify] verifying response-concept features (single-text)")
    resp_fidelity = verify_single_text_features(
        responses, z_resp, resp_names, client, instruction_ids=resp_ids,
        concurrency=INTERPRET_CONCURRENCY)
    resp_fidelity.to_csv(comp_dir / "feature_fidelity.csv", index=False)

    # prompt-concept lens.
    z_prompt = np.load(prompt_dir / "z_prompt.npy")
    log.info(f"[interpret] naming {z_prompt.shape[1]} prompt-concept features")
    prompt_names = name_prompt_features(
        prompts, z_prompt, client,
        instruction_ids=prompt_df["instruction_id"].tolist(),
        concurrency=INTERPRET_CONCURRENCY)
    prompt_names.to_csv(prompt_dir / "feature_names.csv", index=False)
    log.info("[verify] verifying prompt-concept features (single-text)")
    prompt_fidelity = verify_single_text_features(
        prompts, z_prompt, prompt_names, client,
        instruction_ids=prompt_df["instruction_id"].tolist(),
        concurrency=INTERPRET_CONCURRENCY)
    prompt_fidelity.to_csv(prompt_dir / "feature_fidelity.csv", index=False)

    # ---- (d) cluster by co-activation + LLM-name each cluster --------------
    _, resp_summary = _interpret_and_cluster(
        "response", comp_dir, z_resp, resp_names, client, fidelity=resp_fidelity)
    _, prompt_summary = _interpret_and_cluster(
        "prompt", prompt_dir, z_prompt, prompt_names, client, fidelity=prompt_fidelity)

    # ---- (e) save a combined hierarchical index ----------------------------
    index = {
        "source": SOURCE, "n_battles": int(len(battles)),
        "embed_model": EMBED_MODEL, "interpreter_model": INTERPRETER_MODEL,
        "response_lens": {
            "dir": str(comp_dir), "m_total": COMP_M, "k": COMP_K,
            "n_features": int(z_resp.shape[1]), "n_responses": int(z_resp.shape[0]),
            "n_clusters": int(resp_summary["cluster_id"].nunique()),
            "n_fidelity_pass": int(resp_fidelity["fidelity_pass"].sum()),
        },
        "prompt_lens": {
            "dir": str(prompt_dir), "m_total": PROMPT_M, "k": PROMPT_K,
            "n_features": int(z_prompt.shape[1]),
            "n_clusters": int(prompt_summary["cluster_id"].nunique()),
            "n_fidelity_pass": int(prompt_fidelity["fidelity_pass"].sum()),
        },
        "artifacts": {
            "per_feature_concepts": "feature_names.csv",
            "feature_fidelity": "feature_fidelity.csv  (single-text verification)",
            "feature_to_cluster": "clusters.csv",
            "cluster_behaviors": "cluster_summary.csv  (behavior_llm + member_concepts)",
        },
    }
    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    log.info(f"\n[done] results under {OUT}/  "
          f"(response_lens/, prompt_lens/, index.json)")


if __name__ == "__main__":
    main()
