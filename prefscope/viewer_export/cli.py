"""The `export_viewer_data` CLI: argparse main() that writes the JSON bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from prefscope.artifacts import FEATURE_CLUSTERS, Z_DIFF
from prefscope.config import VIEWER_EXPORT_DEFAULTS

from .diagnosis import export_diagnosis, export_head_to_head
from .clusters import export_feature_clusters
from .comparison import export_paired_comparison
from .examples import (export_examples, export_examples_by_model,
                       export_joint_examples, export_prompt_examples,
                       export_report_battles)
from .features import (export_features, export_meta, feature_fire_rate,
                       feature_prompt_types)
from .maps import (export_feature_map, export_map, export_prompt_map,
                   export_response_map)
from .overview import (export_coactivation, export_concept_distribution,
                       export_prompt_coactivation,
                       export_prompt_concept_distribution)
from .sanitize import _dumps, _read_csv, _round
from .tables import (export_bias_screen, export_conditional, export_delta,
                     export_elicitation, export_prompt_features)

BUNDLE_SCHEMA_VERSION = 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens-dir", required=True)
    ap.add_argument(
        "--analysis-dir",
        default=None,
        help="directory containing completion feature names/fidelity/roles/clusters and "
             "other analysis tables; defaults to --lens-dir",
    )
    ap.add_argument("--delta", default=None,
                    help="prompt_conditioned_delta.csv (RAW prompt concepts) -> delta.json")
    ap.add_argument("--delta-clustered", default=None, dest="delta_clustered",
                    help="prompt_conditioned_delta_clustered.csv (prompt CLUSTERS) -> "
                         "delta.json['clustered'] for the Wins-within cluster checkbox")
    ap.add_argument("--bias-screen", default=None, dest="bias_screen",
                    help="bias_screen.csv -> bias_screen.json (confound screen)")
    ap.add_argument("--conditional", default=None,
                    help="conditional_win_relevance.csv (RAW prompt concepts) -> "
                         "conditional.json['raw']. Defaults to "
                         "<lens>/conditional_win_relevance.csv if present.")
    ap.add_argument("--conditional-clustered", default=None, dest="conditional_clustered",
                    help="conditional_win_relevance_clustered.csv (prompt CLUSTERS) -> "
                         "conditional.json['clustered']. Defaults to "
                         "<lens>/conditional_win_relevance_clustered.csv if present.")
    ap.add_argument("--elicitation", default=None,
                    help="prompt_response_elicitation.csv -> elicitation.json (Prompt→Response "
                         "tab). Defaults to <lens>/prompt_response_elicitation.csv if present.")
    ap.add_argument("--prompt-interpret-dir", default=None, dest="prompt_interpret_dir",
                    help="dir with prompt_feature_{names,fidelity,clusters}.csv -> prompt_features.json")
    ap.add_argument("--corpus", default="", help="corpus parquet for example battles")
    ap.add_argument("--out", default="viewer-data")
    ap.add_argument("--examples-per-feature", type=int, default=VIEWER_EXPORT_DEFAULTS["examples_per_feature"])
    ap.add_argument("--examples-per-group", type=int, default=VIEWER_EXPORT_DEFAULTS["examples_per_group"],
                    help="additional strongest examples per available language/source "
                         "for each response feature (default: 2)")
    ap.add_argument("--examples-random", type=int, default=VIEWER_EXPORT_DEFAULTS["examples_random"],
                    help="random active examples per response feature (default: 4)")
    ap.add_argument("--examples-boundary", type=int, default=VIEWER_EXPORT_DEFAULTS["examples_boundary"],
                    help="near-threshold/boundary examples per response feature (default: 4)")
    ap.add_argument("--examples-by-model", action="store_true", dest="examples_by_model",
                    help="emit examples_by_model.json: per (model × feature) the model's OWN "
                         "top answers exhibiting the feature (needs --corpus + individual lens)")
    ap.add_argument("--examples-by-model-per", type=int, default=4,
                    dest="examples_by_model_per",
                    help="answers per (model × feature) in examples_by_model.json")
    ap.add_argument("--report-battles", action="store_true", dest="report_battles",
                    help="emit report_battles.json: per (model × prompt-concept) sample "
                         "battles for the report-card drill-in (needs --corpus + --prompt-lens)")
    ap.add_argument("--report-battles-per-type", type=int, default=5,
                    dest="report_battles_per_type",
                    help="sample battles per (model × prompt concept) for the drill-in")
    ap.add_argument("--joint-examples", action="store_true", dest="joint_examples",
                    help="emit prompt-sharded examples where a selected prompt feature and "
                         "response feature are both strongly active (needs --corpus + "
                         "--prompt-lens + individual z_a; z_b is optional)")
    ap.add_argument("--joint-examples-per-pair", type=int, default=3,
                    dest="joint_examples_per_pair",
                    help="matched examples per prompt-feature × response-feature pair")
    ap.add_argument("--head-to-head", action="store_true", dest="head_to_head",
                    help="emit head_to_head.json: paired prompt-matched feature contrast "
                         "between model pairs, for the report card's 'vs model' mode "
                         "(needs a per-side/individual lens: z_a.npy + z_b.npy)")
    ap.add_argument("--head-to-head-min-shared", type=int, default=30,
                    dest="h2h_min_shared",
                    help="minimum shared battles for a model pair to appear in "
                         "head_to_head.json (the viewer further gates on significance)")
    ap.add_argument("--map", action="store_true",
                    help="also compute the UMAP 2D map (needs umap-learn)")
    ap.add_argument("--feature-map", action="store_true", dest="feature_map",
                    help="emit feature_map.json: one point for every SAE decoder axis; "
                         "uses UMAP when available and deterministic SVD otherwise")
    ap.add_argument("--feature-map-neighbors", type=int, default=30,
                    dest="feature_map_neighbors",
                    help="UMAP neighbor count for --feature-map (default: 30)")
    ap.add_argument("--prompt-feature-map", action="store_true", dest="prompt_feature_map",
                    help="emit prompt_feature_map.json: one point per prompt SAE decoder "
                         "axis (needs --prompt-lens + --prompt-interpret-dir)")
    ap.add_argument("--prompt-examples-per-feature", type=int, default=VIEWER_EXPORT_DEFAULTS["prompt_examples_per_feature"],
                    dest="prompt_examples_per_feature",
                    help="top activating prompts per prompt feature (default: 8)")
    ap.add_argument("--prompt-examples-per-group", type=int, default=VIEWER_EXPORT_DEFAULTS["prompt_examples_per_group"],
                    help="additional strongest prompts per available language/source "
                         "for each prompt feature (default: 2)")
    ap.add_argument("--prompt-examples-random", type=int, default=VIEWER_EXPORT_DEFAULTS["prompt_examples_random"],
                    help="random active prompts per prompt feature (default: 4)")
    ap.add_argument("--prompt-examples-boundary", type=int, default=VIEWER_EXPORT_DEFAULTS["prompt_examples_boundary"],
                    help="near-boundary prompts per prompt feature (default: 4)")
    ap.add_argument("--map-sample", type=int, default=VIEWER_EXPORT_DEFAULTS["map_sample"], dest="map_sample",
                    help="battles to subsample for the map scatter")
    ap.add_argument("--map-sample-mode", default=VIEWER_EXPORT_DEFAULTS["map_sample_mode"], dest="map_sample_mode",
                    choices=["random", "top-activating", "hybrid"],
                    help="which battles to show: random (faithful), top-activating "
                         "(clean clusters), or hybrid (default: half each)")
    ap.add_argument("--prompt-map", action="store_true", dest="prompt_map",
                    help="also emit prompt_map.json (prompt-space UMAP; needs "
                         "--prompt-lens + --completion-lens)")
    ap.add_argument("--prompt-lens", default=None, dest="prompt_lens",
                    help="prompt lens dir (z_prompt.npy) for --prompt-map")
    ap.add_argument("--completion-lens", default=None, dest="completion_lens",
                    help="completion/difference lens dir (z_diff.npy) for --prompt-map")
    ap.add_argument("--response-map", action="store_true", dest="response_map",
                    help="emit response_map.json — feature UMAP at the SINGLE-RESPONSE level "
                         "(individual lens z_a, plus z_b for paired data); a click shows "
                         "one response, not an A/B pair")
    ap.add_argument("--coactivation-top-k", type=int, default=VIEWER_EXPORT_DEFAULTS["coactivation_top_k"],
                    dest="coactivation_top_k",
                    help="strongest co-firing partners retained per concept")
    ap.add_argument("--coactivation-max-pairs", type=int, default=VIEWER_EXPORT_DEFAULTS["coactivation_max_pairs"],
                    dest="coactivation_max_pairs",
                    help="cap on retained co-activation pairs")
    ap.add_argument("--comparison-dir", default=None, dest="comparison_dir",
                    help="paired compare-responses output -> paired_comparison.json")
    a = ap.parse_args(argv)

    lens = Path(a.lens_dir)
    analysis = Path(a.analysis_dir) if a.analysis_dir else lens
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Bundle manifest bookkeeping: every artifact written THIS run is recorded, and
    # processing failures are recorded as errors instead of only a stderr line. The
    # viewer loads bundle_manifest.json first and treats any file NOT listed as absent —
    # so a stale file from an older export can't masquerade as current data (that
    # exact failure shipped once: a sharded-examples viewer over a monolithic bundle).
    written: list[str] = []
    errors: list[dict] = []

    def _record(name: str) -> None:
        written.append(name)

    def _record_error(stage: str, err: Exception | str) -> None:
        errors.append({"stage": stage, "error": str(err)})
        print(f"  (!{stage}: {err})", file=sys.stderr)

    lens_manifest = json.loads((lens / "manifest.json").read_text())
    # Model comparison, head-to-head and diagnosis all need an opponent and a
    # preference label. A single-response SFT lens has neither, so those artifacts are
    # skipped rather than written empty or attempted and failed.
    single_response = (lens_manifest.get("dataset_mode") == "single"
                       or not (lens / Z_DIFF).exists())
    if single_response:
        print("single-response lens: model/preference artifacts will be skipped")

    features = export_features(lens, analysis)
    validation = _read_csv(analysis / "diagnosis_validation.csv")

    # feature -> behavior cluster (optional, from `cluster-features`)
    clusters = _read_csv(analysis / FEATURE_CLUSTERS)
    fid2c, behaviors = {}, {}
    if clusters is not None and "cluster_id" in clusters.columns:
        fid2c = dict(zip(clusters["feature_id"].astype(int), clusters["cluster_id"].astype(int)))
        features = features.merge(clusters[["feature_id", "cluster_id"]], on="feature_id", how="left")
        if "behavior" in clusters.columns:
            behaviors = {int(c): str(b) for c, b in
                         clusters.dropna(subset=["behavior"]).groupby("cluster_id")["behavior"].first().items()}
            features["behavior"] = features["cluster_id"].map(behaviors)

    # generality = pervasiveness: fraction of responses each feature fires in, straight from
    # the completion lens's per-side codes. A behaviour pervades responses; niche content
    # fires rarely. (Topic-based measures can't isolate niche content the prompt lens has no
    # concept for; fire rate is the robust signal.)
    fr = feature_fire_rate(lens, features)
    if fr:
        features["generality"] = features["feature_id"].map(lambda f: fr.get(int(f)))
    # n_prompt_types: # prompt concepts that significantly elicit the feature — a secondary
    # context column, from the elicitation table (resolved as for the conditional export).
    elic_for_gen = a.elicitation or (
        str(analysis / "prompt_response_elicitation.csv")
        if (analysis / "prompt_response_elicitation.csv").exists() else None)
    npt = feature_prompt_types(elic_for_gen)
    if npt:
        features["n_prompt_types"] = features["feature_id"].map(lambda f: npt.get(int(f)))

    (out / "features.json").write_text(_dumps(_round(features)))
    _record("features.json")
    print(f"features.json  ({len(features)} features"
          f"{', ' + str(len(behaviors)) + ' behaviors' if behaviors else ''})")

    response_clusters = export_feature_clusters(
        clusters,
        features,
        kind="response",
        summary=analysis / "feature_clusters_summary.csv",
        diagnostics=analysis / "feature_clusters_diagnostics.csv",
    )
    if response_clusters is not None:
        (out / "feature_clusters.json").write_text(_dumps(response_clusters))
        _record("feature_clusters.json")
        print(
            f"feature_clusters.json  ({response_clusters['n_clusters']} communities, "
            f"{response_clusters['n_clustered_features']} member features)"
        )

    if validation is not None:
        (out / "validation.json").write_text(_dumps(_round(validation)))
        _record("validation.json")
        print(f"validation.json  ({len(validation)} models)")

    meta = export_meta(lens, validation, features)
    (out / "meta.json").write_text(_dumps(meta, indent=2))
    _record("meta.json")
    print(f"meta.json  (EV={meta['ev']}, LOO-R2={meta['loo_r2']})")

    # report-card extras: prompt-type win rates need a prompt lens + its concept names.
    # Reuse the same prompt-lens / interpret-dir args the prompt-map path obtains.
    prompt_names_df = None
    if a.prompt_lens:
        pidir = Path(a.prompt_interpret_dir or a.prompt_lens)
        pnames_csv = pidir / "prompt_feature_names.csv"
        if pnames_csv.exists():
            prompt_names_df = pd.read_csv(pnames_csv)
    diag = None if single_response else export_diagnosis(
        lens, features, prompt_lens=a.prompt_lens, prompt_names=prompt_names_df)
    if single_response:
        print("diagnosis.json  (skipped: single-response dataset has no opponents)")
    elif diag is not None:
        if behaviors:
            diag["clusters"] = [fid2c.get(int(f), -1) for f in diag["features"]]
            diag["behaviors"] = {str(c): b for c, b in behaviors.items()}
        (out / "diagnosis.json").write_text(_dumps(diag))
        _record("diagnosis.json")
        print(f"diagnosis.json  ({len(diag['models'])} models x {len(diag['features'])} features)")
    else:
        # No oriented bank -> the Report card / diagnosis can't be built. Don't leave a
        # stale diagnosis.json silently in place (that's what made an old bundle look
        # current); write an honest stub the viewer can detect, and shout why.
        bankdir = lens / "bank"
        msg = (f"!! NO ORIENTED BANK at {bankdir} -> diagnosis.json was NOT regenerated.\n"
               f"!! The Report card and Model diagnosis need it. Build it first:\n"
               f"!!   prefscope build-bank --lens-dir {lens} "
               f"--from-embeddings <dump> --label human --corpus <corpus> "
               f"--out {bankdir}\n"
               f"!! then re-run this export.")
        print("\n" + msg + "\n", file=sys.stderr)
        (out / "diagnosis.json").write_text(_dumps({
            "error": "no_bank", "message": msg,
            "features": [], "concepts": [], "models": [], "rows": {}}))
        _record("diagnosis.json")
        _record_error("diagnosis", "no oriented bank — stub written")

    dist = export_concept_distribution(lens, features)
    if dist is not None:
        (out / "concept_distribution.json").write_text(_dumps(dist))
        _record("concept_distribution.json")
        print(f"concept_distribution.json  ({dist['n_features']} concepts, "
              f"coverage {dist['coverage']:.1%}, "
              f"{len(dist['dead_features'])} never fire)")

    coact = export_coactivation(lens, features, top_k=a.coactivation_top_k,
                                max_pairs=a.coactivation_max_pairs,
                                corpus_path=a.corpus)
    if coact is not None:
        (out / "coactivation.json").write_text(_dumps(coact))
        _record("coactivation.json")
        print(f"coactivation.json  ({len(coact['pairs'])} pairs"
              f"{', truncated' if coact['truncated'] else ''})")

    ex = export_examples(lens, a.corpus, features, a.examples_per_feature,
                         n_per_group=a.examples_per_group,
                         n_random=a.examples_random, n_boundary=a.examples_boundary)
    if ex is not None:
        # Shard per feature: data/examples/<fid>.json. The viewer fetches only the
        # selected feature's shard (lazy + cached), so covering all named features with
        # many examples each costs ~nothing at startup. A stale monolithic examples.json
        # from older exports is removed so it can't shadow the shards.
        ex_dir = out / "examples"
        ex_dir.mkdir(parents=True, exist_ok=True)
        # clear shards from a previous run — a feature renamed/dropped since then must
        # not keep serving its old shard.
        for old in ex_dir.glob("*.json"):
            old.unlink()
        for fid, rows in ex.items():
            (ex_dir / f"{fid}.json").write_text(_dumps(rows))
        _record("examples/")
        legacy = out / "examples.json"
        if legacy.exists():
            legacy.unlink()
        n_rows = sum(len(r) for r in ex.values())
        print(f"examples/  ({len(ex)} feature shards, {n_rows} rows, ~{a.examples_per_feature}/feature)")

    if a.examples_by_model and diag is not None:
        try:
            ebm = None if single_response else export_examples_by_model(
                lens, a.corpus, features, diag,
                                           n_per=a.examples_by_model_per)
        except Exception as e:  # never abort the bundle over the drill-in
            _record_error("examples_by_model", e)
            ebm = None
        if ebm is None:
            print("  (--examples-by-model needs --corpus + an individual lens with z_a/z_b)",
                  file=sys.stderr)
        else:
            (out / "examples_by_model.json").write_text(_dumps(ebm))
            _record("examples_by_model.json")
            n = sum(len(v) for v in ebm.values())
            print(f"examples_by_model.json  ({len(ebm)} models, {n} model×feature cells)")

    if a.report_battles and diag is not None:
        try:
            rb = None if single_response else export_report_battles(
                lens, a.corpus, a.prompt_lens, diag, prompt_names_df,
                                       per_type=a.report_battles_per_type)
        except Exception as e:  # never abort the bundle over the drill-in
            _record_error("report_battles", e)
            rb = None
        if rb is None:
            print("  (--report-battles needs --corpus with human_pref + --prompt-lens)",
                  file=sys.stderr)
        else:
            (out / "report_battles.json").write_text(_dumps(rb))
            _record("report_battles.json")
            n_cells = sum(len(v) for v in rb.values())
            print(f"report_battles.json  ({len(rb)} models, {n_cells} model×concept cells)")

    if a.head_to_head and diag is not None and diag.get("models"):
        try:
            h2h = None if single_response else export_head_to_head(
                lens, features, diag, min_shared=a.h2h_min_shared)
        except Exception as e:  # never abort the bundle over the head-to-head view
            _record_error("head_to_head", e)
            h2h = None
        if h2h is None:
            print("  (head_to_head not built: needs an individual lens with z_a.npy/z_b.npy "
                  "+ battles.parquet with model_a/model_b, AND row-aligned codes — a "
                  "misaligned z/battles dump is refused rather than exported wrong)",
                  file=sys.stderr)
        else:
            (out / "head_to_head.json").write_text(_dumps(h2h))
            _record("head_to_head.json")
            print(f"head_to_head.json  ({len(h2h['pairs'])} model pairs "
                  f">= {a.h2h_min_shared} shared × {len(h2h['features'])} features)")

    if a.map:
        mp = export_map(lens, a.corpus, features, sample=a.map_sample,
                        mode=a.map_sample_mode)
        if mp is not None:
            if behaviors:
                mp["clusters"] = [fid2c.get(int(f), -1) for f in mp["features"]]
                mp["behaviors"] = {str(c): b for c, b in behaviors.items()}
            (out / "map.json").write_text(_dumps(mp))
            _record("map.json")
            print(f"map.json  ({mp['n_sampled']} of {mp['n_total']} battles)")

    if a.feature_map:
        fm = export_feature_map(
            lens, features, seed=0, n_neighbors=a.feature_map_neighbors)
        if fm is None:
            print("  (--feature-map needs sae_model.pt in --lens-dir)", file=sys.stderr)
        else:
            (out / "feature_map.json").write_text(_dumps(fm))
            _record("feature_map.json")
            print(f"feature_map.json  ({fm['n_total']} features; "
                  f"{fm['projection']} of decoder directions)")

    if a.response_map:
        rm = export_response_map(lens, a.corpus, features, sample=a.map_sample,
                                 mode=a.map_sample_mode)
        if rm is None:
            print("  (--response-map needs an individual lens with z_a.npy/z_b.npy)", file=sys.stderr)
        else:
            if behaviors:
                rm["clusters"] = [fid2c.get(int(f), -1) for f in rm["features"]]
                rm["behaviors"] = {str(c): b for c, b in behaviors.items()}
            (out / "response_map.json").write_text(_dumps(rm))
            _record("response_map.json")
            print(f"response_map.json  ({rm['n_sampled']} of {rm['n_total']} responses)")

    # delta.json wraps two keyspaces: RAW (individual prompt concepts, default) and
    # CLUSTERED (prompt clusters, the Wins-within "group into clusters" checkbox).
    dj_raw = export_delta(a.delta, features, a.bias_screen)
    dj_clu = export_delta(a.delta_clustered, features, a.bias_screen)
    if dj_raw is not None or dj_clu is not None:
        dj = {"raw": dj_raw, "clustered": dj_clu}
        (out / "delta.json").write_text(_dumps(dj))
        _record("delta.json")
        base = dj_raw or dj_clu
        print(f"delta.json  ({len(base['prompt_concepts'])} prompt concepts x "
              f"{len(base['completion_features'])} features, {base['n_significant']} sig cells"
              f"{'; +clustered' if dj_clu is not None else ''})")

    bs = export_bias_screen(a.bias_screen)
    if bs is not None:
        (out / "bias_screen.json").write_text(_dumps(bs))
        _record("bias_screen.json")
        print(f"bias_screen.json  ({len(bs)} features)")

    cond_csv = a.conditional or (
        str(analysis / "conditional_win_relevance.csv")
        if (analysis / "conditional_win_relevance.csv").exists() else None)
    cond_clu_csv = a.conditional_clustered or (
        str(analysis / "conditional_win_relevance_clustered.csv")
        if (analysis / "conditional_win_relevance_clustered.csv").exists() else None)
    cj_raw = export_conditional(cond_csv, features, a.delta)
    cj_clu = export_conditional(cond_clu_csv, features, a.delta_clustered)
    if cj_raw is not None or cj_clu is not None:
        cj = {"raw": cj_raw, "clustered": cj_clu}
        (out / "conditional.json").write_text(_dumps(cj))
        _record("conditional.json")
        base = cj_raw or cj_clu
        print(f"conditional.json  ({len(base['prompt_concepts'])} prompt types x "
              f"{len(base['features'])} features, {base['n_significant']} sig cells"
              f"{'; +clustered' if cj_clu is not None else ''})")

    elic_csv = a.elicitation or (
        str(analysis / "prompt_response_elicitation.csv")
        if (analysis / "prompt_response_elicitation.csv").exists() else None)
    ej = export_elicitation(elic_csv)
    if ej is not None:
        (out / "elicitation.json").write_text(_dumps(ej))
        _record("elicitation.json")
        print(f"elicitation.json  ({len(ej['prompt_concepts'])} prompt x "
              f"{len(ej['response_concepts'])} response concepts, "
              f"{ej['n_significant']}/{ej['n_edges']} significant, "
              f"{ej['n_shown']} shown)")

    if a.joint_examples:
        # Reuse exactly the relationships shipped to the browser. This keeps the evidence
        # payload bounded and guarantees that every selectable elicitation/conditional row
        # can request its corresponding examples.
        joint_pairs = set()
        if ej is not None:
            joint_pairs.update((int(e["px"]), int(e["cy"])) for e in ej["edges"])
        if cj_raw is not None:
            joint_pairs.update((int(c["pc"]), int(c["f"])) for c in cj_raw["cells"])
        try:
            joint = export_joint_examples(
                lens, a.corpus, a.prompt_lens, joint_pairs,
                per_pair=a.joint_examples_per_pair)
        except Exception as e:  # evidence is optional; preserve the rest of the bundle
            _record_error("joint_examples", e)
            joint = None
        if joint is None:
            print("  (--joint-examples needs --corpus + --prompt-lens + an individual "
                  "completion lens with z_a)", file=sys.stderr)
        else:
            joint_dir = out / "joint_examples"
            joint_dir.mkdir(parents=True, exist_ok=True)
            for old in joint_dir.glob("*.json"):
                old.unlink()
            for pc, shard in joint.items():
                (joint_dir / f"{pc}.json").write_text(_dumps(shard))
            _record("joint_examples/")
            n_pairs = sum(len(shard["examples"]) for shard in joint.values())
            n_rows = sum(len(rows) for shard in joint.values()
                         for rows in shard["examples"].values())
            print(f"joint_examples/  ({len(joint)} prompt shards, {n_pairs} pairs, "
                  f"{n_rows} rows)")

    pf = export_prompt_features(a.prompt_interpret_dir)
    if pf is not None:
        (out / "prompt_features.json").write_text(_dumps(pf))
        _record("prompt_features.json")
        print(f"prompt_features.json  ({len(pf['features'])} prompt concepts)")

        prompt_feature_frame = pd.DataFrame(pf["features"])
        prompt_interpret = Path(a.prompt_interpret_dir)
        prompt_cluster_payload = export_feature_clusters(
            prompt_interpret / "prompt_feature_clusters.csv",
            prompt_feature_frame,
            kind="prompt",
            summary=prompt_interpret / "prompt_feature_clusters_summary.csv",
            diagnostics=prompt_interpret / "prompt_feature_clusters_diagnostics.csv",
        )
        if prompt_cluster_payload is not None:
            (out / "prompt_feature_clusters.json").write_text(
                _dumps(prompt_cluster_payload)
            )
            _record("prompt_feature_clusters.json")
            print(
                "prompt_feature_clusters.json  "
                f"({prompt_cluster_payload['n_clusters']} communities, "
                f"{prompt_cluster_payload['n_clustered_features']} member features)"
            )
        if a.prompt_lens:
            prompt_dist = export_prompt_concept_distribution(
                a.prompt_lens, prompt_feature_frame)
            if prompt_dist is not None:
                (out / "prompt_concept_distribution.json").write_text(
                    _dumps(prompt_dist)
                )
                _record("prompt_concept_distribution.json")
                print(
                    "prompt_concept_distribution.json  "
                    f"({prompt_dist['n_features']} concepts, "
                    f"coverage {prompt_dist['coverage']:.1%}, "
                    f"{len(prompt_dist['dead_features'])} never fire)"
                )

            prompt_coact = export_prompt_coactivation(
                a.prompt_lens, prompt_feature_frame,
                top_k=a.coactivation_top_k, max_pairs=a.coactivation_max_pairs,
                corpus_path=a.corpus)
            if prompt_coact is not None:
                (out / "prompt_coactivation.json").write_text(_dumps(prompt_coact))
                _record("prompt_coactivation.json")
                print(f"prompt_coactivation.json  ({len(prompt_coact['pairs'])} pairs)")

            prompt_examples = export_prompt_examples(
                a.prompt_lens, a.corpus, prompt_feature_frame,
                n_per=a.prompt_examples_per_feature,
                n_per_group=a.prompt_examples_per_group,
                n_random=a.prompt_examples_random,
                n_boundary=a.prompt_examples_boundary)
            if prompt_examples is not None:
                prompt_example_dir = out / "prompt_examples"
                prompt_example_dir.mkdir(parents=True, exist_ok=True)
                for old in prompt_example_dir.glob("*.json"):
                    old.unlink()
                for feature_id, rows in prompt_examples.items():
                    (prompt_example_dir / f"{feature_id}.json").write_text(_dumps(rows))
                _record("prompt_examples/")
                n_prompt_rows = sum(len(rows) for rows in prompt_examples.values())
                print(f"prompt_examples/  ({len(prompt_examples)} feature shards, "
                      f"{n_prompt_rows} rows)")

            if a.feature_map or a.prompt_feature_map:
                prompt_fm = export_feature_map(
                    a.prompt_lens, prompt_feature_frame, seed=0,
                    n_neighbors=a.feature_map_neighbors)
                if prompt_fm is not None:
                    (out / "prompt_feature_map.json").write_text(_dumps(prompt_fm))
                    _record("prompt_feature_map.json")
                    print(f"prompt_feature_map.json  ({prompt_fm['n_total']} features; "
                          f"{prompt_fm['projection']} of decoder directions)")
        elif a.prompt_feature_map:
            print("  (--prompt-feature-map needs --prompt-lens)", file=sys.stderr)

    if a.comparison_dir:
        try:
            comparison = export_paired_comparison(a.comparison_dir)
        except Exception as e:
            _record_error("paired_comparison", e)
            comparison = None
        if comparison is not None:
            (out / "paired_comparison.json").write_text(_dumps(comparison))
            _record("paired_comparison.json")
            print(f"paired_comparison.json  ({len(comparison['concepts'])} concept shifts, "
                  f"{len(comparison['contexts'])} context cells, "
                  f"{len(comparison['examples'])} examples)")

    if a.prompt_map:
        if not (a.prompt_lens and a.completion_lens):
            print("  (--prompt-map needs --prompt-lens and --completion-lens)", file=sys.stderr)
        else:
            pm = export_prompt_map(a.prompt_lens, a.completion_lens, a.delta,
                                   a.prompt_interpret_dir or a.prompt_lens,
                                   sample=a.map_sample, mode=a.map_sample_mode,
                                   corpus_path=a.corpus)
            if pm is not None:
                (out / "prompt_map.json").write_text(_dumps(pm))
                _record("prompt_map.json")
                print(f"prompt_map.json  ({pm['n_sampled']} of {pm['n_total']} prompts)")

    # Manifest LAST, so it only ever describes a completed run. The viewer loads this
    # first: files not listed are treated as absent (stale leftovers can't masquerade as
    # current data), version mismatches surface as a banner instead of silent weirdness.
    from datetime import datetime, timezone
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lens": lens.name,
        "files": sorted(set(written)),
        "errors": errors,
    }
    (out / "bundle_manifest.json").write_text(_dumps(manifest, indent=2))
    print(f"bundle_manifest.json  ({len(manifest['files'])} artifacts"
          f"{', ' + str(len(errors)) + ' errors' if errors else ''})")

    print(f"\nwrote bundle to {out}")
    return 0
