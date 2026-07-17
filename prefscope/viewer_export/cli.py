"""The `export_viewer_data` CLI: argparse main() that writes the JSON bundle."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from prefscope.artifacts import FEATURE_CLUSTERS

from .diagnosis import export_diagnosis, export_head_to_head
from .examples import export_examples, export_examples_by_model, export_report_battles
from .features import (export_features, export_meta, feature_fire_rate,
                       feature_prompt_types)
from .maps import export_map, export_prompt_map, export_response_map
from .sanitize import _dumps, _read_csv, _round
from .tables import (export_bias_screen, export_conditional, export_delta,
                     export_elicitation, export_prompt_features)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens-dir", required=True)
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
    ap.add_argument("--out", default="viewer-web/public/data")
    ap.add_argument("--examples-per-feature", type=int, default=12)
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
    ap.add_argument("--map-sample", type=int, default=2500, dest="map_sample",
                    help="battles to subsample for the map scatter")
    ap.add_argument("--map-sample-mode", default="hybrid", dest="map_sample_mode",
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
    a = ap.parse_args()

    lens = Path(a.lens_dir)
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

    features = export_features(lens)
    validation = _read_csv(lens / "diagnosis_validation.csv")

    # feature -> behavior cluster (optional, from `cluster-features`)
    clusters = _read_csv(lens / FEATURE_CLUSTERS)
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
    fr = feature_fire_rate(lens)
    if fr:
        features["generality"] = features["feature_id"].map(lambda f: fr.get(int(f)))
    # n_prompt_types: # prompt concepts that significantly elicit the feature — a secondary
    # context column, from the elicitation table (resolved as for the conditional export).
    elic_for_gen = a.elicitation or (str(lens / "prompt_response_elicitation.csv")
                                     if (lens / "prompt_response_elicitation.csv").exists() else None)
    npt = feature_prompt_types(elic_for_gen)
    if npt:
        features["n_prompt_types"] = features["feature_id"].map(lambda f: npt.get(int(f)))

    (out / "features.json").write_text(_dumps(_round(features)))
    _record("features.json")
    print(f"features.json  ({len(features)} features"
          f"{', ' + str(len(behaviors)) + ' behaviors' if behaviors else ''})")

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
    diag = export_diagnosis(lens, features, prompt_lens=a.prompt_lens,
                            prompt_names=prompt_names_df)
    if diag is not None:
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

    ex = export_examples(lens, a.corpus, features, a.examples_per_feature)
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
            ebm = export_examples_by_model(lens, a.corpus, features, diag,
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
            rb = export_report_battles(lens, a.corpus, a.prompt_lens, diag, prompt_names_df,
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
            h2h = export_head_to_head(lens, features, diag, min_shared=a.h2h_min_shared)
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

    cond_csv = a.conditional or (str(lens / "conditional_win_relevance.csv")
                                 if (lens / "conditional_win_relevance.csv").exists() else None)
    cond_clu_csv = a.conditional_clustered or (
        str(lens / "conditional_win_relevance_clustered.csv")
        if (lens / "conditional_win_relevance_clustered.csv").exists() else None)
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

    elic_csv = a.elicitation or (str(lens / "prompt_response_elicitation.csv")
                                 if (lens / "prompt_response_elicitation.csv").exists() else None)
    ej = export_elicitation(elic_csv)
    if ej is not None:
        (out / "elicitation.json").write_text(_dumps(ej))
        _record("elicitation.json")
        print(f"elicitation.json  ({len(ej['prompt_concepts'])} prompt x "
              f"{len(ej['response_concepts'])} response concepts, "
              f"{ej['n_significant']}/{ej['n_edges']} significant, "
              f"{ej['n_shown']} shown)")

    pf = export_prompt_features(a.prompt_interpret_dir)
    if pf is not None:
        (out / "prompt_features.json").write_text(_dumps(pf))
        _record("prompt_features.json")
        print(f"prompt_features.json  ({len(pf['features'])} prompt concepts)")

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
        "schema_version": 2,
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
