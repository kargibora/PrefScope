"""PrefScope viewer — explore SAE difference-axes, their meanings, and what a
target model over/under-expresses.

Run:
    uv run --extra viewer streamlit run prefscope/viewer/app.py -- \
        --lens-dir lens_diff_m32_k4 \
        --annotations /path/to/agreement_annotations.json

Paths can also be set/overridden in the sidebar. The annotation file supplies
the completion text (the lens stores only metadata + activations).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit runs this file as a script, and the package isn't installed
# (uv package=false), so add the project root to sys.path before importing prefscope.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prefscope.viewer.data import (  # noqa: E402
    diagnosis_battles, feature_table, load_lens_for_view, top_battles,
)


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--lens-dir", default="")
    p.add_argument("--annotations", nargs="*", default=[])
    p.add_argument("--corpus", default="")
    p.add_argument("--names", default="")
    p.add_argument("--fidelity", default="")
    p.add_argument("--diagnosis", default="")
    p.add_argument("--diagnosis-battles", default="", dest="diagnosis_battles")
    p.add_argument("--win-relevance", default="", dest="win_relevance")
    p.add_argument("--validation", default="")
    p.add_argument("--bank", default="")
    # streamlit passes script args after a `--`
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    known, _ = p.parse_known_args(argv)
    return known


@st.cache_data(show_spinner="Loading lens…")
def _load(lens_dir: str, annotations: tuple[str, ...], corpus: str):
    return load_lens_for_view(lens_dir, list(annotations) or None, corpus or None)


@st.cache_data(show_spinner="Loading oriented-code bank…")
def _load_bank(bank_dir: str):
    from prefscope.pipeline.oriented_bank import load_bank
    Z, meta, manifest = load_bank(bank_dir)
    return Z, meta, manifest


def _read_csv(path: str):
    return pd.read_csv(path) if path and Path(path).exists() else None


def _read_parquet(path: str):
    return pd.read_parquet(path) if path and Path(path).exists() else None


def _default_csv(lens_dir: str, override: str, fname: str) -> str:
    if override:
        return override
    cand = Path(lens_dir) / fname if lens_dir else None
    return str(cand) if cand and cand.exists() else ""


def main() -> None:
    st.set_page_config(page_title="PrefScope viewer", layout="wide")
    a = _args()

    st.sidebar.header("Lens")
    lens_dir = st.sidebar.text_input("Lens directory", a.lens_dir)
    ann_default = "\n".join(a.annotations)
    corpus_path = st.sidebar.text_input(
        "corpus parquet — optional", a.corpus,
        help="Merged-corpus parquet the lens was built from; supplies the A/B "
             "completion text for corpus-built lenses.")
    ann_text = st.sidebar.text_area(
        "Annotation JSON(s) (one per line) — optional", ann_default,
        help="Alternative text source (OpenJury annotation JSON). Leave both "
             "empty to run from the lens folder alone (directions + activations).")
    annotations = tuple(s.strip() for s in ann_text.splitlines() if s.strip())
    names_path = st.sidebar.text_input(
        "feature_names.csv", _default_csv(lens_dir, a.names, "feature_names.csv"))
    fidelity_path = st.sidebar.text_input(
        "feature_fidelity.csv", _default_csv(lens_dir, a.fidelity, "feature_fidelity.csv"))
    win_rel_path = st.sidebar.text_input(
        "win_relevance.csv", _default_csv(lens_dir, a.win_relevance, "win_relevance.csv"))
    validation_path = st.sidebar.text_input(
        "diagnosis_validation.csv",
        _default_csv(lens_dir, a.validation, "diagnosis_validation.csv"))
    bank_default = a.bank or (str(Path(lens_dir) / "bank")
                              if lens_dir and (Path(lens_dir) / "bank").exists() else "")
    bank_dir = st.sidebar.text_input(
        "oriented-code bank dir — optional", bank_default,
        help="from `build-bank`; lets you diagnose any model in the pool here, "
             "no external run.")
    diagnosis_path = st.sidebar.text_input("model diagnosis CSV — optional", a.diagnosis)
    diag_battles_path = st.sidebar.text_input(
        "diagnosis evidence parquet — optional", a.diagnosis_battles,
        help="from `diagnose --battles-out`; enables inspecting the model's own "
             "battles per feature.")

    if not lens_dir:
        st.info("Set a lens directory in the sidebar.")
        return

    try:
        battles, z_diff, manifest = _load(lens_dir, annotations, corpus_path)
    except Exception as e:  # noqa: BLE001 — surface the load error in the UI
        st.error(f"Failed to load lens: {e}")
        return
    has_text = bool(corpus_path or annotations)

    names = _read_csv(names_path)
    fidelity = _read_csv(fidelity_path)
    ftable = feature_table(z_diff, names=names, fidelity=fidelity)

    st.title("PrefScope viewer")
    st.caption(f"{len(battles)} battles · {z_diff.shape[1]} axes · "
               f"input_rep={manifest.get('input_rep', '?')}")

    win_rel = _read_csv(win_rel_path)
    validation = _read_csv(validation_path)

    tab_feat, tab_detail, tab_reward, tab_valid, tab_diag = st.tabs(
        ["Features", "Feature detail", "Win relevance", "Validation",
         "Model diagnosis"])

    with tab_feat:
        only_pass = False
        if "fidelity_pass" in ftable.columns:
            only_pass = st.checkbox("Show only fidelity-passing axes", value=False)
        view = ftable[ftable["fidelity_pass"] == True] if only_pass else ftable  # noqa: E712
        st.dataframe(view, use_container_width=True, hide_index=True)

    with tab_detail:
        def _label(fid: int) -> str:
            if names is not None and "concept" in names.columns:
                row = names[names["feature_id"] == fid]
                if len(row):
                    return f"{fid}: {row.iloc[0]['concept']}"
            return f"feature {fid}"

        fid = st.selectbox("Axis", list(range(z_diff.shape[1])), format_func=_label)
        col = z_diff[:, fid]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("fires", f"{(col != 0).mean():.0%}")
        c2.metric("target-more", f"{(col > 0).mean():.0%}")
        c3.metric("target-less", f"{(col < 0).mean():.0%}")
        if fidelity is not None and "correlation" in fidelity.columns:
            frow = fidelity[fidelity["feature_id"] == fid]
            if len(frow):
                c4.metric("fidelity corr", f"{frow.iloc[0]['correlation']:.2f}")

        nz = col[col != 0]
        if len(nz):
            st.plotly_chart(
                px.histogram(x=nz, nbins=40,
                             labels={"x": "signed activation (A − B)"},
                             title="Activation distribution (firing battles)"),
                use_container_width=True)

        mode = st.radio("Rank battles by", ["abs", "pos", "neg"], horizontal=True,
                        format_func={"abs": "|activation|", "pos": "A-side (+)",
                                     "neg": "B-side (−)"}.get)
        n = st.slider("How many", 3, 50, 10)
        rows = top_battles(z_diff, battles, fid, mode=mode, n=n)
        if not has_text:
            st.caption("Completion text not loaded — set the corpus parquet (or an "
                       "annotation JSON) in the sidebar to see the A/B responses "
                       "behind each activation.")
        for _, r in rows.iterrows():
            with st.expander(f"z = {r['z']:+.3f}   ·   {str(r['prompt'])[:90]}"):
                st.markdown(f"**Prompt**\n\n{r['prompt']}")
                lc, rc = st.columns(2)
                lc.markdown(f"**A — {r.get('model_a', '?')}**\n\n{r['completion_a']}")
                rc.markdown(f"**B — {r.get('model_b', '?')}**\n\n{r['completion_b']}")
                st.caption(f"judge y = {r.get('y_judge', '?')}")

    with tab_reward:
        st.subheader("Which behaviours do humans reward?")
        if win_rel is None:
            st.info("Provide win_relevance.csv (from `prefscope win-relevance`).")
        else:
            wr = win_rel.copy()
            ycol = "win_assoc" if "win_assoc" in wr.columns else "correlation"
            wr = wr.sort_values(ycol)
            lab = "concept" if "concept" in wr.columns else "feature_id"
            wr["_label"] = wr[lab].astype(str).str.slice(0, 70)
            color = None
            if "significant" in wr.columns:
                wr["_sig"] = np.where(wr["significant"], "significant", "n.s.")
                color = "_sig"
            fig = px.bar(
                wr, x=ycol, y="_label", orientation="h", color=color,
                labels={ycol: "← humans penalise   ·   humans reward →", "_label": ""},
                title="Per-feature win association (A expresses concept ⇒ A preferred)")
            fig.add_vline(x=0, line_dash="dot")
            fig.update_layout(height=max(300, 26 * len(wr)))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(win_rel, use_container_width=True, hide_index=True)

    with tab_valid:
        st.subheader("Does the diagnosed deficit predict actual win rate?")
        if validation is None:
            st.info("Provide diagnosis_validation.csv (from `prefscope validate-diagnosis`).")
        else:
            v = validation.copy()
            xcol = "predicted_score_loo" if "predicted_score_loo" in v.columns else "predicted_score"
            if {xcol, "actual_win_rate"} <= set(v.columns):
                r = np.corrcoef(v[xcol], v["actual_win_rate"])[0, 1]
                c1, c2 = st.columns(2)
                c1.metric("R² (predicted vs actual)", f"{r * r:.3f}")
                c2.metric("models", f"{len(v)}")
                kw = dict(
                    x=xcol, y="actual_win_rate",
                    hover_name="model" if "model" in v.columns else None,
                    size="n_battles" if "n_battles" in v.columns else None,
                    labels={xcol: "predicted deficit score (Σ net_direction · win_assoc)",
                            "actual_win_rate": "actual human win rate"},
                    title="Predicted advantage vs actual win rate (one point per model)")
                try:                                  # OLS trendline needs statsmodels
                    import statsmodels  # noqa: F401
                    fig = px.scatter(v, trendline="ols", **kw)
                except Exception:
                    fig = px.scatter(v, **kw)
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(v.sort_values("actual_win_rate", ascending=False),
                         use_container_width=True, hide_index=True)

    with tab_diag:
        diag = _read_csv(diagnosis_path)
        # No external diagnosis CSV? Diagnose any pooled model from the bank, live.
        if diag is None and bank_dir and Path(bank_dir).exists():
            try:
                bank_Z, bank_meta, _ = _load_bank(bank_dir)
                from prefscope.pipeline.diagnose import diagnose_from_bank
                models = (bank_meta["self_model"].value_counts())
                models = models[models >= 20].index.tolist()
                st.caption(f"Diagnosing from bank · {len(models)} pooled models")
                model = st.selectbox("Model to diagnose", models, key="bank_model")
                ddf, dsum = diagnose_from_bank(bank_Z, bank_meta, model, names=fidelity)
                if win_rel is not None and "win_assoc" in win_rel.columns:
                    ddf = ddf.merge(win_rel[["feature_id", "win_assoc"]],
                                    on="feature_id", how="left")
                st.metric(f"{model} — win rate", f"{dsum['win_rate']:.3f}",
                          help=f"{dsum['n_battles']} battles vs the pool")
                if {"delta_vs_pool", "win_assoc"} <= set(ddf.columns):
                    hover = "concept" if "concept" in ddf.columns else "feature_id"
                    fig = px.scatter(
                        ddf, x="delta_vs_pool", y="win_assoc", hover_name=hover,
                        labels={"delta_vs_pool": "← does LESS than pool   ·   does MORE →",
                                "win_assoc": "← humans penalise   ·   reward →"},
                        title=f"{model}: gap quadrant — top-left = under-does a rewarded behaviour")
                    fig.add_hline(y=0, line_dash="dot")
                    fig.add_vline(x=0, line_dash="dot")
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(ddf, use_container_width=True, hide_index=True)
                diag = None  # handled
            except Exception as e:  # noqa: BLE001
                st.error(f"bank diagnosis failed: {e}")
        elif diag is None:
            st.info("Provide a diagnosis CSV (from `prefscope diagnose`) or a bank dir "
                    "in the sidebar to diagnose a pooled model here.")
        if diag is not None:
            st.dataframe(diag, use_container_width=True, hide_index=True)
            if {"net_direction", "outcome_assoc"} <= set(diag.columns):
                hover = "concept" if "concept" in diag.columns else "feature_id"
                fig = px.scatter(
                    diag, x="net_direction", y="outcome_assoc",
                    size="fire_rate" if "fire_rate" in diag.columns else None,
                    hover_name=hover,
                    labels={"net_direction": "← does LESS   ·   does MORE →",
                            "outcome_assoc": "← hurts   ·   helps win →"},
                    title="Strength / gap quadrant")
                fig.add_hline(y=0, line_dash="dot")
                fig.add_vline(x=0, line_dash="dot")
                st.plotly_chart(fig, use_container_width=True)

            pb = _read_parquet(diag_battles_path)
            if pb is not None:
                st.subheader("Evidence — the model's own battles")
                opts = list(diag["feature_id"]) if "feature_id" in diag.columns else []

                def _dlabel(fid):
                    if "concept" in diag.columns:
                        row = diag[diag["feature_id"] == fid]
                        if len(row):
                            return f"{fid}: {row.iloc[0]['concept']}"
                    return f"feature {fid}"

                dfid = st.selectbox("Feature", opts, format_func=_dlabel, key="diag_feat")
                dmode = st.radio(
                    "Show battles where the model…", ["more", "less", "abs"],
                    horizontal=True, key="diag_mode",
                    format_func={"more": "over-expresses (z>0)",
                                 "less": "under-expresses (z<0)",
                                 "abs": "strongest |z|"}.get)
                dn = st.slider("How many", 3, 30, 8, key="diag_n")
                ev = diagnosis_battles(pb, int(dfid), mode=dmode, n=dn)
                self_name = ev["self_model"].iloc[0] if len(ev) and "self_model" in ev else "model"
                for _, r in ev.iterrows():
                    head = f"z = {r['z']:+.3f}   ·   {r.get('outcome', '?')}   ·   {str(r['prompt'])[:80]}"
                    with st.expander(head):
                        st.markdown(f"**Prompt**\n\n{r['prompt']}")
                        lc, rc = st.columns(2)
                        lc.markdown(f"**{self_name} (diagnosed)**\n\n{r.get('self_completion','')}")
                        rc.markdown(f"**{r.get('other_model','opponent')}**\n\n{r.get('other_completion','')}")
                        st.caption(f"outcome for diagnosed model: {r.get('outcome','?')}")


if __name__ == "__main__":
    main()
