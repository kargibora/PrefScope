"""Register oriented-bank construction and validation commands."""

from __future__ import annotations

from prefscope.cli.bank import _cmd_build_bank, _cmd_validate_diagnosis


def register_bank_commands(sub) -> None:
    pbk = sub.add_parser(
        "build-bank",
        help="project every battle in BOTH orientations -> pool baseline for "
        "diagnose --bank and validate-diagnosis",
    )
    pbk.add_argument("--lens-dir", required=True, help="frozen lens directory")
    pbk.add_argument(
        "--from-embeddings",
        required=True,
        dest="from_embeddings",
        help="dumped embedding dir (e_a.npy/e_b.npy/meta.parquet) from "
        "build-lens --dump-embeddings",
    )
    pbk.add_argument(
        "--label",
        choices=["judge", "human"],
        default="judge",
        help="orient outcomes by judge y_judge (default) or by human "
        "preference (needs --corpus with human_pref)",
    )
    pbk.add_argument(
        "--corpus",
        default=None,
        help="corpus parquet with human_pref (for --label human)",
    )
    pbk.add_argument("--out", required=True, help="output bank directory")
    pbk.add_argument(
        "--device",
        default="cpu",
        choices=["cuda", "mps", "cpu"],
        help="device for the SAE forward pass (CPU is fine)",
    )
    pbk.set_defaults(func=_cmd_build_bank)

    pv = sub.add_parser(
        "validate-diagnosis",
        help="does the diagnosed deficit predict actual win rate? (R^2 across models)",
    )
    pv.add_argument("--bank", required=True, help="oriented-code bank dir (build-bank)")
    pv.add_argument(
        "--win-relevance",
        required=True,
        dest="win_relevance",
        help="win-relevance CSV (feature reward weights)",
    )
    pv.add_argument("--out", required=True, help="output per-model CSV")
    pv.add_argument(
        "--weight-col",
        default="delta_win_rate",
        dest="weight_col",
        help="win-relevance column to weight features by "
        "(default delta_win_rate: length-controlled AME)",
    )
    pv.add_argument(
        "--all-features",
        action="store_true",
        help="weight by every feature, not just significant ones",
    )
    pv.add_argument(
        "--min-battles",
        type=int,
        default=20,
        dest="min_battles",
        help="skip models with fewer oriented battles than this",
    )
    pv.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the bootstrap CI and permutation null",
    )
    pv.add_argument(
        "--loo",
        action="store_true",
        help="leave-one-model-out: refit reward weights excluding each "
        "model's own battles (honest held-out R^2)",
    )
    pv.set_defaults(func=_cmd_validate_diagnosis)



__all__ = ["register_bank_commands"]
