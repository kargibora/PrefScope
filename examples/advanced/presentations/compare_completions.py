#!/usr/bin/env python3
"""Compare two completions with a native PrefScope lens or a SAELens SAE.

Native completion lenses preserve their bundled proposed feature labels. Direct SAELens
checkpoints usually do not include names, so those runs report feature IDs honestly.
"""

from __future__ import annotations

import argparse
import math
import sys
import unicodedata
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Mapping, Sequence, TextIO


_DEFAULT_PROMPT = "Why do leaves often look green?"
_DEFAULT_COMPLETION_A = (
    "Leaves often look green because chlorophyll absorbs much of the red and blue "
    "light and reflects more green light."
)
_DEFAULT_COMPLETION_B = (
    "The green color usually comes from chlorophyll in leaf cells, which reflects "
    "green wavelengths more strongly than many other visible wavelengths."
)
_MAX_TOP_K = 50
_NATIVE_NOTE = (
    "Note: These values are raw completion-lens SAE activity. A sign or A-minus-B "
    "difference does not indicate reward, a winner, or semantic presence. The values "
    "also do not establish quality or a generator mechanism."
)
_SAELENS_NOTE = (
    "Note: These values are max-token-pooled raw SAELens activity. A sign or A-minus-B "
    "difference does not indicate reward, a winner, or semantic presence. The values "
    "also do not establish quality or a generator mechanism."
)
_DIRECT_DIFFERENCE_NOTE = (
    "Note: These are signed direct-difference codes f(e_A - e_B). A positive value "
    "uses the proposed positive pole; a negative value is the opposite pole. This is "
    "not f(e_A) - f(e_B). The sign and difference do not indicate reward, a winner, "
    "or semantic presence, and do not establish quality or a generator mechanism."
)


@dataclass(frozen=True)
class _RankedValue:
    feature_id: int
    value: float


def _bounded_top_k(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"must be an integer from 1 to {_MAX_TOP_K}"
        ) from error
    if parsed < 1 or parsed > _MAX_TOP_K:
        raise argparse.ArgumentTypeError(f"must be an integer from 1 to {_MAX_TOP_K}")
    return parsed


def _nonempty(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError("must be non-empty")
    return value


def _checked_vector(values: Sequence[Real], *, name: str) -> tuple[float, ...]:
    """Return one real-valued vector without silently flattening or coercing it."""
    shape = getattr(values, "shape", None)
    if shape is not None and len(shape) != 1:
        raise ValueError(f"{name} must be one-dimensional")
    try:
        raw = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a one-dimensional real vector") from error
    checked = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must contain only non-boolean real values")
        checked.append(float(value))
    return tuple(checked)


def _checked_feature_ids(feature_ids: Sequence[int], *, width: int) -> tuple[int, ...]:
    try:
        raw = tuple(feature_ids)
    except TypeError as error:
        raise ValueError("feature_ids must be a sequence of integers") from error
    if len(raw) != width:
        raise ValueError("feature_ids must exactly match the activity width")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
        raise ValueError("feature_ids must contain only non-boolean integers")
    checked = tuple(int(value) for value in raw)
    if len(set(checked)) != len(checked):
        raise ValueError("feature_ids must be unique")
    return checked


def _checked_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or int(top_k) < 1:
        raise ValueError("top_k must be a positive integer")
    if int(top_k) > _MAX_TOP_K:
        raise ValueError(f"top_k must not exceed {_MAX_TOP_K}")
    return int(top_k)


def _rank_activity(
    values: Sequence[Real], feature_ids: Sequence[int], top_k: int
) -> tuple[_RankedValue, ...]:
    """Rank finite raw activity from largest to smallest, with stable tie-breaking."""
    vector = _checked_vector(values, name="activity")
    ids = _checked_feature_ids(feature_ids, width=len(vector))
    limit = min(_checked_top_k(top_k), len(vector))
    rows = [
        _RankedValue(feature_id, value)
        for feature_id, value in zip(ids, vector)
        if math.isfinite(value) and value != 0.0
    ]
    rows.sort(key=lambda row: (-row.value, row.feature_id))
    return tuple(rows[:limit])


def _rank_differences(
    values_a: Sequence[Real],
    values_b: Sequence[Real],
    feature_ids: Sequence[int],
    top_k: int,
) -> tuple[_RankedValue, ...]:
    """Rank finite signed A-minus-B differences by decreasing absolute magnitude."""
    vector_a = _checked_vector(values_a, name="activity_a")
    vector_b = _checked_vector(values_b, name="activity_b")
    if len(vector_a) != len(vector_b):
        raise ValueError("activity_a and activity_b must have exactly the same width")
    ids = _checked_feature_ids(feature_ids, width=len(vector_a))
    limit = min(_checked_top_k(top_k), len(vector_a))
    rows = []
    for feature_id, value_a, value_b in zip(ids, vector_a, vector_b):
        difference = value_a - value_b
        if (
            math.isfinite(value_a)
            and math.isfinite(value_b)
            and math.isfinite(difference)
            and difference != 0.0
        ):
            rows.append(_RankedValue(feature_id, difference))
    rows.sort(key=lambda row: (-abs(row.value), row.feature_id))
    return tuple(rows[:limit])


def _rank_signed_activity(
    values: Sequence[Real], feature_ids: Sequence[int], top_k: int
) -> tuple[_RankedValue, ...]:
    """Rank finite signed direct-difference codes by absolute magnitude."""
    vector = _checked_vector(values, name="signed_activity")
    ids = _checked_feature_ids(feature_ids, width=len(vector))
    limit = min(_checked_top_k(top_k), len(vector))
    rows = [
        _RankedValue(feature_id, value)
        for feature_id, value in zip(ids, vector)
        if math.isfinite(value) and value != 0.0
    ]
    rows.sort(key=lambda row: (-abs(row.value), row.feature_id))
    return tuple(rows[:limit])


def _safe_label(value: str, *, max_chars: int = 120) -> str:
    """Normalize one untrusted display label and impose a hard character bound."""
    if not isinstance(value, str):
        raise ValueError("proposed labels must be strings")
    normalized = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in value
    )
    normalized = " ".join(normalized.split())
    if len(normalized) > max_chars:
        normalized = f"{normalized[: max_chars - 1]}…"
    return normalized


def _proposed_labels(lens: object, feature_ids: Sequence[int]) -> dict[int, str]:
    """Read optional bundled names and expose them only as proposed labels."""
    names = getattr(lens, "concept_names", None)
    if names is None:
        return {}
    allowed = set(_checked_feature_ids(feature_ids, width=len(feature_ids)))
    try:
        items = names.items()
    except AttributeError as error:
        raise ValueError("lens concept_names must be an ID-to-label mapping") from error
    labels: dict[int, str] = {}
    for raw_feature_id, raw_label in items:
        if (
            isinstance(raw_feature_id, bool)
            or not isinstance(raw_feature_id, Integral)
            or int(raw_feature_id) not in allowed
            or not isinstance(raw_label, str)
        ):
            continue
        label = _safe_label(raw_label)
        if label:
            labels[int(raw_feature_id)] = label
    return labels


def _plain_table(
    title: str,
    value_heading: str,
    rows: Sequence[_RankedValue],
    labels: Mapping[int, str],
    *,
    label_heading: str = "Proposed label",
) -> str:
    """Build a bounded plain-text table without terminal-specific dependencies."""
    headings = ["Feature ID", value_heading]
    if labels:
        headings.append(label_heading)
    body = []
    for row in rows[:_MAX_TOP_K]:
        cells = [str(row.feature_id), f"{row.value:+.6g}"]
        if labels:
            cells.append(labels.get(row.feature_id, "—"))
        body.append(cells)
    widths = [len(heading) for heading in headings]
    for cells in body:
        widths = [max(width, len(cell)) for width, cell in zip(widths, cells)]

    def line(cells: Sequence[str]) -> str:
        return " | ".join(
            cell.ljust(width) for cell, width in zip(cells, widths)
        ).rstrip()

    separator = "-+-".join("-" * width for width in widths)
    rendered = [title, line(headings), separator]
    rendered.extend(line(cells) for cells in body)
    if not body:
        rendered.append("(no active finite values)")
    return "\n".join(rendered)


def _render_comparison(
    activity_a: Sequence[_RankedValue],
    activity_b: Sequence[_RankedValue],
    differences: Sequence[_RankedValue],
    labels: Mapping[int, str],
    *,
    note: str = _NATIVE_NOTE,
    stream: TextIO | None = None,
) -> None:
    """Render colored Rich tables when available, otherwise clean plain text."""
    output = sys.stdout if stream is None else stream
    specifications = (
        ("Completion A", "Raw activity", activity_a, "cyan"),
        ("Completion B", "Raw activity", activity_b, "magenta"),
        (
            "Strongest signed A-minus-B differences",
            "Signed A-minus-B",
            differences,
            "yellow",
        ),
    )
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        for index, (title, heading, rows, _color) in enumerate(specifications):
            if index:
                print(file=output)
            print(_plain_table(title, heading, rows, labels), file=output)
        print(f"\n{note}", file=output)
        return

    console = Console(file=output, markup=False, highlight=False)
    for title, heading, rows, color in specifications:
        table = Table(title=title, title_style=f"bold {color}")
        table.add_column("Feature ID", justify="right", style=color)
        table.add_column(heading, justify="right")
        if labels:
            table.add_column("Proposed label")
        for row in rows[:_MAX_TOP_K]:
            value_style = "cyan" if row.value > 0 else "magenta"
            cells = [
                Text(str(row.feature_id)),
                Text(f"{row.value:+.6g}", style=value_style),
            ]
            if labels:
                cells.append(Text(labels.get(row.feature_id, "—")))
            table.add_row(*cells)
        console.print(table)
    console.print(Text(note, style="dim"))


def _render_direct_difference(
    rows: Sequence[_RankedValue],
    labels: Mapping[int, str],
    *,
    stream: TextIO | None = None,
) -> None:
    """Render one signed direct A-minus-B contrast without per-side claims."""
    output = sys.stdout if stream is None else stream
    title = "Direct difference lens: Completion A minus Completion B"
    heading = "Signed direct code"
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        print(
            _plain_table(
                title,
                heading,
                rows,
                labels,
                label_heading="Positive-pole proposed label",
            ),
            file=output,
        )
        print(f"\n{_DIRECT_DIFFERENCE_NOTE}", file=output)
        return

    console = Console(file=output, markup=False, highlight=False)
    table = Table(title=title, title_style="bold yellow")
    table.add_column("Feature ID", justify="right", style="cyan")
    table.add_column(heading, justify="right")
    if labels:
        table.add_column("Positive-pole proposed label")
    for row in rows[:_MAX_TOP_K]:
        value_style = "cyan" if row.value > 0 else "magenta"
        cells = [
            Text(str(row.feature_id)),
            Text(f"{row.value:+.6g}", style=value_style),
        ]
        if labels:
            cells.append(Text(labels.get(row.feature_id, "—")))
        table.add_row(*cells)
    console.print(table)
    console.print(Text(_DIRECT_DIFFERENCE_NOTE, style="dim"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two completions with a native PrefScope lens or SAELens."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--lens-repo",
        type=_nonempty,
        help="Hugging Face repository containing a native PrefScope lens.",
    )
    source.add_argument(
        "--lens-dir",
        type=_nonempty,
        help="Local native PrefScope lens directory.",
    )
    source.add_argument(
        "--saelens-release",
        type=_nonempty,
        help="Registered or explicitly trusted SAELens release.",
    )
    parser.add_argument(
        "--subfolder",
        type=_nonempty,
        help="Lens subfolder inside --lens-repo.",
    )
    parser.add_argument(
        "--revision",
        type=_nonempty,
        help="Exact or named Hugging Face revision for --lens-repo.",
    )
    parser.add_argument(
        "--sae-id",
        type=_nonempty,
        help="SAE ID required with --saelens-release.",
    )
    parser.add_argument(
        "--allow-unregistered-release",
        action="store_true",
        help=(
            "Allow a SAELens repository outside the installed registry. "
            "Use this only for a repository you trust."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only cached files when loading --lens-repo.",
    )
    parser.add_argument(
        "--prompt",
        type=_nonempty,
        help="Custom prompt; requires both custom completions.",
    )
    parser.add_argument(
        "--completion-a",
        type=_nonempty,
        help="Custom completion A; requires --prompt and --completion-b.",
    )
    parser.add_argument(
        "--completion-b",
        type=_nonempty,
        help="Custom completion B; requires --prompt and --completion-a.",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--top-k", default=10, type=_bounded_top_k)
    parser.add_argument(
        "--events",
        default=None,
        type=_nonempty,
        help="Opt in to automatic PrefScope operation events at this JSONL path.",
    )
    parser.add_argument(
        "--pretty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show or hide the compact event view when --events is set (default: show).",
    )
    return parser


def _comparison_text(args: argparse.Namespace) -> tuple[str, str, str]:
    """Use either the complete built-in demo or one complete custom triplet."""
    values = (args.prompt, args.completion_a, args.completion_b)
    provided = tuple(value is not None for value in values)
    if not any(provided):
        return _DEFAULT_PROMPT, _DEFAULT_COMPLETION_A, _DEFAULT_COMPLETION_B
    if not all(provided):
        raise ValueError(
            "--prompt, --completion-a, and --completion-b must be provided together"
        )
    return values


def _validate_lens_options(args: argparse.Namespace) -> None:
    """Reject source-specific options before creating the event file."""
    if args.lens_repo is not None:
        if args.sae_id is not None or args.allow_unregistered_release:
            raise ValueError(
                "--sae-id/--allow-unregistered-release require --saelens-release"
            )
        return
    if args.lens_dir is not None:
        if (
            args.subfolder is not None
            or args.revision is not None
            or args.sae_id is not None
            or args.allow_unregistered_release
            or args.local_files_only
        ):
            raise ValueError(
                "--lens-dir cannot be combined with repository or SAELens options"
            )
        return
    if args.sae_id is None:
        raise ValueError("--sae-id is required with --saelens-release")
    if args.subfolder is not None or args.revision is not None or args.local_files_only:
        raise ValueError(
            "--subfolder/--revision/--local-files-only apply only to --lens-repo"
        )


def _load_lens(args: argparse.Namespace, lens_type: object) -> object:
    """Load the validated native PrefScope or SAELens source."""
    _validate_lens_options(args)
    if args.lens_repo is not None:
        return lens_type.from_pretrained(
            args.lens_repo,
            subfolder=args.subfolder,
            revision=args.revision,
            device=args.device,
            local_files_only=args.local_files_only,
        )
    if args.lens_dir is not None:
        return lens_type.from_dir(args.lens_dir, device=args.device)
    return lens_type.from_saelens(
        args.saelens_release,
        args.sae_id,
        input_rep="individual",
        device=args.device,
        long_text_policy="truncate",
        include_bos=False,
        allow_unregistered_release=args.allow_unregistered_release,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)

    # These imports stay here so importing the example and its formatting helpers never
    # triggers an optional model or tensor dependency.
    from prefscope import Lens, PairItem
    from prefscope.observability import observe_run

    try:
        _validate_lens_options(args)
        prompt, completion_a, completion_b = _comparison_text(args)
    except ValueError as error:
        raise SystemExit(f"invalid arguments: {error}") from error

    item = PairItem(
        id="completion-comparison-1",
        x=prompt,
        y_a=completion_a,
        y_b=completion_b,
    )
    observation = (
        observe_run(args.events, pretty=args.pretty)
        if args.events is not None
        else nullcontext()
    )
    with observation:
        lens = _load_lens(args, Lens)
        input_rep = getattr(lens, "input_rep", None)
        if input_rep == "difference":
            features = lens.featurize([item], views=("response_difference",))
        elif input_rep == "individual":
            features = lens.featurize([item], views=("response_a", "response_b"))
        else:
            raise ValueError(
                "completion comparison requires an individual or difference lens"
            )

    feature_ids = features.feature_ids
    labels = _proposed_labels(lens, feature_ids)
    if input_rep == "difference":
        _render_direct_difference(
            _rank_signed_activity(features.array("z_diff")[0], feature_ids, args.top_k),
            labels,
        )
        return

    activity_a = features.array("z_a")[0]
    activity_b = features.array("z_b")[0]
    _render_comparison(
        _rank_activity(activity_a, feature_ids, args.top_k),
        _rank_activity(activity_b, feature_ids, args.top_k),
        _rank_differences(activity_a, activity_b, feature_ids, args.top_k),
        labels,
        note=_SAELENS_NOTE if args.saelens_release is not None else _NATIVE_NOTE,
    )


if __name__ == "__main__":
    main()
