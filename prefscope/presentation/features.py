"""Bounded terminal presentation for feature-activation tables."""

from __future__ import annotations

import os
import sys
import unicodedata
from numbers import Integral

import pandas as pd


def _safe_text(value: object, *, limit: int) -> str:
    if value is None:
        return ""
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return ""
    except (TypeError, ValueError):
        pass
    text = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in str(value)
    )
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[: max(0, limit - 1)]}…"


class FeatureTableRenderer:
    """Render activation tables with lazy Rich and deterministic plain fallback."""

    def __init__(
        self,
        *,
        style: str = "auto",
        max_rows: int = 20,
        max_description_chars: int = 120,
        max_row_id_chars: int = 40,
    ) -> None:
        if style not in {"auto", "plain", "rich"}:
            raise ValueError("style must be auto, plain, or rich")
        for name, value in (
            ("max_rows", max_rows),
            ("max_description_chars", max_description_chars),
            ("max_row_id_chars", max_row_id_chars),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.style = style
        self.max_rows = int(max_rows)
        self.max_description_chars = int(max_description_chars)
        self.max_row_id_chars = int(max_row_id_chars)

    def _display_frame(self, table: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(table, pd.DataFrame):
            raise ValueError("feature activation output must be a pandas DataFrame")
        required = {"feature_id", "activation"}
        missing = required - set(table)
        if missing:
            raise ValueError(f"feature activation table is missing {sorted(missing)}")
        frame = table.head(self.max_rows).copy()
        label_column = None
        labels = pd.Series([None] * len(frame), index=frame.index, dtype=object)
        for candidate in ("description", "name", "concept"):
            if candidate not in frame:
                continue
            rendered = frame[candidate].map(
                lambda value: _safe_text(value, limit=self.max_description_chars)
            )
            if rendered.map(bool).any():
                label_column = candidate
                labels = frame[candidate].copy()
                labels = labels.mask(
                    labels.map(
                        lambda value: (
                            not _safe_text(value, limit=self.max_description_chars)
                        )
                    )
                )
                for fallback in ("description", "name", "concept"):
                    if fallback in frame and fallback != candidate:
                        labels = labels.combine_first(frame[fallback])
                break
        evidence = (
            set(frame["evidence_layer"].dropna())
            if "evidence_layer" in frame
            else set()
        )
        columns = {}
        multi_row = "row_id" in frame and frame["row_id"].nunique(dropna=False) > 1
        if multi_row:
            columns["row_id"] = frame["row_id"].map(
                lambda value: _safe_text(value, limit=self.max_row_id_chars)
            )
            if "rank" in frame:
                columns["rank"] = frame["rank"].map(lambda value: str(int(value)))
        columns["feature_id"] = frame["feature_id"].map(lambda value: str(int(value)))
        columns["activation"] = frame["activation"].map(
            lambda value: f"{float(value):.3f}"
        )
        if label_column is not None:
            display_label_column = (
                f"proposed_{label_column}"
                if evidence == {"proposed_label"}
                else label_column
            )
            columns[display_label_column] = labels.map(
                lambda value: _safe_text(value, limit=self.max_description_chars)
            )
        frame = pd.DataFrame(columns)
        return frame

    def format(self, table: pd.DataFrame) -> str:
        """Return deterministic plain text without importing Rich."""
        return self._display_frame(table).to_string(index=False)

    def print(self, table: pd.DataFrame, *, stream=None) -> None:
        """Print one bounded table to ``stream`` using Rich only when requested."""
        stream = sys.stdout if stream is None else stream
        use_rich = self.style == "rich" or (
            self.style == "auto"
            and not os.environ.get("NO_COLOR")
            and bool(getattr(stream, "isatty", lambda: False)())
        )
        if use_rich:
            try:
                from rich.console import Console
                from rich.table import Table
            except ImportError:
                if self.style == "rich":
                    raise ImportError(
                        "rich feature tables need the optional 'rich' package"
                    ) from None
            else:
                frame = self._display_frame(table)
                rich_table = Table(show_header=True, header_style="bold")
                for column in frame.columns:
                    style = "cyan" if column in {"row_id", "feature_id"} else None
                    if column == "activation":
                        style = "magenta"
                    justify = (
                        "right"
                        if column in {"rank", "feature_id", "activation"}
                        else "left"
                    )
                    rich_table.add_column(str(column), style=style, justify=justify)
                for row in frame.itertuples(index=False):
                    rich_table.add_row(*row)
                Console(
                    file=stream,
                    markup=False,
                    highlight=False,
                    force_terminal=self.style == "rich",
                    no_color=bool(os.environ.get("NO_COLOR")),
                ).print(rich_table)
                return
        stream.write(self.format(table) + chr(10))


__all__ = ["FeatureTableRenderer"]
