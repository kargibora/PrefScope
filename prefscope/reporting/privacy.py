"""Fail-closed privacy policies for portable report data."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import html
import math
from numbers import Integral, Real
import os
import re
from types import MappingProxyType
from typing import Mapping, Sequence
import unicodedata

from prefscope.core.redaction import is_sensitive_key, reject_secrets
from prefscope.core.representation import validate_portable_mapping


class PrivacyProfile(str, Enum):
    LOCAL = "local"
    SHAREABLE = "shareable"


class TextPolicy(str, Enum):
    NONE = "none"
    SNIPPETS = "snippets"
    FULL = "full"


_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_PHONE = re.compile(r"(?<![A-Za-z0-9])(?:\+?\d[ .()\-]*){7,}\d(?![A-Za-z0-9])")
_OPAQUE_ID = re.compile(r"^opaque:([a-z][a-z0-9_]*):([0-9a-f]{24})$")
_PII_KEYS = frozenset({
    "address", "birth_date", "contact", "email", "email_address", "first_name",
    "full_name", "ip_address", "last_name", "person_name", "phone", "phone_number",
    "social_security_number", "ssn", "user_name", "username",
})
_JSON_SAFE_INTEGER = 2 ** 53 - 1
_MAX_DEPTH = 32
_MAX_NODES = 100_000
_MAX_STRING_LENGTH = 100_000



def normalize_field_name(value: object) -> str:
    """Normalize snake/kebab/camelCase names to stable lower_snake_case."""
    if not isinstance(value, str) or not value:
        raise ValueError("privacy field names must be non-empty strings")
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    normalized = re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")
    if not normalized:
        raise ValueError("privacy field names must contain letters or digits")
    return normalized


def html_neutral_text(value: str) -> str:
    """Remove unsafe controls and escape text for inert HTML transport."""
    if not isinstance(value, str):
        raise ValueError("HTML-neutral text input must be a string")
    if any(_unsafe_text_character(char) for char in value):
        raise ValueError("text must not contain control or Unicode format characters")
    return html.escape(value, quote=True)


def validate_html_neutral_snippet(
    value: object, *, where: str = "snippet", max_chars: int | None = None,
) -> str:
    """Validate one canonical, inert, bounded text snippet."""
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    if max_chars is not None and (
        not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0
    ):
        raise ValueError("max_chars must be a positive integer or null")
    decoded = html.unescape(value)
    if html_neutral_text(decoded) != value:
        raise ValueError(f"{where} must use canonical HTML-neutral encoding")
    if _EMAIL.search(decoded) or _PHONE.search(decoded):
        raise ValueError(f"{where} must not contain direct PII literals")
    if any(_unsafe_text_character(char) for char in decoded):
        raise ValueError(f"{where} contains control or Unicode format characters")
    if max_chars is not None and len(decoded) > max_chars:
        if len(decoded) != max_chars + 1 or not decoded.endswith("…"):
            raise ValueError(f"{where} exceeds its bounded snippet length")
    return value


def _reject_pii_literals(value: object, *, where: str) -> None:
    def visit(item: object) -> None:
        if isinstance(item, str):
            if _EMAIL.search(item) or _PHONE.search(item):
                raise ValueError(f"{where} contains a direct PII literal")
            if any(_unsafe_text_character(char) for char in item):
                raise ValueError(
                    f"{where} contains control or Unicode format characters")
        elif isinstance(item, Mapping):
            for key, nested in item.items():
                _validate_raw_key(key, where=where)
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
    visit(value)


def _normalized_fields(values: Sequence[str], *, where: str) -> tuple[str, ...]:
    try:
        resolved = tuple(normalize_field_name(value) for value in values)
    except TypeError as exc:
        raise ValueError(f"{where} must be a sequence of field names") from exc
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{where} must not contain duplicate normalized names")
    return resolved


def _key_is_pii(key: str) -> bool:
    return key in _PII_KEYS or key.endswith(
        ("_email", "_address", "_phone", "_ssn", "_birth_date"))


_BIDI_CLASSES = frozenset({"RLE", "LRE", "RLO", "LRO", "PDF", "RLI", "LRI", "FSI", "PDI"})


def _unsafe_text_character(char: str) -> bool:
    return (
        (ord(char) < 0x20 and char not in "\n\r\t")
        or unicodedata.category(char) == "Cf"
        or unicodedata.bidirectional(char) in _BIDI_CLASSES
    )


def _validate_raw_key(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} keys must be non-empty strings")
    if _EMAIL.search(value) or any(char in value for char in "<>&\"'"):
        raise ValueError(f"{where} contains PII or markup in a field name")
    if any(_unsafe_text_character(char) for char in value):
        raise ValueError(f"{where} contains control characters in a field name")
    normalized = normalize_field_name(value)
    if is_sensitive_key(value):
        raise ValueError(f"{where} contains a credential-like field {value!r}")
    if _key_is_pii(normalized):
        raise ValueError(f"{where} contains PII-bearing field {value!r}")
    return normalized


def _secret_scan_projection(value: object, *, where: str) -> object:
    active: set[int] = set()
    nodes = 0

    def visit(item: object, depth: int) -> object:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError(f"{where} exceeds privacy traversal limits")
        if isinstance(item, str):
            if len(item) > _MAX_STRING_LENGTH:
                raise ValueError(f"{where} contains an overlong string")
            return item
        if item is None or isinstance(item, (bool, int, float)):
            return item
        if hasattr(item, "item") and callable(item.item):
            try:
                scalar = item.item()
            except (TypeError, ValueError):
                scalar = item
            if scalar is not item:
                return visit(scalar, depth + 1)
        item_id = id(item)
        if item_id in active:
            raise ValueError(f"{where} contains a cycle")
        if isinstance(item, Mapping):
            active.add(item_id)
            try:
                return {key: visit(nested, depth + 1) for key, nested in item.items()}
            finally:
                active.remove(item_id)
        if isinstance(item, (list, tuple)):
            active.add(item_id)
            try:
                return [visit(nested, depth + 1) for nested in item]
            finally:
                active.remove(item_id)
        return item

    return visit(value, 0)


def validate_privacy_safe(value: object, *, where: str) -> None:
    """Validate portable report metadata/prose without silently redacting it."""
    wrapped = value if isinstance(value, Mapping) else {"value": value}
    reject_secrets(_secret_scan_projection(wrapped, where=where), where=where)
    validate_portable_mapping(wrapped, where=where)

    def visit(item: object, key: str | None = None) -> None:
        if key is not None and _key_is_pii(key):
            raise ValueError(f"{where} must not contain PII-bearing field {key!r}")
        if isinstance(item, int) and not isinstance(item, bool):
            if not -_JSON_SAFE_INTEGER <= item <= _JSON_SAFE_INTEGER:
                raise ValueError(f"{where} integers must be browser-safe")
        if isinstance(item, str):
            validate_html_neutral_snippet(item, where=where)
        elif isinstance(item, Mapping):
            for raw_key, nested in item.items():
                normalized = _validate_raw_key(raw_key, where=where)
                visit(nested, normalized)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested, key)

    visit(value)


@dataclass(frozen=True)
class PrivacyPolicy:
    """Explicit persisted privacy schema plus a runtime-only opaque-ID salt."""

    profile_name: PrivacyProfile = PrivacyProfile.SHAREABLE
    text: TextPolicy = TextPolicy.NONE
    snippet_chars: int = 160
    allow_fields: tuple[str, ...] = field(default_factory=tuple)
    text_fields: tuple[str, ...] = field(default_factory=tuple)
    id_fields: tuple[str, ...] = field(default_factory=lambda: ("row_id",))
    cell_count_fields: tuple[str, ...] = field(default_factory=tuple)
    categorical_fields: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    object_fields: tuple[str, ...] = field(default_factory=tuple)
    list_fields: tuple[str, ...] = field(default_factory=tuple)
    redact_fields: tuple[str, ...] = field(default_factory=tuple)
    minimum_cell_count: int = 5
    opaque_ids: bool = True
    _salt: bytes = field(default_factory=lambda: os.urandom(32), repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            profile = (
                self.profile_name if isinstance(self.profile_name, PrivacyProfile)
                else PrivacyProfile(self.profile_name))
        except (TypeError, ValueError) as exc:
            raise ValueError("privacy profile must be 'local' or 'shareable'") from exc
        try:
            text = self.text if isinstance(self.text, TextPolicy) else TextPolicy(self.text)
        except (TypeError, ValueError) as exc:
            raise ValueError("privacy text must be 'none', 'snippets', or 'full'") from exc
        if profile is PrivacyProfile.SHAREABLE and text is TextPolicy.FULL:
            raise ValueError("shareable privacy does not permit full text")
        if (
            not isinstance(self.snippet_chars, int)
            or isinstance(self.snippet_chars, bool)
            or self.snippet_chars <= 0
        ):
            raise ValueError("privacy snippet_chars must be a positive integer")
        if (
            not isinstance(self.minimum_cell_count, int)
            or isinstance(self.minimum_cell_count, bool)
            or self.minimum_cell_count <= 0
        ):
            raise ValueError("privacy minimum_cell_count must be a positive integer")
        if not isinstance(self.opaque_ids, bool):
            raise ValueError("privacy opaque_ids must be a boolean")
        if profile is PrivacyProfile.SHAREABLE and not self.opaque_ids:
            raise ValueError("shareable privacy requires opaque_ids")
        if not isinstance(self._salt, bytes) or len(self._salt) < 16:
            raise ValueError("privacy internal salt must contain at least 16 bytes")

        resolved = {}
        for name in (
            "allow_fields", "text_fields", "id_fields", "cell_count_fields",
            "object_fields", "list_fields", "redact_fields",
        ):
            raw = getattr(self, name)
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f"privacy {name} must be a sequence")
            resolved[name] = _normalized_fields(raw, where=f"privacy {name}")
        if not isinstance(self.categorical_fields, Mapping):
            raise ValueError("privacy categorical_fields must be an object")
        categories = {}
        for raw_field, raw_values in self.categorical_fields.items():
            field_name = normalize_field_name(raw_field)
            if field_name in categories:
                raise ValueError("privacy categorical field names collide after normalization")
            if not isinstance(raw_values, (list, tuple)) or not raw_values:
                raise ValueError(
                    f"categorical field {field_name!r} needs a non-empty enum array")
            values = tuple(raw_values)
            if len(set(values)) != len(values) or any(
                not isinstance(value, str) or not value or len(value) > 256
                for value in values
            ):
                raise ValueError(
                    f"categorical field {field_name!r} enum values must be unique strings")
            for value in values:
                validate_html_neutral_snippet(
                    value, where=f"categorical field {field_name!r} enum")
            categories[field_name] = values
        role_sets = [
            set(resolved["allow_fields"]), set(resolved["text_fields"]),
            set(resolved["id_fields"]), set(resolved["cell_count_fields"]),
            set(categories), set(resolved["object_fields"]),
            set(resolved["list_fields"]), set(resolved["redact_fields"]),
        ]
        overlaps = set()
        for index, selected in enumerate(role_sets):
            for other in role_sets[index + 1:]:
                overlaps.update(selected & other)
        if overlaps:
            raise ValueError(
                "privacy field roles must be disjoint: " f"{sorted(overlaps)}")
        classified = set().union(*role_sets)
        secret = sorted(field for field in classified if is_sensitive_key(field))
        pii = sorted(field for field in classified if _key_is_pii(field))
        if secret:
            raise ValueError(f"privacy field declarations contain secret keys: {secret}")
        if pii:
            raise ValueError(f"privacy field declarations contain PII-bearing keys: {pii}")
        object.__setattr__(self, "profile_name", profile)
        object.__setattr__(self, "text", text)
        for name, values in resolved.items():
            object.__setattr__(self, name, values)
        object.__setattr__(
            self, "categorical_fields", MappingProxyType(categories))

    @classmethod
    def profile(cls, name: PrivacyProfile | str, **options) -> "PrivacyPolicy":
        try:
            resolved = name if isinstance(name, PrivacyProfile) else PrivacyProfile(name)
        except (TypeError, ValueError) as exc:
            raise ValueError("privacy profile must be 'local' or 'shareable'") from exc
        defaults = {
            "profile_name": resolved,
            "text": TextPolicy.FULL if resolved is PrivacyProfile.LOCAL else TextPolicy.NONE,
            "opaque_ids": resolved is PrivacyProfile.SHAREABLE,
            "minimum_cell_count": 1 if resolved is PrivacyProfile.LOCAL else 5,
        }
        defaults.update(options)
        return cls(**defaults)

    @classmethod
    def local(cls, **options) -> "PrivacyPolicy":
        return cls.profile(PrivacyProfile.LOCAL, **options)

    @classmethod
    def shareable(cls, **options) -> "PrivacyPolicy":
        return cls.profile(PrivacyProfile.SHAREABLE, **options)

    def opaque_id(self, field_name: str, value: object) -> str:
        field = normalize_field_name(field_name)
        if field not in self.id_fields:
            raise ValueError(f"field {field!r} is not declared as an ID field")
        if not isinstance(value, str) or not value:
            raise ValueError("opaque ID values must be non-empty strings")
        encoded = value.encode("utf-8")
        digest = hashlib.sha256(
            b"prefscope-report-opaque-id-v2\0"
            + self._salt
            + field.encode("utf-8")
            + b"\0"
            + encoded
        ).hexdigest()[:24]
        return f"opaque:{field}:{digest}"

    def to_manifest(self) -> dict[str, object]:
        return {
            "profile": self.profile_name.value,
            "text": self.text.value,
            "snippet_chars": self.snippet_chars,
            "allow_fields": list(self.allow_fields),
            "text_fields": list(self.text_fields),
            "id_fields": list(self.id_fields),
            "cell_count_fields": list(self.cell_count_fields),
            "categorical_fields": {
                name: list(values) for name, values in self.categorical_fields.items()
            },
            "object_fields": list(self.object_fields),
            "list_fields": list(self.list_fields),
            "redact_fields": list(self.redact_fields),
            "minimum_cell_count": self.minimum_cell_count,
            "opaque_ids": self.opaque_ids,
            "opaque_id_format": "opaque:{field}:sha256-96",
            "opaque_id_scope": "bundle_random_salt" if self.opaque_ids else "not_applied",
            "html_neutral": True,
            "unknown_shareable_fields": "reject",
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, object]) -> "PrivacyPolicy":
        expected = {
            "profile", "text", "snippet_chars", "allow_fields", "text_fields",
            "id_fields", "cell_count_fields", "categorical_fields", "object_fields",
            "list_fields", "redact_fields", "minimum_cell_count",
            "opaque_ids", "opaque_id_format", "opaque_id_scope", "html_neutral",
            "unknown_shareable_fields",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(f"privacy fields must be exactly {sorted(expected)}")
        if value["html_neutral"] is not True:
            raise ValueError("report privacy html_neutral must be true")
        if value["unknown_shareable_fields"] != "reject":
            raise ValueError("report privacy must reject unknown shareable fields")
        if value["opaque_id_format"] != "opaque:{field}:sha256-96":
            raise ValueError("report privacy opaque_id_format is unsupported")
        expected_scope = "bundle_random_salt" if value["opaque_ids"] else "not_applied"
        if value["opaque_id_scope"] != expected_scope:
            raise ValueError("report privacy opaque_id_scope is inconsistent")
        sequence_fields = (
            "allow_fields", "text_fields", "id_fields", "cell_count_fields",
            "object_fields", "list_fields", "redact_fields",
        )
        if any(not isinstance(value[name], (list, tuple)) for name in sequence_fields):
            raise ValueError("report privacy field declarations must be arrays")
        if not isinstance(value["categorical_fields"], Mapping):
            raise ValueError("report privacy categorical_fields must be an object")
        return cls(
            profile_name=value["profile"], text=value["text"],
            snippet_chars=value["snippet_chars"], allow_fields=tuple(value["allow_fields"]),
            text_fields=tuple(value["text_fields"]), id_fields=tuple(value["id_fields"]),
            cell_count_fields=tuple(value["cell_count_fields"]),
            categorical_fields={
                name: tuple(values)
                for name, values in value["categorical_fields"].items()
            },
            object_fields=tuple(value["object_fields"]),
            list_fields=tuple(value["list_fields"]),
            redact_fields=tuple(value["redact_fields"]),
            minimum_cell_count=value["minimum_cell_count"],
            opaque_ids=value["opaque_ids"],
        )

    def _field_role(self, field_name: str) -> str | None:
        if field_name in self.allow_fields:
            return "scalar"
        if field_name in self.text_fields:
            return "text"
        if field_name in self.id_fields:
            return "id"
        if field_name in self.cell_count_fields:
            return "cell_count"
        if field_name in self.categorical_fields:
            return "categorical"
        if field_name in self.object_fields:
            return "object"
        if field_name in self.list_fields:
            return "list"
        if field_name in self.redact_fields:
            return "redact"
        return None

    def _missing(self, value: object) -> bool:
        if value is None:
            return True
        value_type = type(value)
        if (
            value_type.__module__.startswith("pandas.")
            and value_type.__name__ in {"NAType", "NaTType"}
        ):
            return True
        return isinstance(value, Real) and not isinstance(value, bool) and math.isnan(float(value))

    def _check_cell_count(self, field_name: str, value: object) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"cell count field {field_name!r} must be an integer or null")
        if int(value) < self.minimum_cell_count:
            raise ValueError(
                f"cell count field {field_name!r} is below minimum_cell_count; "
                "suppress the cell before export")

    def sanitize(self, value: object) -> object:
        """Normalize, redact and validate one JSON-compatible data tree."""
        projection = _secret_scan_projection(value, where="report data")
        reject_secrets(projection, where="report data")
        _reject_pii_literals(projection, where="report data")
        result = self._sanitize(value, field_name=None)
        reject_secrets(result, where="report data")
        self.validate_sanitized(result)
        return result

    def _sanitize(self, value: object, *, field_name: str | None) -> object:
        role = None if field_name is None else self._field_role(field_name)
        if role == "redact":
            return None
        if field_name is not None and _key_is_pii(field_name):
            raise ValueError(f"report data contains PII-bearing field {field_name!r}")
        if (
            self.profile_name is PrivacyProfile.SHAREABLE
            and field_name is not None and role is None
        ):
            raise ValueError(f"unknown shareable field {field_name!r}")
        if self._missing(value):
            return None
        is_object = isinstance(value, Mapping)
        is_list = isinstance(value, (list, tuple))
        if self.profile_name is PrivacyProfile.SHAREABLE and field_name is not None:
            if role is None:
                raise ValueError(f"unknown shareable field {field_name!r}")
            if is_object and role != "object":
                raise ValueError(f"shareable object field {field_name!r} is not declared")
            if is_list and role != "list":
                raise ValueError(f"shareable list field {field_name!r} is not declared")
            if not is_object and not is_list and role in {"object", "list"}:
                raise ValueError(f"shareable container field {field_name!r} has wrong type")
        elif field_name is not None:
            if role == "object" and not is_object:
                raise ValueError(f"object field {field_name!r} has wrong type")
            if role == "list" and not is_list:
                raise ValueError(f"list field {field_name!r} has wrong type")

        if is_object:
            result = {}
            for raw_key, nested in value.items():
                normalized = _validate_raw_key(raw_key, where="report data")
                if normalized in result:
                    raise ValueError(
                        f"report keys collide after normalization at {normalized!r}")
                result[normalized] = self._sanitize(nested, field_name=normalized)
            return result
        if is_list:
            if self.profile_name is PrivacyProfile.SHAREABLE and any(
                not isinstance(item, Mapping) for item in value
            ):
                raise ValueError(
                    f"shareable list field {field_name!r} must contain typed objects")
            return [self._sanitize(item, field_name=None) for item in value]
        if field_name is None and self.profile_name is PrivacyProfile.SHAREABLE:
            raise ValueError("shareable root values must be objects")
        if role == "id":
            if not isinstance(value, str) or not value:
                raise ValueError("ID field values must be non-empty strings")
            return self.opaque_id(field_name, value) if self.opaque_ids else value
        if role == "text":
            if self.text is TextPolicy.NONE:
                return None
            if not isinstance(value, str):
                raise ValueError(f"text field {field_name!r} must contain a string or null")
            text = value
            if self.text is TextPolicy.SNIPPETS and len(text) > self.snippet_chars:
                text = text[: self.snippet_chars].rstrip() + "…"
            return html_neutral_text(text)
        if role == "cell_count":
            self._check_cell_count(field_name, value)
        if role == "categorical":
            if not isinstance(value, str) or value not in self.categorical_fields[field_name]:
                raise ValueError(
                    f"categorical field {field_name!r} is outside its declared enum")
            return value
        if role == "scalar":
            if isinstance(value, str):
                raise ValueError(
                    f"generic allow field {field_name!r} accepts only numeric/bool/null")
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, Integral):
            integer = int(value)
            if not -_JSON_SAFE_INTEGER <= integer <= _JSON_SAFE_INTEGER:
                raise ValueError("report integers must be browser-safe")
            return integer
        if isinstance(value, Real):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("report numbers must be finite or missing")
            return number
        if isinstance(value, str):
            # Undeclared dynamic strings are local-only. Direct PII was rejected in
            # the bounded preflight; transport still escapes markup.
            if self.profile_name is PrivacyProfile.SHAREABLE:
                raise ValueError(f"shareable string field {field_name!r} has no string role")
            return html_neutral_text(value)
        if isinstance(value, Enum):
            return self._sanitize(value.value, field_name=field_name)
        if hasattr(value, "item") and callable(value.item):
            try:
                scalar = value.item()
            except (TypeError, ValueError):
                scalar = value
            if scalar is not value:
                return self._sanitize(scalar, field_name=field_name)
        raise ValueError(f"report data contains unsupported value {type(value).__name__}")

    def validate_sanitized(self, value: object) -> None:
        """Validate already-sanitized data against every persisted policy role."""
        projection = _secret_scan_projection(value, where="sanitized report data")
        reject_secrets(projection, where="sanitized report data")
        _reject_pii_literals(projection, where="sanitized report data")

        def visit(item: object, field_name: str | None = None) -> None:
            role = None if field_name is None else self._field_role(field_name)
            if role == "redact":
                if item is not None:
                    raise ValueError(f"redacted field {field_name!r} must be null")
                return
            if field_name is not None and _key_is_pii(field_name):
                raise ValueError(f"sanitized data contains PII-bearing field {field_name!r}")
            if (
                self.profile_name is PrivacyProfile.SHAREABLE
                and field_name is not None and role is None
            ):
                raise ValueError(f"unknown shareable field {field_name!r}")
            if item is None:
                return
            is_object = isinstance(item, Mapping)
            is_list = isinstance(item, (list, tuple))
            if self.profile_name is PrivacyProfile.SHAREABLE and field_name is not None:
                if role is None:
                    raise ValueError(f"unknown shareable field {field_name!r}")
                if is_object and role != "object":
                    raise ValueError(f"shareable object field {field_name!r} is not declared")
                if is_list and role != "list":
                    raise ValueError(f"shareable list field {field_name!r} is not declared")
                if not is_object and not is_list and role in {"object", "list"}:
                    raise ValueError(
                        f"shareable container field {field_name!r} has wrong type")
            elif field_name is not None:
                if role == "object" and not is_object:
                    raise ValueError(f"object field {field_name!r} has wrong type")
                if role == "list" and not is_list:
                    raise ValueError(f"list field {field_name!r} has wrong type")

            if is_object:
                for raw_key, nested in item.items():
                    normalized = _validate_raw_key(
                        raw_key, where="sanitized report data")
                    if normalized != raw_key:
                        raise ValueError("sanitized report keys must be normalized")
                    visit(nested, normalized)
                return
            if is_list:
                if self.profile_name is PrivacyProfile.SHAREABLE and any(
                    not isinstance(nested, Mapping) for nested in item
                ):
                    raise ValueError(
                        f"shareable list field {field_name!r} must contain typed objects")
                for nested in item:
                    visit(nested, None)
                return
            if field_name is None and self.profile_name is PrivacyProfile.SHAREABLE:
                raise ValueError("shareable root values must be objects")
            if role == "id":
                if not isinstance(item, str):
                    raise ValueError(f"identifier field {field_name!r} is not opaque")
                if self.opaque_ids:
                    match = _OPAQUE_ID.fullmatch(item)
                    if match is None or match.group(1) != field_name:
                        raise ValueError(
                            f"identifier field {field_name!r} has the wrong type tag")
            elif role == "text":
                if self.text is TextPolicy.NONE:
                    raise ValueError(f"text field {field_name!r} must be null")
                validate_html_neutral_snippet(
                    item, where=f"text field {field_name!r}",
                    max_chars=(
                        self.snippet_chars
                        if self.text is TextPolicy.SNIPPETS else None),
                )
            elif role == "cell_count":
                self._check_cell_count(field_name, item)
            elif role == "categorical":
                if not isinstance(item, str) or item not in self.categorical_fields[field_name]:
                    raise ValueError(
                        f"categorical field {field_name!r} is outside its declared enum")
            elif role == "scalar" and isinstance(item, str):
                raise ValueError(
                    f"generic allow field {field_name!r} accepts only numeric/bool/null")
            elif isinstance(item, str) and self.profile_name is PrivacyProfile.SHAREABLE:
                raise ValueError(f"shareable string field {field_name!r} has no string role")

            if isinstance(item, str):
                validate_html_neutral_snippet(item, where="report string")
            elif isinstance(item, bool):
                return
            elif isinstance(item, Integral):
                if not -_JSON_SAFE_INTEGER <= int(item) <= _JSON_SAFE_INTEGER:
                    raise ValueError("report integers must be browser-safe")
            elif isinstance(item, Real):
                number = float(item)
                if math.isnan(number):
                    raise ValueError("sanitized missing numbers must be null")
                if not math.isfinite(number):
                    raise ValueError("report numbers must be finite")
            else:
                raise ValueError("report data contains a non-portable value")

        if isinstance(value, Mapping) and value.get("format") == "prefscope.json_table":
            expected = {"format", "version", "schema", "records"}
            if set(value) != expected or not isinstance(value["records"], list):
                raise ValueError("sanitized JSON-table envelope is malformed")
            validate_privacy_safe(
                {"format": value["format"], "version": value["version"],
                 "schema": value["schema"]},
                where="JSON-table envelope",
            )
            for record in value["records"]:
                visit(record)
        else:
            visit(value)
        validate_portable_mapping(
            value if isinstance(value, Mapping) else {"value": value},
            where="sanitized report data",
        )


def sanitize_json(value: object, policy: PrivacyPolicy) -> object:
    if not isinstance(policy, PrivacyPolicy):
        raise ValueError("policy must be a PrivacyPolicy")
    return policy.sanitize(value)


__all__ = [
    "PrivacyPolicy", "PrivacyProfile", "TextPolicy", "html_neutral_text",
    "normalize_field_name", "sanitize_json", "validate_html_neutral_snippet",
    "validate_privacy_safe",
]
