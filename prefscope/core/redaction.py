"""Credential detection and redaction for JSON-like metadata.

This module is intentionally Torch-free. It centralizes the conservative secret
policy used by observability and by callers that must reject credentials rather
than store a redacted value.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, cast

REDACTED = "[REDACTED]"
DEFAULT_MAX_REDACTION_DEPTH = 32
DEFAULT_MAX_REDACTION_NODES = 100_000
DEFAULT_MAX_REDACTION_STRING_LENGTH = 100_000

_T = TypeVar("_T")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_AUTHORIZATION_RE = re.compile(r"(?i)\bauthorization(\s*[:=]\s*)[^\r\n,;]+")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{6,}")
_BASIC_RE = re.compile(r"(?i)\bBasic\s+[A-Za-z0-9+/]{8,}={0,2}")
_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]{4,}\."
    r"[A-Za-z0-9_-]{4,}(?![A-Za-z0-9_-])"
)
_PROVIDER_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{8,}|"
    r"hf_[A-Za-z0-9]{8,}|"
    r"gh[pousr]_[A-Za-z0-9]{8,}|"
    r"github_pat_[A-Za-z0-9_]{8,}|"
    r"glpat-[A-Za-z0-9_-]{8,}|"
    r"AKIA[A-Z0-9]{12,}|"
    r"ASIA[A-Z0-9]{12,}|"
    r"AIza[A-Za-z0-9_-]{12,}|"
    r"xox[baprs]-[A-Za-z0-9-]{8,}|"
    r"(?:sk|rk)_live_[A-Za-z0-9]{8,}|"
    r"npm_[A-Za-z0-9]{8,}|"
    r"pypi-[A-Za-z0-9_-]{8,}"
    r")(?![A-Za-z0-9_-])"
)
_URI_CREDENTIAL_RE = re.compile(r"(?i)([a-z][a-z0-9+.-]*://[^:/\s]+:)[^@\s/]+@")
_QUERY_SECRET_RE = re.compile(r"(?i)([?&](?:sig|signature|key|token)=)[^&#\s]+")
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[\w.+-]+@(?:[\w.-]+\.[A-Za-z]{2,}|"
    r"\[(?:IPv6:)?[0-9A-Fa-f:.]+\])(?![\w.-])"
)
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)|\d{2,4})"
    r"[ .-]\d{3,4}[ .-]\d{4}(?!\w)|(?<!\w)\+\d{10,15}(?!\w)"
)
_ASSIGNMENT_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_.-]{1,64})(\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY[^-\r\n]*-----.*?"
    r"(?:-----END [^-\r\n]*PRIVATE KEY[^-\r\n]*-----|$)",
    re.DOTALL,
)

_HARMLESS_TOKEN_KEYS = frozenset(
    {
        "completion_tokens",
        "input_tokens",
        "max_tokens",
        "num_tokens",
        "output_tokens",
        "prompt_tokens",
        "token_count",
        "token_counts",
        "token_index",
        "token_indices",
        "token_length",
        "token_lengths",
        "total_tokens",
    }
)
_SENSITIVE_COMPACT_KEYS = frozenset(
    {
        "accesskey",
        "accesskeyid",
        "accountkey",
        "accesstoken",
        "apikey",
        "auth",
        "authtoken",
        "authorization",
        "authorizationheader",
        "awsaccesskeyid",
        "awssecretaccesskey",
        "clientsecret",
        "connectionstring",
        "connectionstrings",
        "connstr",
        "connstring",
        "defaultconnection",
        "databaseurl",
        "credential",
        "credentials",
        "hftoken",
        "password",
        "passwd",
        "privatekey",
        "refreshtoken",
        "secretaccesskey",
        "sessioncookie",
        "sharedaccesskey",
        "sharedaccesssignature",
        "token",
    }
)


def normalize_key(key: str) -> str:
    """Normalize separators and camelCase without losing token boundaries."""
    separated = _CAMEL_BOUNDARY_RE.sub("_", key)
    return re.sub(r"[^a-z0-9]+", "_", separated.lower()).strip("_")


def is_sensitive_key(key: str) -> bool:
    """Return whether a field name is likely to contain credential material."""
    normalized = normalize_key(key)
    if normalized in _HARMLESS_TOKEN_KEYS:
        return False
    compact = normalized.replace("_", "")
    if compact in _SENSITIVE_COMPACT_KEYS:
        return True
    segments = set(normalized.split("_"))
    if {"access", "key"} <= segments or {"access", "token"} <= segments:
        return True
    if {"account", "key"} <= segments:
        return True
    if "connection" in segments and segments & {"string", "strings"}:
        return True
    if normalized.endswith("_connection"):
        return True
    if {"secret", "key"} <= segments or {"private", "key"} <= segments:
        return True
    if {"auth", "token"} <= segments or {"api", "key"} <= segments:
        return True
    if segments & {"authorization", "credential", "credentials", "password", "passwd"}:
        return True
    if "secret" in segments or "cookie" in segments:
        return True
    if normalized.endswith("_token"):
        return True
    return False


def _is_pii_key(key: str) -> bool:
    normalized = normalize_key(key)
    return normalized in {
        "email",
        "email_address",
        "phone",
        "phone_number",
        "telephone",
    } or normalized.endswith(("_email", "_email_address", "_phone", "_phone_number"))


def redact_text(value: str) -> str:
    """Redact common credentials embedded in a string."""
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("secret-scanned strings must be valid UTF-8") from exc

    value = _PRIVATE_KEY_RE.sub(REDACTED, value)
    value = _AUTHORIZATION_RE.sub(
        lambda match: f"authorization{match.group(1)}{REDACTED}", value
    )
    value = _BEARER_RE.sub(f"Bearer {REDACTED}", value)
    value = _BASIC_RE.sub(f"Basic {REDACTED}", value)
    value = _JWT_RE.sub(REDACTED, value)
    value = _PROVIDER_TOKEN_RE.sub(REDACTED, value)
    value = _URI_CREDENTIAL_RE.sub(lambda match: f"{match.group(1)}{REDACTED}@", value)
    value = _QUERY_SECRET_RE.sub(lambda match: f"{match.group(1)}{REDACTED}", value)
    value = _EMAIL_RE.sub(REDACTED, value)
    value = _PHONE_RE.sub(REDACTED, value)

    def redact_assignment(match: re.Match[str]) -> str:
        if not is_sensitive_key(match.group(1)):
            return match.group(0)
        return f"{match.group(1)}{match.group(2)}{REDACTED}"

    return _ASSIGNMENT_RE.sub(redact_assignment, value)


def _walk(
    value: Any,
    *,
    reject: bool,
    where: str,
    path: str,
    ancestors: set[int],
    depth: int,
    nodes: list[int],
) -> Any:
    nodes[0] += 1
    if nodes[0] > DEFAULT_MAX_REDACTION_NODES:
        raise ValueError(
            f"{where} exceeds maximum secret-scan node count "
            f"{DEFAULT_MAX_REDACTION_NODES}"
        )
    if depth > DEFAULT_MAX_REDACTION_DEPTH:
        raise ValueError(
            f"{where} exceeds maximum secret-scan depth {DEFAULT_MAX_REDACTION_DEPTH}"
        )
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > DEFAULT_MAX_REDACTION_STRING_LENGTH:
            raise ValueError(
                f"{where} exceeds maximum secret-scan string length "
                f"{DEFAULT_MAX_REDACTION_STRING_LENGTH} at {path}"
            )
        redacted = redact_text(value)
        if reject and redacted != value:
            raise ValueError(f"{where} contains credential-like text at {path}")
        return redacted

    value_id = id(value)
    if value_id in ancestors:
        raise ValueError(f"{where} contains a cycle at {path}")

    if isinstance(value, Mapping):
        ancestors.add(value_id)
        output: dict[str, Any] = {}
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{where} contains a non-string key at {path}")
                if len(key) > DEFAULT_MAX_REDACTION_STRING_LENGTH:
                    raise ValueError(
                        f"{where} exceeds maximum secret-scan key length "
                        f"{DEFAULT_MAX_REDACTION_STRING_LENGTH} at {path}"
                    )
                clean_key = redact_text(key)
                key_path = f"{path}.{clean_key}" if path else clean_key
                key_is_sensitive = is_sensitive_key(key) or _is_pii_key(key)
                if reject and (key_is_sensitive or clean_key != key):
                    raise ValueError(
                        f"{where} contains a credential-like field at {key_path}"
                    )
                if clean_key in output:
                    raise ValueError(
                        f"{where} contains keys that collide after redaction at {path or '<root>'}"
                    )
                if key_is_sensitive:
                    # Validate the hidden subtree too, so many sensitive fields
                    # cannot bypass the traversal budgets merely because their
                    # values will be replaced.
                    _walk(
                        item,
                        reject=False,
                        where=where,
                        path=key_path,
                        ancestors=ancestors,
                        depth=depth + 1,
                        nodes=nodes,
                    )
                    output[clean_key] = REDACTED
                else:
                    output[clean_key] = _walk(
                        item,
                        reject=reject,
                        where=where,
                        path=key_path,
                        ancestors=ancestors,
                        depth=depth + 1,
                        nodes=nodes,
                    )
        finally:
            ancestors.remove(value_id)
        return output

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        ancestors.add(value_id)
        try:
            return [
                _walk(
                    item,
                    reject=reject,
                    where=where,
                    path=f"{path}[{index}]",
                    ancestors=ancestors,
                    depth=depth + 1,
                    nodes=nodes,
                )
                for index, item in enumerate(value)
            ]
        finally:
            ancestors.remove(value_id)

    raise ValueError(
        f"{where} contains unsupported value type {type(value).__name__} at {path}"
    )


def _validate_where(where: str) -> None:
    if not isinstance(where, str) or not where:
        raise ValueError("where must be a non-empty string")
    if len(where) > DEFAULT_MAX_REDACTION_STRING_LENGTH:
        raise ValueError("where exceeds maximum secret-scan string length")


def redact_secrets(value: _T, where: str = "value") -> _T:
    """Return a bounded recursive copy with credentials replaced."""
    _validate_where(where)
    return cast(
        _T,
        _walk(
            value,
            reject=False,
            where=where,
            path="",
            ancestors=set(),
            depth=0,
            nodes=[0],
        ),
    )


def reject_secrets(value: _T, *, where: str) -> _T:
    """Fail closed if a bounded JSON-like value contains credentials."""
    _validate_where(where)
    _walk(
        value,
        reject=True,
        where=where,
        path="",
        ancestors=set(),
        depth=0,
        nodes=[0],
    )
    return value


__all__ = [
    "DEFAULT_MAX_REDACTION_DEPTH",
    "DEFAULT_MAX_REDACTION_NODES",
    "DEFAULT_MAX_REDACTION_STRING_LENGTH",
    "REDACTED",
    "is_sensitive_key",
    "normalize_key",
    "redact_secrets",
    "redact_text",
    "reject_secrets",
]
