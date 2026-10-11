"""Typed records for the pinned data sources in configs/sources.toml.

Parsing is strict: every field is required, unknown access kinds and unexpected fields
are rejected, and nothing is fetched. Network access belongs to dataset loaders.
"""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, get_type_hints
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

CACHE_BUSTER_PARAM = "nocache"


@dataclass(frozen=True)
class ZipSource:
    """One zip archive read selectively over HTTP range requests."""

    key: str
    title: str
    doi: str
    version: str
    license: str
    url: str
    archive_bytes: int
    archive_md5: str
    cdn_caches_signed_redirect: bool

    def request_url(self, nonce: int) -> str:
        """URL for one remote read.

        When the host's CDN caches a short-lived signed redirect, a unique `nonce`
        query parameter forces a fresh redirect; otherwise the pinned URL is returned.
        """
        if not self.cdn_caches_signed_redirect:
            return self.url
        parts = urlsplit(self.url)
        query = parse_qsl(parts.query, keep_blank_values=True)
        query.append((CACHE_BUSTER_PARAM, str(nonce)))
        return urlunsplit(parts._replace(query=urlencode(query)))


@dataclass(frozen=True)
class S3Source:
    """One S3 bucket prefix read with a named AWS credentials profile."""

    key: str
    title: str
    terms: str
    bucket: str
    prefix: str
    region: str
    aws_profile: str


Source = ZipSource | S3Source

_ACCESS_KINDS: dict[str, type[ZipSource] | type[S3Source]] = {
    "remote_zip": ZipSource,
    "s3": S3Source,
}


def _build(key: str, table: dict[str, Any]) -> Source:
    if "access" not in table:
        raise ValueError(f"Source {key!r} has no 'access' field")
    access = table["access"]
    if access not in _ACCESS_KINDS:
        raise ValueError(f"Source {key!r} has unknown access kind {access!r}")
    cls = _ACCESS_KINDS[access]
    hints = get_type_hints(cls)
    expected = {f.name: hints[f.name] for f in fields(cls) if f.name != "key"}
    given = {name: value for name, value in table.items() if name != "access"}
    missing = sorted(set(expected) - set(given))
    unexpected = sorted(set(given) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"Source {key!r}: missing fields {missing}, unexpected fields {unexpected}"
        )
    for name, value in given.items():
        # Exact type match: a TOML boolean must not pass where an integer is expected.
        if type(value) is not expected[name]:
            raise TypeError(
                f"Source {key!r} field {name!r} must be {expected[name].__name__}, "
                f"got {type(value).__name__}"
            )
    return cls(key=key, **given)


def load_sources(path: Path) -> dict[str, Source]:
    """Parse every table in a sources TOML file into its typed record."""
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    return {key: _build(key, table) for key, table in document.items()}
