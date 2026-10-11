"""Offline checks for the pinned source configuration."""

from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest

from fmri_fus_bits.sources import CACHE_BUSTER_PARAM, S3Source, ZipSource, load_sources

REPO_SOURCES = Path(__file__).resolve().parents[1] / "configs" / "sources.toml"

ZIP_TABLE = """
[example]
access = "remote_zip"
title = "Example"
doi = "10.0000/example"
version = "1"
license = "CC BY 4.0"
url = "https://example.org/files/a.zip?download=1"
archive_bytes = 1024
archive_md5 = "00000000000000000000000000000000"
cdn_caches_signed_redirect = true
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sources.toml"
    path.write_text(text)
    return path


def test_repository_sources_parse_into_expected_kinds() -> None:
    sources = load_sources(REPO_SOURCES)
    assert set(sources) == {"griggs2023_bmi", "nunez_elizalde2022", "hcp_ya_motor"}
    griggs = sources["griggs2023_bmi"]
    nunez = sources["nunez_elizalde2022"]
    assert isinstance(griggs, ZipSource) and griggs.cdn_caches_signed_redirect
    assert isinstance(nunez, ZipSource) and not nunez.cdn_caches_signed_redirect
    assert isinstance(sources["hcp_ya_motor"], S3Source)


def test_cache_buster_is_appended_and_existing_query_preserved(tmp_path: Path) -> None:
    source = load_sources(write(tmp_path, ZIP_TABLE))["example"]
    assert isinstance(source, ZipSource)
    first = urlsplit(source.request_url(1))
    second = urlsplit(source.request_url(2))
    assert parse_qsl(first.query) == [("download", "1"), (CACHE_BUSTER_PARAM, "1")]
    assert first.path == "/files/a.zip"
    assert first.query != second.query


def test_unflagged_source_returns_pinned_url(tmp_path: Path) -> None:
    text = ZIP_TABLE.replace(
        "cdn_caches_signed_redirect = true", "cdn_caches_signed_redirect = false"
    )
    source = load_sources(write(tmp_path, text))["example"]
    assert isinstance(source, ZipSource)
    assert source.request_url(123) == source.url


def test_unknown_access_kind_is_rejected(tmp_path: Path) -> None:
    text = ZIP_TABLE.replace('access = "remote_zip"', 'access = "ftp"')
    with pytest.raises(ValueError, match="unknown access kind 'ftp'"):
        load_sources(write(tmp_path, text))


def test_missing_and_unexpected_fields_are_named(tmp_path: Path) -> None:
    text = ZIP_TABLE.replace('archive_md5 = "00000000000000000000000000000000"\n', "")
    text += 'mirror = "https://example.org/mirror.zip"\n'
    with pytest.raises(ValueError, match=r"missing fields \['archive_md5'\].*\['mirror'\]"):
        load_sources(write(tmp_path, text))


def test_boolean_is_not_accepted_as_integer(tmp_path: Path) -> None:
    text = ZIP_TABLE.replace("archive_bytes = 1024", "archive_bytes = true")
    with pytest.raises(TypeError, match="archive_bytes"):
        load_sources(write(tmp_path, text))
