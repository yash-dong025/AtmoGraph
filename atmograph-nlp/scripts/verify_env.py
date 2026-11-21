r"""Verify that the local Python environment can run the NLP module.

Run from the atmograph-nlp folder:
    .\.venv\Scripts\python.exe scripts\verify_env.py
"""

from __future__ import annotations

import importlib
import platform
import struct
import sys
from importlib import metadata

MIN_VERSION = (3, 10)
# (import name, distribution name, required)
PACKAGES: tuple[tuple[str, str, bool], ...] = (
    ("fastapi", "fastapi", True),
    ("pydantic", "pydantic", True),
    ("pydantic_settings", "pydantic-settings", True),
    ("spacy", "spacy", True),
    ("feedparser", "feedparser", True),
    ("httpx", "httpx", True),
    ("requests", "requests", True),
    ("bs4", "beautifulsoup4", True),
    ("lxml", "lxml", True),
    ("dateutil", "python-dateutil", True),
    ("tenacity", "tenacity", True),
    ("rapidfuzz", "rapidfuzz", True),
    ("transformers", "transformers", False),
    ("torch", "torch", False),
)
SPACY_MODEL = "en_core_web_sm"


def main() -> int:
    print("AtmoGraph NLP - environment verification")
    print("-" * 55)
    print(f"Python      : {platform.python_version()} ({struct.calcsize('P') * 8}-bit)")
    print(f"Executable  : {sys.executable}")
    print(f"Platform    : {platform.platform()}")

    failures: list[str] = []
    warnings: list[str] = []

    if sys.version_info < MIN_VERSION:
        failures.append(f"Python >= {MIN_VERSION[0]}.{MIN_VERSION[1]} is required")
    if struct.calcsize("P") * 8 != 64:
        failures.append("64-bit Python is required (spaCy has no 32-bit wheels)")

    print("\nPackages:")
    for import_name, distribution, required in PACKAGES:
        try:
            importlib.import_module(import_name)
            version = metadata.version(distribution)
            print(f"  [OK]      {distribution:<22} {version}")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            label = "MISSING" if required else "OPTIONAL"
            print(f"  [{label}] {distribution:<22} {type(exc).__name__}: {exc}")
            (failures if required else warnings).append(distribution)

    print("\nspaCy model:")
    try:
        print(f"  [OK]      {SPACY_MODEL:<22} {metadata.version(SPACY_MODEL)}")
    except metadata.PackageNotFoundError:
        print(f"  [MISSING] {SPACY_MODEL:<22} run: python -m spacy download {SPACY_MODEL}")
        warnings.append(SPACY_MODEL)

    print("\nResult:")
    if failures:
        print(f"  FAILED - missing required components: {', '.join(failures)}")
        return 1
    if warnings:
        print(f"  PASSED with warnings: {', '.join(warnings)}")
    else:
        print("  PASSED - environment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
