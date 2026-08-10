"""Pytest configuration for ``dataknobs-llm`` tests.

This suite shares scaffolding between modules — the aiohttp transport stubs in
``_aiohttp_error_stub``, the Anthropic SDK stand-ins in ``_anthropic_stubs``,
the Bedrock boundary stubs in ``_bedrock_stubs`` — and imports each by bare
name. That resolved only because pytest's ``prepend`` import mode inserts each
collected file's directory onto ``sys.path`` as a side effect of collecting it,
which the root configuration's ``importlib`` mode does not do. So the imports
worked under ``pytest packages/llm/tests`` and failed under the same command
with any second package named after it.

Declaring the root here states what was being relied on, and holds under both
modes and every invocation: pytest loads this file before collecting anything
beside it.

The declared root is ``_support/`` rather than this directory. Both make the
three modules importable by bare name, but a root exposes *every* immediate
child as a top-level name, and this directory has ten subdirectories — two of
which, ``prompts`` and ``integration``, are also names ``packages/bots/tests``
supplies. A regular package beats a namespace portion no matter which came
first on ``sys.path``, so a later ``from prompts import ...`` here would have
bound to bots' package while still passing under ``pytest packages/llm/tests``:
the same defect this file exists to close, one directory further down. A
directory holding nothing but scaffolding has no such children to leak.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_common.testing import declare_import_root

declare_import_root(Path(__file__).parent / "_support")
