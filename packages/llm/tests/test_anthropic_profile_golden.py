"""Golden-master characterization of the Anthropic model-metadata substrate.

**Purpose.** The Anthropic provider's live token-ceiling cache (formerly the
process globals ``_MODEL_LIMITS_CACHE`` / ``_CeilingEntry`` and the
``_refresh_model_limits`` poll) was lifted into a *generic*, reusable
``LiveApiSource`` class, so any provider — not just Anthropic — can source a facet
live with the same provenance / TTL / lock machinery. That lift was a genuine
structural change: it moved where the cache lives and rewrote the tests that
pinned the old module globals. So the invariant W1 relied on — *"the existing
tests pass byte-for-byte through the refactor"* — did not apply (the pinning tests
moved with the cache to the layer where the logic now lives,
``test_live_api_source.py``).

**This file is the replacement proof (now the durable guard).** It captures the
provider's *observable* model-metadata behavior — resolved capabilities and
request-shape constraints — across a ``(model x cache-state x config)`` matrix,
from the **pre-lift** code on ``main``, into a committed fixture
(``golden/anthropic_profile_golden.json``). The lifted code reproduces that fixture
**exactly** (the acceptance gate). This decoupled the proof from the internal cache
*structure* (which deliberately changed) and pins the only thing a consumer can
actually observe (which must not change).

Kept afterward, the fixture is a permanent regression + **vendor-drift** guard:
if a resource edit or a family-matching change silently shifts a resolved ceiling
or capability set, a golden cell breaks. Drift in these tables is the exact defect
class the substrate exists to kill, so the guard earns its keep beyond the lift.

**Why the harness survives the lift untouched.** The matrix drives only the
**public** surface:

- cache state is materialized through the public refresh entry point
  (``refresh_model_limits()``) fed by a scripted Models-API stand-in — never by
  poking the module globals directly, so the harness does not depend on *where*
  the cache lives;
- outputs are read through ``get_capabilities()`` / ``get_constraints()`` — the
  consumer-facing template methods, not the internal resolver free functions.

The single seam that needed a touch when the cache moved was :func:`_reset_caches`
(per-cell isolation): the per-instance ``LiveApiSource`` cache is now cold per cell
by construction, so it only clears the still-global rejected-params overlay.

**Scope.** This characterizes the *substrate* — capability + constraint (ceiling /
rejected-param) resolution, which flows through the cache being lifted.
``validate_model`` is deliberately excluded: it queries the live listing directly
and does not read the profile cache, so the lift cannot affect it and its own
tests already cover it.

**Regenerating the fixture** (only when the matrix or the *intended* behavior
changes — never to paper over an unexpected diff):

    uv run python packages/llm/tests/test_anthropic_profile_golden.py --regen
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from dataknobs_llm.llm.providers import anthropic as anthropic_mod

# Reuse the sanctioned Anthropic SDK stand-ins + the initialised-provider builder
# from the constraints test module (siblings are import-visible under pytest's
# default ``prepend`` import mode; the same path serves the ``--regen`` entry).
from _anthropic_stubs import _ScriptedModel, _provider_with_capture

_FIXTURE_PATH = Path(__file__).parent / "golden" / "anthropic_profile_golden.json"

# Sentinel ceilings scripted "from the Models API" — chosen to differ from every
# value in the bundled resource so a golden cell unambiguously shows whether the
# dynamic (live) or the resource layer supplied a resolved ceiling.
_DYN_OUT = 222_222
_DYN_IN = 888_888


def _reset_caches() -> None:
    """Isolate the process-global model-metadata state between matrix cells.

    Post-lift, the live ceiling cache is **per-provider-instance** (owned by each
    provider's ``LiveApiSource``), and every cell builds a fresh provider via
    :func:`_provider_with_capture`, so the live cache starts cold per cell with no
    clearing needed. Only the discovered-rejected-params overlay remains
    process-global (a separate self-correction cache, not part of the live-source
    lift) — cleared here so no cell inherits another's 400-recovery discovery.
    Defensive by name so it stays correct as the module evolves.

    LIFT NOTE (resolved): the live ceiling cache moved out of the module globals
    (``_MODEL_LIMITS_CACHE`` / ``_MODEL_LIMITS_LAST_FETCH`` / ``_MODEL_LIMITS_LOCKS``)
    into the per-instance ``LiveApiSource``; per-cell isolation is now inherent in
    the fresh-provider-per-cell construction. This is still the *only* place in the
    harness that knows where cache state lives — the matrix and fixture do not.
    """
    for name in ("_DISCOVERED_REJECTED_PARAMS",):
        cache = getattr(anthropic_mod, name, None)
        if cache is not None:
            cache.clear()


# ---------------------------------------------------------------------------
# The matrix: (model x cache-state x config)
# ---------------------------------------------------------------------------

# Representative model ids, one per distinct resolution path through the
# substrate (family-alias to the resource, dated snapshot, Claude-5 vs 4.x
# temperature rejection, fable/mythos family names, unknown / non-Claude).
_MODELS: tuple[str, ...] = (
    "claude-opus-5",  # Claude 5 flagship; resource ceilings; rejects temperature
    "claude-sonnet-5",  # Claude 5; resource ceilings; rejects temperature
    "claude-mythos-5",  # Claude 5 (invitation-only family name); rejects temperature
    "claude-fable-5",  # Claude 5 family name without opus/sonnet/haiku marker
    "claude-haiku-4-5-20251001",  # dated snapshot → `claude-haiku-4-5` alias; accepts temperature
    "claude-opus-4-8",  # Claude 4.x; resource ceilings; accepts temperature
    "claude-sonnet-5-20260514",  # dated Claude-5 snapshot → bare-family alias
    "claude-3-5-sonnet-20241022",  # older modern-capability family; no resource ceiling
    "some-unknown-model",  # nothing matches → permissive, base capabilities only
    "gpt-4o",  # non-Claude id → base capabilities only, no ceilings
)


def _cache_states(model: str) -> dict[str, list[tuple[str, int | None, int | None]]]:
    """The scripted Models-API listings for each named cache state, per model.

    Each value is a list of ``(model_id, max_tokens, max_input_tokens)`` triples
    materialized through the real refresh path (see :func:`_capture_cell`).
    """
    return {
        # Cold cache — no live poll; resolves via resource + heuristic only.
        "cold": [],
        # Live entry for the exact requested id, both ceilings present → dynamic
        # wins over the resource per facet.
        "dyn_full": [(model, _DYN_OUT, _DYN_IN)],
        # Live entry supplies only the input window → output ceiling falls through
        # to the resource; pins the per-facet (not per-record) merge.
        "dyn_input_only": [(model, None, _DYN_IN)],
        # Live cache populated but for an unrelated id → must not affect this
        # model's resolution (still resource/heuristic).
        "dyn_noise": [("claude-zzz-unrelated-9", 111, 222)],
    }


def _build_cells() -> list[dict[str, Any]]:
    """Assemble the ordered list of golden cells.

    Two blocks: a full ``model x cache-state`` cross with the plain config (the
    core lift-risk surface — live-vs-resource ceiling resolution), plus curated
    special cells for family-alias dynamic matches and every config-override path
    (all of which the lift must also preserve).
    """
    cells: list[dict[str, Any]] = []

    # Block 1 — model x cache-state, plain config.
    for model in _MODELS:
        for state_name, scripted in _cache_states(model).items():
            cells.append(
                {
                    "id": f"cross::{model}::{state_name}",
                    "model": model,
                    "scripted_models": [list(m) for m in scripted],
                    "config": {},
                }
            )

    # Block 2 — curated special cells (cold cache unless a dynamic list is given).
    def special(
        cid: str,
        model: str,
        config: dict[str, Any],
        scripted: list[tuple[str, int | None, int | None]] | None = None,
    ) -> None:
        cells.append(
            {
                "id": f"special::{cid}",
                "model": model,
                "scripted_models": [list(m) for m in (scripted or [])],
                "config": config,
            }
        )

    # Family-alias dynamic matches: the request and the live cache key differ but
    # resolve to each other via the substring family-alias rule.
    special(
        "bare_alias_dynamic",
        "claude-sonnet-5",
        {},
        [("claude-sonnet-5-20260514", _DYN_OUT, _DYN_IN)],  # dated key, bare request
    )
    special(
        "family_alias_dynamic",
        "claude-haiku-4-5-20251001",
        {},
        [("claude-haiku-4-5", _DYN_OUT, _DYN_IN)],  # short key, dated request
    )

    # Config-override facets via ``model_profile_overrides`` (the substrate's
    # always-wins layer), each pinning override precedence for one facet.
    special(
        "override_caps_profile",
        "claude-opus-5",
        {"model_profile_overrides": {"capabilities": ["vision"]}},
    )
    special(
        "override_output_ceiling",
        "claude-opus-5",
        {"model_profile_overrides": {"max_output_tokens": 4096}},
    )
    special(
        "override_input_ceiling",
        "claude-opus-5",
        {"model_profile_overrides": {"context_window": 50_000}},
    )
    special(
        "override_rejected_params",
        "claude-opus-4-8",  # normally accepts temperature
        {"model_profile_overrides": {"rejected_params": ["top_p"]}},
    )
    special(
        "override_per_model_mapping",
        "claude-opus-5",
        {"model_profile_overrides": {"claude-opus-5": {"max_output_tokens": 333}}},
    )
    # Override wins over a *dynamic* value on the same facet, while the untouched
    # facet still resolves from the live cache — the per-facet precedence crux.
    special(
        "override_beats_dynamic",
        "claude-sonnet-5",
        {"model_profile_overrides": {"max_output_tokens": 4096}},
        [("claude-sonnet-5", _DYN_OUT, _DYN_IN)],
    )

    # The separate ``config.capabilities`` / ``config.constraints`` override
    # surfaces (applied by the base template methods, not the resolver) — also
    # part of the observable contract the lift must not disturb.
    special(
        "config_capabilities_override",
        "claude-opus-5",
        {"capabilities": ["json_mode", "chat"]},
    )
    special(
        "config_constraints_ceiling",
        "claude-sonnet-5",
        {"constraints": {"max_tokens_ceiling": 100}},
    )
    special(
        "config_constraints_rejected",
        "claude-opus-4-8",
        {"constraints": {"rejected_params": ["frequency_penalty"]}},
    )

    return cells


async def _capture_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Materialize one cell and record its observable model-metadata outputs.

    Cache state is driven through the public refresh path (a scripted Models-API
    stand-in + ``refresh_model_limits()``), never by poking the module cache — so
    this reads identically before and after the cache is lifted into a generic
    source. Outputs come from the public template methods.
    """
    _reset_caches()
    provider, client = _provider_with_capture(cell["model"], **cell["config"])
    scripted = cell["scripted_models"]
    if scripted:
        client.models.models = [_ScriptedModel(*m) for m in scripted]
        await provider.refresh_model_limits()  # force=True internally; bypasses TTL

    capabilities = [c.value for c in provider.get_capabilities()]
    constraints = provider.get_constraints()
    return {
        "capabilities": capabilities,
        "constraints": {
            "rejected_params": sorted(constraints.rejected_params),
            "accepts_inline_system": constraints.accepts_inline_system,
            "max_tokens_ceiling": constraints.max_tokens_ceiling,
            "max_input_tokens": constraints.max_input_tokens,
        },
    }


async def _capture_all() -> list[dict[str, Any]]:
    """Capture every cell in matrix order (each cell isolated from the others)."""
    rows: list[dict[str, Any]] = []
    for cell in _build_cells():
        outputs = await _capture_cell(cell)
        rows.append({**cell, "outputs": outputs})
    return rows


def _load_fixture() -> list[dict[str, Any]]:
    with _FIXTURE_PATH.open(encoding="utf-8") as fh:
        data: list[dict[str, Any]] = json.load(fh)
    return data


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


async def test_golden_master_matches_fixture() -> None:
    """Live substrate outputs reproduce the committed golden fixture exactly.

    Pre-lift (on ``main``) this passes trivially — the fixture was captured from
    this code. Its job begins at the lift: the generic ``LiveApiSource`` +
    Anthropic migration must reproduce every cell, or this test names the drift.
    """
    assert _FIXTURE_PATH.exists(), (
        f"Golden fixture missing at {_FIXTURE_PATH}. Regenerate with:\n"
        f"  uv run python {Path(__file__).name} --regen"
    )
    expected = _load_fixture()
    actual = await _capture_all()

    expected_by_id = {row["id"]: row for row in expected}
    actual_by_id = {row["id"]: row for row in actual}

    # Matrix-drift guard: the fixture's cell set must equal the current matrix, so
    # adding a cell without regenerating (or a stale fixture) fails loudly rather
    # than silently skipping coverage.
    assert set(actual_by_id) == set(expected_by_id), (
        "Golden matrix drifted from the fixture — regenerate with --regen.\n"
        f"  only in code:    {sorted(set(actual_by_id) - set(expected_by_id))}\n"
        f"  only in fixture: {sorted(set(expected_by_id) - set(actual_by_id))}"
    )

    diffs: list[str] = []
    for cid, actual_row in actual_by_id.items():
        expected_outputs = expected_by_id[cid]["outputs"]
        if actual_row["outputs"] != expected_outputs:
            diffs.append(
                f"\n[{cid}] model={actual_row['model']} "
                f"config={actual_row['config']} "
                f"cache={actual_row['scripted_models']}\n"
                f"    expected: {expected_outputs}\n"
                f"    actual:   {actual_row['outputs']}"
            )
    assert not diffs, "Golden-master drift in {} cell(s):{}".format(len(diffs), "".join(diffs))


def test_matrix_has_expected_shape() -> None:
    """Sanity: cell ids are unique and the matrix is non-trivially sized.

    Cheap structural guard so a refactor that accidentally collapses the matrix
    (duplicate/empty ids) is caught independently of the data comparison.
    """
    cells = _build_cells()
    ids = [c["id"] for c in cells]
    assert len(ids) == len(set(ids)), "duplicate cell id in the golden matrix"
    assert len(cells) == len(_MODELS) * len(_cache_states(_MODELS[0])) + 11, (
        "cell count changed — update this guard and regenerate the fixture "
        "if the change is intended"
    )


def _regen() -> None:
    """Regenerate and write the golden fixture from the current code."""
    rows = asyncio.run(_capture_all())
    _FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _FIXTURE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"Wrote {len(rows)} golden cells to {_FIXTURE_PATH}")


if __name__ == "__main__":
    if "--regen" in sys.argv:
        _regen()
    else:
        print(
            "Golden-master harness. To (re)generate the committed fixture:\n"
            f"  uv run python {Path(__file__).name} --regen\n"
            "Otherwise run under pytest to assert against the fixture."
        )
