"""Amazon Bedrock LLM provider implementation.

This module provides Amazon Bedrock integration for dataknobs-llm, serving
**both** completion/chat (via the unified Converse API) and embeddings
(Titan / Cohere via ``invoke_model``) through a single provider registered
as ``"bedrock"``.

Authentication uses the standard AWS credential chain (environment,
``~/.aws`` shared config, EC2/ECS instance or task IAM role) — there is no
API key. Explicit credentials and region may be supplied via
``LLMConfig.options`` when the default chain is not desired.

The provider reuses the shared, loop-safe aioboto3 session factory in
``dataknobs_common.aws`` (the same factory every AWS consumer uses), so
session construction is offloaded off the event loop and process-cached.
Per-operation ``bedrock-runtime`` clients are short-lived async context
managers.

``aioboto3`` is an *optional* dependency imported lazily, so importing this
module never requires it. Install the async Bedrock transport with::

    pip install 'dataknobs-llm[bedrock]'

Example:
    ```python
    from dataknobs_llm import BedrockProvider
    from dataknobs_llm.llm.base import LLMConfig

    config = LLMConfig(
        provider="bedrock",
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        temperature=0.7,
        max_tokens=1024,
        options={"region_name": "us-west-2"},  # credentials via IAM chain
    )

    async with BedrockProvider(config) as llm:
        response = await llm.complete("Explain quantum computing")
        print(response.content)

        async for chunk in llm.stream_complete("Write a haiku"):
            print(chunk.delta, end="", flush=True)

    # Embeddings (Titan / Cohere)
    embed_config = LLMConfig(
        provider="bedrock",
        model="amazon.titan-embed-text-v2:0",
        dimensions=1024,
        options={"region_name": "us-west-2"},
    )
    async with BedrockProvider(embed_config) as embedder:
        vector = await embedder.embed("sample text")
        print(f"Dimensions: {len(vector)}")
    ```

See Also:
    - Amazon Bedrock Converse API:
      https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
    - dataknobs_common.aws: shared aioboto3 session factory
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, overload

from dataknobs_common.aws import AwsSessionConfig, create_aioboto3_session
from dataknobs_common.exceptions import ConfigurationError

from ..base import (
    AsyncLLMProvider,
    LLMAdapter,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMStreamResponse,
    ModelCapability,
    normalize_claude_stop_reason,
    normalize_llm_config,
)
from ..model_profile import (
    BundledResourceSource,
    CallableModelMetadataSource,
    ConfigOverrideSource,
    LayeredModelProfileResolver,
    LiveApiSource,
    ModelProfile,
)
from ..profile_detection import ProfileDetectionMixin
from ._claude_shared import (
    CLAUDE_ONLY_HEURISTIC_PROFILE_SOURCE,
    CLAUDE_RESOURCE_PROFILE_SOURCE,
)

if TYPE_CHECKING:
    from dataknobs_config.config import Config

    from dataknobs_llm.prompts import AsyncPromptBuilder

logger = logging.getLogger(__name__)


# Fixed connect timeout (seconds) for bedrock-runtime clients — fail fast on a
# stalled TCP connect rather than hang on boto's 60s default. The *read*
# timeout is per-request (``LLMConfig.timeout``), sized to the generation
# budget; see :meth:`BedrockProvider._client_kwargs`.
_CONNECT_TIMEOUT_SECONDS = 10


# Region / cross-region inference-profile prefixes prepended to a base model
# id (e.g. ``us.anthropic.claude-...``). Stripped to recover the base id for
# family / capability detection.
_REGION_PREFIXES: tuple[str, ...] = ("us.", "eu.", "apac.", "us-gov.")

# Non-Claude vendor-id prefixes recognised as valid Bedrock foundation models
# by the data-sourced ``validate_model`` heuristic. Region / inference-profile
# prefixes (``us.`` / ``eu.`` / ``apac.`` / ``us-gov.``) are stripped by
# :func:`_canonical_model_id` before this check, so only the bare vendor segment
# matters. No control-plane call — inference-only task roles keep working (the
# opt-in ``ListFoundationModels`` availability source is separate; see
# :meth:`BedrockProvider.validate_model`).
_VENDOR_PREFIXES: tuple[str, ...] = (
    "amazon.",
    "anthropic.",
    "meta.",
    "mistral.",
    "cohere.",
    "ai21.",
)

# Non-Claude family-id substrings that carry vision (multimodal input).
# Consumed only by the last-resort :func:`_bedrock_heuristic` — a model present
# in ``bedrock_models.yaml`` resolves its capabilities from that resource, and
# Claude vision comes from the shared Claude capability source.
_VISION_FAMILIES: tuple[str, ...] = (
    "nova-lite",
    "nova-pro",
    "nova-premier",
    "llama3-2-11b",
    "llama3-2-90b",
    "pixtral",
)


def _canonical_model_id(model: str) -> str:
    """Strip a leading region / inference-profile prefix from a model id.

    ``us.anthropic.claude-...`` -> ``anthropic.claude-...``. Used so family
    and capability detection work uniformly for both plain model ids and
    cross-region inference-profile ids.
    """
    for prefix in _REGION_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _bedrock_heuristic(model: str) -> ModelProfile:
    """Last-resort capability/availability source for unlisted Bedrock models.

    The corrected, demoted form of the old inline ``_detect_capabilities`` +
    ``_KNOWN_MODEL_PREFIXES`` logic. Lowest precedence in the resolver, so a
    model present in ``bedrock_models.yaml`` (or, for Claude, the shared Claude
    sources) resolves from there; this only classifies an *unlisted* family by
    name. Contributes:

    - ``capabilities`` — EMBEDDINGS-only for an embedding family (disjoint: an
      embed model cannot chat / stream / call tools), else the base chat set
      plus VISION for a known multimodal family.
    - ``available`` — ``True`` for a recognised vendor prefix (preserving the
      old permissive prefix ``validate_model``, now data-sourced), else ``None``
      so an unknown vendor resolves to unavailable.
    """
    model_lower = _canonical_model_id(model.lower())
    available = model_lower.startswith(_VENDOR_PREFIXES) or None
    if any(token in model_lower for token in ("titan-embed", "cohere.embed", "-embed-")):
        return ModelProfile(
            capabilities=frozenset({ModelCapability.EMBEDDINGS}),
            available=available,
        )
    capabilities = {
        ModelCapability.TEXT_GENERATION,
        ModelCapability.CHAT,
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
    }
    if any(token in model_lower for token in _VISION_FAMILIES):
        capabilities.add(ModelCapability.VISION)
    return ModelProfile(capabilities=frozenset(capabilities), available=available)


def _bedrock_availability_extractor(summary: dict[str, Any]) -> ModelProfile:
    """Project one ``ListFoundationModels`` summary into an availability partial.

    ``ListFoundationModels`` returns model *summaries* — it serves availability
    and modalities but **not** token ceilings — so the only facet the live
    source contributes is ``available=True`` (a model in the account catalog is
    available). Capabilities stay resource-authoritative: a partial capability
    set derived from modalities would, under the first-non-``None``-per-facet
    merge, *replace* the resource's full set — so modalities feed only the
    tooling drift check, not the runtime resolver.
    """
    return ModelProfile(available=True)


def _bedrock_summary_model_id(summary: dict[str, Any]) -> str | None:
    """Read a ``ListFoundationModels`` summary's ``modelId`` (LiveApiSource key).

    The summaries are dicts, so the default attribute-based
    :func:`~..model_profile._default_model_id` does not apply.
    """
    value = summary.get("modelId")
    return str(value) if value is not None else None


#: The stateless lower-precedence Bedrock model-metadata sources — module
#: singletons (they read only the bundled resource / apply a pure family rule,
#: no per-instance state). The Claude family ceiling + capability sources are the
#: SHARED ``_claude_shared`` singletons, composed *between* the Bedrock resource
#: and the Bedrock heuristic (see :meth:`BedrockProvider._profile_resolver`) so a
#: Claude-on-Bedrock id draws its Bedrock-owned pricing/availability from the
#: resource and its family caps/ceilings from the shared sources — no
#: duplication.
_BEDROCK_RESOURCE_SOURCE = BundledResourceSource.from_resource(
    "dataknobs_llm.llm.providers",
    "data/bedrock_models.yaml",
    name="bedrock_resource",
)
_BEDROCK_HEURISTIC_SOURCE = CallableModelMetadataSource("bedrock_heuristic", _bedrock_heuristic)


class BedrockConverseAdapter(LLMAdapter):
    """Adapter for the Amazon Bedrock Converse API format.

    Pure request/response mapping with no I/O — unit-testable without AWS.
    Converts between dataknobs standard types (``LLMMessage``,
    ``LLMResponse``, ``LLMConfig``) and Converse-specific shapes. Key
    Converse conventions handled here:

    - System content is a top-level ``system`` list of ``{"text": ...}``
      blocks, not part of the message list (like Anthropic's ``system``).
    - Assistant tool calls are ``toolUse`` content blocks.
    - Tool results are ``role="user"`` messages with ``toolResult`` content
      blocks, with consecutive tool results consolidated into one user
      message (Converse rejects consecutive same-role messages).
    """

    def adapt_messages(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Convert messages to Converse ``(system_blocks, messages)``.

        Args:
            messages: Standard ``LLMMessage`` list.
            system_prompt: Optional system prompt from provider config,
                prepended to any ``system`` messages found in the list.

        Returns:
            Tuple ``(system_blocks, converse_messages)`` — pass
            ``system_blocks`` as the ``system`` API parameter (when
            non-empty) and ``converse_messages`` as ``messages``.
        """
        system_blocks: list[dict[str, Any]] = []
        if system_prompt:
            system_blocks.append({"text": system_prompt})

        converse_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                system_blocks.append({"text": msg.content})
            elif msg.role == "assistant" and msg.tool_calls:
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": tc.id or tc.name,
                                "name": tc.name,
                                "input": tc.parameters,
                            }
                        }
                    )
                converse_messages.append(
                    {
                        "role": "assistant",
                        "content": content_blocks,
                    }
                )
            elif msg.role == "tool":
                # Converse expects tool results as user messages with
                # toolResult content blocks paired by toolUseId.
                # Consecutive tool results must be consolidated into a
                # single user message — the API rejects consecutive
                # messages with the same role.
                tool_use_id = msg.tool_call_id or msg.name or "unknown"
                result_block = {
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"text": msg.content}],
                    }
                }
                last = converse_messages[-1] if converse_messages else None
                if (
                    last is not None
                    and last["role"] == "user"
                    and isinstance(last["content"], list)
                    and last["content"]
                    and "toolResult" in last["content"][0]
                ):
                    last["content"].append(result_block)
                else:
                    converse_messages.append(
                        {
                            "role": "user",
                            "content": [result_block],
                        }
                    )
            else:
                converse_messages.append(
                    {
                        "role": msg.role,
                        "content": [{"text": msg.content}],
                    }
                )

        return system_blocks, converse_messages

    def adapt_config(self, config: LLMConfig) -> dict[str, Any]:
        """Build Converse ``modelId`` + ``inferenceConfig`` from config.

        Only explicitly-set generation parameters are emitted (Converse
        applies model defaults for the rest). Maps the canonical names to
        Converse's camelCase ``inferenceConfig`` keys.
        """
        gen = config.generation_params()
        inference_config: dict[str, Any] = {}
        if "temperature" in gen:
            inference_config["temperature"] = gen["temperature"]
        if "top_p" in gen:
            inference_config["topP"] = gen["top_p"]
        if "max_tokens" in gen:
            inference_config["maxTokens"] = gen["max_tokens"]
        if "stop_sequences" in gen:
            inference_config["stopSequences"] = gen["stop_sequences"]

        params: dict[str, Any] = {"modelId": config.model}
        if inference_config:
            params["inferenceConfig"] = inference_config
        return params

    def adapt_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert Tool objects to Converse ``toolSpec`` entries.

        Returns the list of ``{"toolSpec": {...}}`` entries; the provider
        wraps them in ``{"tools": [...]}`` for the ``toolConfig`` request
        field.
        """
        return [
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {"json": tool.schema if hasattr(tool, "schema") else {}},
                }
            }
            for tool in tools
        ]

    def adapt_response(self, response: dict[str, Any], model: str | None = None) -> LLMResponse:
        """Parse a Converse response dict into an ``LLMResponse``.

        Args:
            response: The ``converse`` response dict.
            model: The model id used for the request. The Converse response
                body does not echo it, so the provider supplies it.

        Returns:
            Standard ``LLMResponse`` with content, tool_calls, usage, and
            finish_reason. ``cost_usd`` is left ``None`` here — this adapter is
            pure/I-O-free and cannot resolve a model profile, so the provider
            stamps cost post-adapt from the resolved per-Mtok
            :class:`~..model_profile.ModelPricing` (see
            :meth:`BedrockProvider._cost_for`).
        """
        message = response.get("output", {}).get("message", {})
        content = ""
        raw_calls: list[tuple[str, Any, str | None]] = []
        for block in message.get("content", []):
            if "text" in block:
                content += block["text"]
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                raw_calls.append(
                    (tool_use.get("name", ""), tool_use.get("input"), tool_use.get("toolUseId"))
                )

        tool_calls = self.build_tool_calls(raw_calls)

        usage_raw = response.get("usage") or {}
        usage: dict[str, int] | None = None
        if usage_raw:
            usage = {
                "prompt_tokens": usage_raw.get("inputTokens", 0),
                "completion_tokens": usage_raw.get("outputTokens", 0),
                "total_tokens": usage_raw.get("totalTokens", 0),
            }

        # Bedrock Converse shares Claude's stopReason vocabulary verbatim
        # (Bedrock runs Claude), so finish_reason is normalized onto the
        # canonical tokens through the same shared helper as the native
        # Anthropic provider — the raw Converse stopReason is preserved on
        # metadata['raw_finish_reason']. stopReason == "max_tokens" is the
        # token-budget cut-off (same silent-truncation hazard as Anthropic).
        finish_reason, truncated, metadata = normalize_claude_stop_reason(
            response.get("stopReason")
        )

        return LLMResponse(
            content=content,
            model=model or "",
            finish_reason=finish_reason,
            truncated=truncated,
            usage=usage,
            tool_calls=tool_calls,
            metadata=metadata,
        )


# Default input type for Cohere embeddings when ``options["input_type"]`` is
# unset. ``search_document`` is the corpus/ingest side; query-time embeddings
# should pass ``search_query`` so retrieval scoring is not skewed (Cohere
# embeds the two asymmetrically).
_COHERE_DEFAULT_INPUT_TYPE = "search_document"


def _bool_option(options: dict[str, Any] | None, key: str, default: bool) -> bool:
    """Read a boolean ``options`` value, parsing strings correctly.

    ``bool("False")`` is ``True`` in Python (any non-empty string is truthy),
    so a raw ``bool()`` coercion of a string option is a footgun. This treats
    ``"false"``/``"0"``/``"no"``/``"off"`` (case-insensitive) as ``False`` and
    passes real bools through unchanged.
    """
    raw = (options or {}).get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(raw)


@overload
def _numeric_option(
    options: dict[str, Any] | None,
    key: str,
    default: float,
    cast: Callable[[Any], float],
) -> float: ...


@overload
def _numeric_option(
    options: dict[str, Any] | None,
    key: str,
    default: None,
    cast: Callable[[Any], float],
) -> float | None: ...


def _numeric_option(
    options: dict[str, Any] | None,
    key: str,
    default: float | None,
    cast: Callable[[Any], float],
) -> float | None:
    """Read a numeric ``options`` value or return ``default`` when unset.

    A present-but-uncoercible value (e.g. ``embed_max_concurrency: "auto"``)
    raises :class:`ConfigurationError` naming the option — the project
    convention — rather than a bare ``ValueError`` with no context.

    Overloaded on *default* because the two uses are different contracts and
    the union hid it: a caller passing a real default cannot receive ``None``,
    and three call sites were passing one to a parameter typed ``float``.
    """
    raw = (options or {}).get(key)
    if raw is None:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"Bedrock option {key!r} must be {cast.__name__}-coercible, got {raw!r}"
        ) from exc


# Embedding families: (model-id prefix, async embed function). Each function
# takes (client, model, texts, config, *, max_concurrency) and returns
# ``list[list[float]]``. Titan embeds one text per ``invoke_model`` call (a
# list is gathered under a concurrency bound); Cohere embeds the whole list in
# one call. Keeping the per-family body/parse shaping here — rather than
# branching inline in ``embed()`` — puts family knowledge in one place.


async def _embed_titan(
    client: Any,
    model: str,
    texts: list[str],
    config: LLMConfig,
    *,
    max_concurrency: int,
    dimensions: int | None = None,
) -> list[list[float]]:
    """Embed each text via a Titan ``invoke_model`` call (one per text).

    Titan has no batch endpoint, so a list of N texts issues N calls. They
    run concurrently but bounded by ``max_concurrency`` (an
    :class:`asyncio.Semaphore`) so a large ingest batch cannot fan out
    unbounded ``invoke_model`` calls and trip Bedrock throttling or exhaust
    the client's connection pool. ``normalize`` defaults to ``True`` and is
    overridable via ``options["normalize"]``.

    Args:
        dimensions: The width to request, already resolved from the call's
            ``dimensions=`` keyword or ``LLMConfig.dimensions`` by
            :meth:`~dataknobs_llm.llm.base.LLMProvider._requested_embedding_dimensions`.
            Taken as a parameter rather than read off *config* so the per-call
            keyword reaches the wire — reading the config here made the
            keyword unreachable on the one provider that forwarded anything.
    """
    normalize = _bool_option(config.options, "normalize", True)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _one(text: str) -> list[float]:
        body: dict[str, Any] = {"inputText": text, "normalize": normalize}
        if dimensions:
            body["dimensions"] = dimensions
        async with semaphore:
            result = await client.invoke_model(modelId=model, body=json.dumps(body))
            raw = await result["body"].read()
        parsed = json.loads(raw)
        vector: list[float] = parsed["embedding"]
        return vector

    return list(await asyncio.gather(*(_one(t) for t in texts)))


async def _embed_cohere(
    client: Any,
    model: str,
    texts: list[str],
    config: LLMConfig,
    *,
    max_concurrency: int,
    dimensions: int | None = None,
) -> list[list[float]]:
    """Embed the whole list via one Cohere ``invoke_model`` call.

    Cohere embeds the full batch in a single request, so ``max_concurrency``
    is accepted for a uniform family signature but unused. ``input_type``
    defaults to :data:`_COHERE_DEFAULT_INPUT_TYPE` and is overridable via
    ``options["input_type"]`` (e.g. ``"search_query"`` at query time).

    ``dimensions`` is likewise accepted for the uniform signature and unused:
    Cohere's Embed models have a fixed width and the invoke body has nowhere
    to put a request for another. A width stated anyway is not dropped —
    ``BedrockProvider.embed`` checks the returned vectors against it.
    """
    input_type = (config.options or {}).get("input_type", _COHERE_DEFAULT_INPUT_TYPE)
    body = {"texts": texts, "input_type": input_type}
    result = await client.invoke_model(modelId=model, body=json.dumps(body))
    raw = await result["body"].read()
    parsed = json.loads(raw)
    vectors: list[list[float]] = parsed["embeddings"]
    return vectors


_EMBED_FAMILIES: tuple[tuple[str, Any], ...] = (
    ("amazon.titan-embed", _embed_titan),
    ("cohere.embed", _embed_cohere),
)


class BedrockProvider(ProfileDetectionMixin, AsyncLLMProvider):
    """Amazon Bedrock LLM provider (Converse chat + Titan/Cohere embeddings).

    Authenticates via the AWS credential chain (IAM role, environment, or
    shared config) — no API key. Region, endpoint, explicit credentials,
    and Bedrock guardrail settings are supplied via ``LLMConfig.options``:

    - ``region_name`` (or ``region``): AWS region for the client.
    - ``endpoint_url``: custom endpoint (PrivateLink / VPC endpoint). This
      is the Bedrock endpoint knob — ``LLMConfig.api_base`` (an
      OpenAI/Anthropic-style base URL) is intentionally not consulted, since
      Bedrock addressing is region- and endpoint-resolved, not base-URL
      based.
    - ``aws_access_key_id`` / ``aws_secret_access_key`` /
      ``aws_session_token``: explicit credentials (omit to use the chain).
    - ``normalize`` (Titan embeddings, default ``True``) / ``input_type``
      (Cohere embeddings, default ``"search_document"``; use
      ``"search_query"`` at query time) / ``embed_max_concurrency`` (bound
      on Titan's per-text ``invoke_model`` fan-out; default
      ``max_pool_connections``).
    - ``stream_read_timeout``: per-socket-read (inter-chunk) timeout for
      ``stream_complete``, in seconds. Streaming has no total-duration knob
      in botocore, so ``LLMConfig.timeout`` (the whole-response budget used by
      ``complete``) is *not* applied to streaming — a long inter-token pause
      must not kill the stream. Defaults to boto's 60s read timeout; raise it
      for slow-thinking models.
    - ``guardrail_identifier`` / ``guardrail_version`` (+ optional
      ``guardrail_trace``): applied to Converse requests when both are set.

    The model id is a Bedrock foundation-model id (e.g.
    ``anthropic.claude-3-5-sonnet-20240620-v1:0``) or a cross-region
    inference-profile id (e.g.
    ``us.anthropic.claude-3-5-sonnet-20240620-v1:0``).

    Args:
        config: LLMConfig, dataknobs Config, or dict with provider settings.
        prompt_builder: Optional AsyncPromptBuilder for prompt rendering.

    Attributes:
        adapter (BedrockConverseAdapter): Converse format adapter.

    See Also:
        LLMConfig: Configuration options
        AsyncLLMProvider: Base provider interface
        BedrockConverseAdapter: Format conversion
    """

    def __init__(
        self,
        config: LLMConfig | Config | dict[str, Any],
        prompt_builder: AsyncPromptBuilder | None = None,
    ) -> None:
        llm_config = normalize_llm_config(config)
        super().__init__(llm_config, prompt_builder=prompt_builder)
        self.adapter = BedrockConverseAdapter()
        self._session: Any = None  # aioboto3.Session
        # Normalized AWS session config (region / credentials / endpoint /
        # retry+pool tuning) built once from LLMConfig.options and reused by
        # ``initialize`` (session build) and ``_client_kwargs`` (per-client
        # kwargs). Partial explicit credentials fail closed here at
        # construction via ``AwsSessionConfig.__post_init__``.
        self._session_config = AwsSessionConfig.from_dict(self.config.options)
        # Opt-in live-availability source (control-plane ListFoundationModels).
        # Off by default so an inference-only IAM role (no
        # ``bedrock:ListFoundationModels``) is never broken; when
        # ``options["model_availability_live"]`` is set, ``validate_model``
        # resolves availability against the account's live catalog. The live
        # source is consulted directly by ``validate_model`` (not composed into
        # ``_profile_resolver``) because "absent from the catalog" must resolve
        # to unavailable — a fact only the live cache can assert, and one the
        # per-facet resolver merge cannot express.
        self._availability_source: LiveApiSource | None = None
        if _bool_option(self.config.options, "model_availability_live", False):
            self._availability_source = LiveApiSource(
                self.list_foundation_models,
                _bedrock_availability_extractor,
                name="live_availability",
                enabled=True,
                model_id=_bedrock_summary_model_id,
                ttl=_numeric_option(self.config.options, "model_availability_ttl", 3600.0, float),
                refresh_timeout=_numeric_option(
                    self.config.options,
                    "model_availability_refresh_timeout",
                    10.0,
                    float,
                ),
            )

    # Same finding as ``AsyncLLMProvider.initialize`` one level up, and the
    # same answer: ``LLMProvider`` declares the pair sync, so the whole async
    # subtree contradicts its own base. Resolving it moves the pair down into
    # ``SyncLLMProvider`` --- a public-ABC contract change needing consumer
    # verification, argued and deferred where the base declares it. Suppressed
    # here for that decision, not because this override is wrong.
    async def initialize(self) -> None:  # type: ignore[override]
        """Build and cache the shared aioboto3 session for Bedrock.

        The session factory offloads construction off the event loop and
        warms a ``bedrock-runtime`` client so the first real client
        creation is a cache hit. No API-key check — Bedrock uses the AWS
        credential chain.

        Raises:
            ImportError: If the optional ``aioboto3`` dependency is missing.
        """
        # Probe the optional dependency up front so callers get an
        # actionable message rather than an opaque ImportError surfacing
        # from the session factory's worker thread.
        import importlib.util

        if importlib.util.find_spec("aioboto3") is None:
            raise ImportError(
                "aioboto3 is required for BedrockProvider. "
                "Install it with: pip install 'dataknobs-llm[bedrock]'"
            )

        self._session = await create_aioboto3_session(
            self._session_config, warm_service="bedrock-runtime"
        )
        self._is_initialized = True

    async def _close_client(self) -> None:
        """No-op — the session holds no open transport.

        Per-operation ``bedrock-runtime`` clients are short-lived async
        context managers closed at the end of each call; the cached session
        is process-wide and holds only botocore's loader caches, so there is
        nothing to close here.
        """

    def _client_kwargs(self, *, read_timeout: float | None = None) -> dict[str, Any]:
        """Per-client kwargs for a ``bedrock-runtime`` client from the session.

        Delegates to the shared
        :meth:`AwsSessionConfig.to_session_client_kwargs` builder so retry /
        pool tuning, ``endpoint_url`` + ``use_ssl`` inference, an explicit
        connect timeout, and the ``extra_client_kwargs`` passthrough all match
        every other AWS consumer (``SqsEventBus`` et al.) instead of a
        hand-rolled subset. Region and credentials ride on the session, so
        they are deliberately absent here.

        Args:
            read_timeout: The per-request generation budget
                (``LLMConfig.timeout``) applied as the socket read timeout
                (security rule 2). ``None`` defers to boto's default.
        """
        return self._session_config.to_session_client_kwargs(
            connect_timeout=_CONNECT_TIMEOUT_SECONDS,
            read_timeout=read_timeout,
        )

    def _stream_read_timeout(self) -> float | None:
        """Resolve the per-socket-read timeout for ``converse_stream``.

        botocore's ``read_timeout`` is a *per-read* (inter-chunk) timeout, not
        a total-stream-duration budget — and streaming has no total-duration
        knob. Reusing ``LLMConfig.timeout`` (the whole-response budget for
        ``complete``) here would kill a stream whenever the model pauses
        between tokens longer than that budget, so streaming is decoupled: the
        inter-chunk timeout comes from ``options["stream_read_timeout"]`` and
        defaults to ``None`` (boto's 60s default), which is a sane
        silence/stall detector independent of the generation budget.
        """
        return _numeric_option(self.config.options, "stream_read_timeout", None, float)

    @staticmethod
    def _guardrail_config(config: LLMConfig) -> dict[str, Any] | None:
        """Build Converse ``guardrailConfig`` from options, when configured.

        Returns ``None`` unless both ``guardrail_identifier`` and
        ``guardrail_version`` are present in ``config.options`` — additive,
        only applied when set.
        """
        opts = config.options or {}
        identifier = opts.get("guardrail_identifier")
        version = opts.get("guardrail_version")
        if not (identifier and version):
            return None
        guardrail: dict[str, Any] = {
            "guardrailIdentifier": identifier,
            "guardrailVersion": version,
        }
        if opts.get("guardrail_trace"):
            guardrail["trace"] = opts["guardrail_trace"]
        return guardrail

    def _build_converse_request(
        self,
        messages: str | list[LLMMessage],
        runtime_config: LLMConfig,
        tools: list[Any] | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the shared ``converse`` / ``converse_stream`` request kwargs.

        Shared by :meth:`complete` and :meth:`stream_complete` so the entry
        points differ only in ``converse`` vs
        ``converse_stream`` and buffered-vs-streamed delivery (no parameter
        drift). Delegates the request-shaping front-half to the base
        :meth:`~..base.LLMProvider._shape_request_params` (resolves constraints
        once, folds shaped per-call kwargs into the config, drops family-rejected
        sampling params, clamps ``max_tokens`` to the model ceiling — all in
        canonical config space), then assembles the Converse request and applies
        any wire-level :meth:`~..base.LLMProvider._apply_param_remaps` — the same
        clamp/drop the native Anthropic provider applies, since Bedrock runs the
        same Claude models. An unknown model resolves an all-permissive profile,
        so shaping is a pass-through for it (the historical behavior).

        The vendor-specific tail: the wire-only kwarg remainder is merged **last**
        via ``request.update(wire_extra)`` (after the full Converse assembly), so a
        genuine wire-only Converse param (``additionalModelRequestFields``, ...)
        lands as a top-level key. The fold-before-shape in the base closes the
        bypass where a raw ``max_tokens=`` kwarg would otherwise land as an
        un-clamped top-level Converse key (a hard ``converse()``
        ValidationException) instead of the shaped ``inferenceConfig.maxTokens``,
        and a ``temperature=`` kwarg on a Claude-5 id would bypass the
        rejected-param drop.
        """
        if isinstance(messages, str):
            msg_list = [LLMMessage(role="user", content=messages)]
        else:
            msg_list = list(messages)

        shaped_config, wire_extra, constraints = self._shape_request_params(runtime_config, extra)

        # NOTE: system_prompt / guardrail read the *unshaped* ``runtime_config``,
        # not ``shaped_config``. This is intentional and byte-identical: shaping
        # only ever folds/drops/clamps *shaped* fields (the union of
        # ``rejected_params``, ``param_remaps``, and ``max_tokens`` — all
        # sampling/output params), never ``system_prompt`` or any guardrail field,
        # so the two configs are equal on exactly the fields these two calls read.
        system_blocks, converse_messages = self.adapter.adapt_messages(
            msg_list, system_prompt=runtime_config.system_prompt
        )

        request = self.adapter.adapt_config(shaped_config)
        # Wire-level param renames (base mechanism), wired symmetric with the
        # OpenAI/Anthropic choke points. Applies only to *top-level* Converse
        # request keys: ``_apply_param_remaps`` renames a key present at the top
        # level, but Bedrock nests every sampling param under
        # ``request["inferenceConfig"]`` in Converse's normalized camelCase
        # (``maxTokens``/``topP``/``temperature``/``stopSequences``), so a
        # sampling-param remap does not reach — and does not need to reach — the
        # nested key. Unlike OpenAI (whose reasoning families genuinely rename
        # ``max_tokens`` -> ``max_completion_tokens`` at the same altitude),
        # Converse's ``inferenceConfig`` *is* the cross-family normalization
        # layer, so no Bedrock family declares a sampling-param remap. The call
        # is retained for symmetry and to honor any remap targeting a genuine
        # top-level Converse key; today it is a uniform no-op.
        request = self._apply_param_remaps(request, constraints.param_remaps)
        request["messages"] = converse_messages
        if system_blocks:
            request["system"] = system_blocks
        if tools:
            request["toolConfig"] = {"tools": self.adapter.adapt_tools(tools)}
        guardrail = self._guardrail_config(runtime_config)
        if guardrail:
            request["guardrailConfig"] = guardrail
        request.update(wire_extra)
        return request

    def _profile_resolver(self, config: LLMConfig) -> LayeredModelProfileResolver:
        """Compose the Bedrock model-profile resolver for *config*.

        Precedence (highest first): config override → the Bedrock bundled
        resource (non-Claude full profiles + Claude-on-Bedrock
        pricing/availability) → the SHARED Claude ceiling source → the SHARED
        Claude capability/rejected-param source → the Bedrock last-resort
        capability/availability heuristic. Per facet, first non-``None`` wins,
        so a Claude-on-Bedrock id draws its Bedrock-owned
        pricing/availability from the Bedrock resource and its
        ceilings/capabilities/rejected-params from the shared Claude sources —
        zero duplication — while a non-Claude id draws everything from the
        Bedrock resource (heuristic last-resort). ``LLMConfig.model_profile_overrides``
        wins over all of them per facet.

        The opt-in live-availability source is deliberately **not** composed
        here — it is consulted directly by :meth:`validate_model` (see the
        ``__init__`` note), because "absent from the account catalog →
        unavailable" is a fact the per-facet merge cannot express (an absent
        model yields no partial, so a lower-precedence permissive ``available``
        would win). The live source contributes only the ``available`` facet,
        which no other resolver consumer reads.
        """
        return LayeredModelProfileResolver(
            [
                ConfigOverrideSource(getattr(config, "model_profile_overrides", None)),
                _BEDROCK_RESOURCE_SOURCE,
                CLAUDE_RESOURCE_PROFILE_SOURCE,
                CLAUDE_ONLY_HEURISTIC_PROFILE_SOURCE,
                _BEDROCK_HEURISTIC_SOURCE,
            ]
        )

    async def list_foundation_models(self) -> list[dict[str, Any]]:
        """Fetch the account's Bedrock foundation-model catalog (control-plane).

        Uses a ``bedrock`` **control-plane** client — distinct from the
        ``bedrock-runtime`` clients used for inference — so it requires the
        ``bedrock:ListFoundationModels`` permission. Passed to the provider's
        :class:`~..model_profile.LiveApiSource` as its ``list_models`` callable
        when ``options["model_availability_live"]`` is set; the source drives it
        out-of-band (TTL-gated, per-loop-locked). ``ListFoundationModels``
        returns the full catalog in one call (no pagination) and serves
        availability + modalities, but not token ceilings.
        """
        if not self._is_initialized:
            await self.initialize()
        async with self._session.client(
            "bedrock", **self._client_kwargs(read_timeout=self.config.timeout)
        ) as client:
            response = await client.list_foundation_models()
        return list(response.get("modelSummaries", []))

    async def validate_model(self) -> bool:
        """Validate the configured model against data-sourced availability.

        Default (no ``options["model_availability_live"]``): reads the
        ``available`` facet off the resolved model profile — ``True`` for a
        model listed in ``bedrock_models.yaml`` or under a recognised vendor
        prefix (the old permissive-prefix behavior, now data-sourced), ``False``
        for an unknown vendor. **No control-plane call**, so an inference-only
        IAM role keeps working (``bedrock:ListFoundationModels`` is a separate
        permission).

        Opt-in (``options["model_availability_live"]=true``): resolves
        availability against the account's live ``ListFoundationModels`` catalog
        — a model absent from the account is ``False``, so an entitlement gap is
        caught at validation time. The live source carries the TTL / per-loop
        lock / non-degradation refresh from the shared
        :class:`~..model_profile.LiveApiSource`.
        """
        canonical = _canonical_model_id(self.config.model)
        if self._availability_source is not None:
            if not self._is_initialized:
                await self.initialize()
            await self._availability_source.refresh_if_stale()
            # The live cache holds only listed (available) models, so an absent
            # model resolves ``available=None`` → ``False`` — the authoritative
            # "not in this account" answer the maintained resource cannot give.
            return bool(self._availability_source.resolve(canonical).available)
        return bool(self._profile_resolver(self.config).resolve(canonical).available)

    def _profile_lookup_key(self, config: LLMConfig) -> str:
        """Resolve profiles under the canonical, region-stripped model id.

        A Bedrock id may carry a cross-region inference-profile prefix
        (``us.``/``eu.``/``apac.``) that is not part of the model family. Stripping
        it via :func:`_canonical_model_id` before the catalog lookup makes a
        cross-region id resolve the same profile (capabilities / ceiling /
        pricing) as its base id — so the shared
        :class:`~..profile_detection.ProfileDetectionMixin` trio reads the right
        family for every regional variant.
        """
        return _canonical_model_id(config.model)

    def _cost_for(self, model: str, usage: dict[str, int] | None) -> float | None:
        """Provider-side USD cost from resolved per-Mtok pricing (post-adapt).

        Keeps :class:`BedrockConverseAdapter` pure/I-O-free (it cannot resolve a
        profile): the provider stamps ``cost_usd`` after ``adapt_response`` via
        the base :meth:`~..base.LLMProvider.estimate_cost` →
        :meth:`~..base.LLMProvider.get_pricing` →
        :class:`~..utils.CostCalculator` path. ``None`` when the model has no
        profile pricing or the response carries no usage. Shared by the
        buffered (:meth:`complete`) and streaming
        (:meth:`stream_complete` final chunk) paths so cost is computed
        identically on both.
        """
        if not usage:
            return None
        return self.estimate_cost(LLMResponse(content="", model=model, usage=usage), model=model)

    def _translate_api_error(self, exc: Exception) -> Exception | None:
        """Translate a raw botocore error into a dataknobs exception.

        Lets consumers catch by a dataknobs exception type instead of coupling
        to ``botocore``. Bedrock's status lives *nested* in a
        ``ClientError.response`` dict (``["ResponseMetadata"]["HTTPStatusCode"]``);
        the throttling *codes* (``ThrottlingException`` /
        ``TooManyRequestsException``) are normalized to 429 even when the HTTP
        status is ambiguous. A ``BotoCoreError`` (connection / endpoint /
        read-timeout — no HTTP status) maps to ``OperationError``. The
        status→type policy is deferred to
        :meth:`~dataknobs_llm.llm.base.LLMProvider._dataknobs_error_for_status`
        (429 → ``RateLimitError``, 400 → ``ValidationError``, else →
        ``OperationError``). Bedrock does not surface a ``retry-after`` header on
        the exception, so ``retry_after`` stays ``None``.

        Returns ``None`` for a non-botocore exception so the caller re-raises it
        unchanged. The original error is preserved on ``__cause__`` — callers
        raise ``... from exc``.
        """
        try:
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError:  # pragma: no cover - botocore installed post-init
            return None
        if isinstance(exc, ClientError):
            response = getattr(exc, "response", None) or {}
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = response.get("Error", {}).get("Code", "")
            if code in ("ThrottlingException", "TooManyRequestsException"):
                status = 429
            return self._dataknobs_error_for_status(status, str(exc))
        if isinstance(exc, BotoCoreError):
            return self._dataknobs_error_for_status(None, str(exc))
        return None

    async def complete(
        self,
        messages: str | list[LLMMessage],
        config_overrides: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion via the Converse API.

        Args:
            messages: Input prompt or message list.
            config_overrides: Optional per-request config overrides.
            tools: Optional list of Tool objects for tool use.
            **kwargs: Additional Converse request parameters. Routed through the
                request-shaping choke point (:meth:`_build_converse_request`): a
                kwarg naming a shaped ``LLMConfig`` field is clamped/dropped/
                remapped like a config override; a genuine wire-only Converse
                param passes straight through.
        """
        if not self._is_initialized:
            await self.initialize()

        runtime_config = self._get_runtime_config(config_overrides)
        request = self._build_converse_request(messages, runtime_config, tools, extra=kwargs)

        start = time.perf_counter()
        async with self._session.client(
            "bedrock-runtime",
            **self._client_kwargs(read_timeout=runtime_config.timeout),
        ) as client:
            try:
                response = await client.converse(**request)
            except Exception as exc:
                self._raise_translated(exc)

        parsed = self.adapter.adapt_response(response, model=runtime_config.model)
        # Stamp cost post-adapt from the resolved per-Mtok profile pricing —
        # the pure adapter cannot resolve a profile.
        parsed.cost_usd = self._cost_for(runtime_config.model, parsed.usage)
        result = self._analyze_response(parsed)
        logger.debug(
            "Bedrock converse complete (model=%s, finish=%s, tokens=%s, latency_ms=%d)",
            runtime_config.model,
            result.finish_reason,
            (result.usage or {}).get("total_tokens"),
            int((time.perf_counter() - start) * 1000),
        )
        return result

    async def stream_complete(
        self,
        messages: str | list[LLMMessage],
        config_overrides: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamResponse]:
        """Generate a streaming completion via ``converse_stream``.

        Yields incremental text deltas as they arrive and one final
        ``LLMStreamResponse`` (``is_final=True``) carrying finish_reason,
        accumulated tool calls, usage, and the model id. The entire
        generator runs inside the client context manager so the event
        stream is fully consumed before the client closes.

        Args:
            messages: Input prompt or message list.
            config_overrides: Optional per-request config overrides.
            tools: Optional list of Tool objects for tool use.
            **kwargs: Additional Converse request parameters. Routed through the
                request-shaping choke point (:meth:`_build_converse_request`), as
                in :meth:`complete`.
        """
        if not self._is_initialized:
            await self.initialize()

        runtime_config = self._get_runtime_config(config_overrides)
        request = self._build_converse_request(messages, runtime_config, tools, extra=kwargs)

        logger.debug("Bedrock converse_stream start (model=%s)", runtime_config.model)
        stream_start = time.perf_counter()
        async with self._session.client(
            "bedrock-runtime",
            **self._client_kwargs(read_timeout=self._stream_read_timeout()),
        ) as client:
            try:
                response = await client.converse_stream(**request)
            except Exception as exc:
                self._raise_translated(exc)

            # Accumulate partial-JSON tool inputs per content-block index,
            # mirroring OpenAI's streamed tool-call accumulation.
            tool_accumulators: dict[int, dict[str, Any]] = {}
            stop_reason: str | None = None
            usage: dict[str, int] | None = None

            # Iterate through the translating wrapper so a vendor error
            # surfacing mid-stream (throttle, connection drop) is translated
            # too — not just the converse_stream() create above.
            async for event in self._iter_translated(response["stream"]):
                if "contentBlockStart" in event:
                    start = event["contentBlockStart"]
                    idx = start.get("contentBlockIndex", 0)
                    tool_use = start.get("start", {}).get("toolUse")
                    if tool_use:
                        tool_accumulators[idx] = {
                            "id": tool_use.get("toolUseId"),
                            "name": tool_use.get("name", ""),
                            "input": "",
                        }
                elif "contentBlockDelta" in event:
                    block = event["contentBlockDelta"]
                    idx = block.get("contentBlockIndex", 0)
                    delta = block.get("delta", {})
                    if "text" in delta:
                        yield LLMStreamResponse(delta=delta["text"], is_final=False)
                    elif "toolUse" in delta and idx in tool_accumulators:
                        tool_accumulators[idx]["input"] += delta["toolUse"].get("input", "")
                elif "messageStop" in event:
                    stop_reason = event["messageStop"].get("stopReason")
                elif "metadata" in event:
                    usage_raw = event["metadata"].get("usage")
                    if usage_raw:
                        usage = {
                            "prompt_tokens": usage_raw.get("inputTokens", 0),
                            "completion_tokens": usage_raw.get("outputTokens", 0),
                            "total_tokens": usage_raw.get("totalTokens", 0),
                        }

            tool_calls = self.adapter.build_tool_calls(
                (acc["name"], acc["input"], acc["id"])
                for _, acc in sorted(tool_accumulators.items())
            )

            logger.debug(
                "Bedrock converse_stream done (model=%s, finish=%s, tokens=%s, latency_ms=%d)",
                runtime_config.model,
                stop_reason,
                (usage or {}).get("total_tokens"),
                int((time.perf_counter() - stream_start) * 1000),
            )
            # Normalize onto the canonical finish_reason vocabulary through the
            # shared Claude helper, so the streaming final chunk matches the
            # buffered path (and the native Anthropic stream, which is already
            # canonical because it is built from adapt_response).
            finish_reason, truncated, _ = normalize_claude_stop_reason(stop_reason)
            final_chunk = LLMStreamResponse(
                delta="",
                is_final=True,
                finish_reason=finish_reason,
                truncated=truncated,
                tool_calls=tool_calls,
                usage=usage,
                model=runtime_config.model,
                # Stamp cost on the final chunk from the resolved per-Mtok
                # profile pricing — the buffered path stamps LLMResponse.cost_usd
                # the same way; the stream previously carried no cost at all.
                cost_usd=self._cost_for(runtime_config.model, usage),
            )
            self._warn_if_truncated(final_chunk)
            yield final_chunk

    async def embed(
        self,
        texts: str | list[str],
        **kwargs: Any,
    ) -> list[float] | list[list[float]]:
        """Generate embeddings via ``invoke_model`` (Titan / Cohere).

        Returns a single vector for a ``str`` input and a list of vectors
        for a list input, per the base contract.

        A stated width is forwarded where the family accepts one (Titan takes
        ``dimensions`` in the invoke body) and checked where it does not
        (Cohere's width is the model's). Either way it is not ignored.

        Args:
            texts: A single text or a batch.
            **kwargs: ``dimensions`` (int) overrides ``LLMConfig.dimensions``
                for this call.

        Raises:
            ValueError: If the model id does not match a supported embedding
                family (Titan ``amazon.titan-embed*`` or Cohere
                ``cohere.embed*``), or if the vectors are not the width that
                was asked for.
        """
        if not self._is_initialized:
            await self.initialize()

        single = isinstance(texts, str)
        text_list = [texts] if isinstance(texts, str) else list(texts)

        model = self.config.model
        canonical = _canonical_model_id(model)
        embed_fn = None
        for prefix, fn in _EMBED_FAMILIES:
            if canonical.startswith(prefix):
                embed_fn = fn
                break
        if embed_fn is None:
            raise ValueError(
                f"Unsupported Bedrock embedding model: {model!r}. Supported "
                "families: Titan ('amazon.titan-embed*') and Cohere "
                "('cohere.embed*')."
            )

        max_concurrency = self._embed_max_concurrency()
        requested = self._requested_embedding_dimensions(kwargs)
        # Titan V2 takes a width; V1 and Cohere do not. Forward only what the
        # model can be asked for -- the rest is caught on the way back out.
        forwardable = self._forwardable_embedding_dimensions(kwargs)
        start = time.perf_counter()
        async with self._session.client(
            "bedrock-runtime",
            **self._client_kwargs(read_timeout=self.config.timeout),
        ) as client:
            vectors: list[list[float]]
            try:
                vectors = await embed_fn(
                    client,
                    model,
                    text_list,
                    self.config,
                    max_concurrency=max_concurrency,
                    dimensions=forwardable,
                )
            except Exception as exc:
                self._raise_translated(exc)
        self._check_embedding_width(vectors, requested)

        logger.debug(
            "Bedrock embed complete (model=%s, count=%d, latency_ms=%d)",
            model,
            len(text_list),
            int((time.perf_counter() - start) * 1000),
        )
        return vectors[0] if single else vectors

    def _embed_max_concurrency(self) -> int:
        """Resolve the max concurrent ``invoke_model`` calls for embeddings.

        Defaults to the session's ``max_pool_connections`` (no point issuing
        more concurrent requests than the connection pool can carry) and is
        overridable via ``options["embed_max_concurrency"]``. Floored at 1.
        Bounds Titan's per-text fan-out (see :func:`_embed_titan`).
        """
        limit = _numeric_option(
            self.config.options,
            "embed_max_concurrency",
            self._session_config.max_pool_connections,
            int,
        )
        return max(1, int(limit))
