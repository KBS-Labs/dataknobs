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
import warnings
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, NoReturn

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
    ToolCall,
    normalize_claude_stop_reason,
    normalize_llm_config,
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

# Prefixes recognised as valid Bedrock model ids / inference profiles by the
# heuristic ``validate_model`` (no control-plane call — avoids needing
# ``bedrock:ListFoundationModels`` on the task role).
_KNOWN_MODEL_PREFIXES: tuple[str, ...] = (
    "amazon.",
    "anthropic.",
    "meta.",
    "mistral.",
    "cohere.",
    "ai21.",
    "us.",
    "eu.",
    "apac.",
    "us-gov.",
)

# Best-effort, deliberately incomplete static price map
# (``model_id -> (input_usd_per_1k, output_usd_per_1k)``) for computing
# ``LLMResponse.cost_usd``. Never gates behaviour — an absent model leaves
# ``cost_usd`` as ``None``. Prices are a convenience estimate only.
_MODEL_PRICE_USD: dict[str, tuple[float, float]] = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0": (0.003, 0.015),
    "anthropic.claude-3-sonnet-20240229-v1:0": (0.003, 0.015),
    "anthropic.claude-3-haiku-20240307-v1:0": (0.00025, 0.00125),
    "anthropic.claude-3-opus-20240229-v1:0": (0.015, 0.075),
    "amazon.nova-lite-v1:0": (0.00006, 0.00024),
    "amazon.nova-pro-v1:0": (0.0008, 0.0032),
}


def _canonical_model_id(model: str) -> str:
    """Strip a leading region / inference-profile prefix from a model id.

    ``us.anthropic.claude-...`` -> ``anthropic.claude-...``. Used so family
    and capability detection work uniformly for both plain model ids and
    cross-region inference-profile ids.
    """
    for prefix in _REGION_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def _estimate_cost(
    model: str, usage: dict[str, int] | None
) -> float | None:
    """Best-effort USD cost from token usage and the static price map.

    Returns ``None`` when the model is not in :data:`_MODEL_PRICE_USD` or
    usage is missing — cost is a convenience estimate, never load-bearing.
    """
    if not usage:
        return None
    price = _MODEL_PRICE_USD.get(_canonical_model_id(model))
    if price is None:
        return None
    in_per_1k, out_per_1k = price
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    return (prompt_tokens / 1000.0) * in_per_1k + (
        completion_tokens / 1000.0
    ) * out_per_1k


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
                    content_blocks.append({
                        "toolUse": {
                            "toolUseId": tc.id or tc.name,
                            "name": tc.name,
                            "input": tc.parameters,
                        }
                    })
                converse_messages.append({
                    "role": "assistant",
                    "content": content_blocks,
                })
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
                    converse_messages.append({
                        "role": "user",
                        "content": [result_block],
                    })
            else:
                converse_messages.append({
                    "role": msg.role,
                    "content": [{"text": msg.content}],
                })

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
                    "inputSchema": {
                        "json": tool.schema if hasattr(tool, "schema") else {}
                    },
                }
            }
            for tool in tools
        ]

    def adapt_raw_functions(
        self, functions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert raw function dicts to Converse ``toolSpec`` entries.

        Used by the deprecated :meth:`BedrockProvider.function_call` which
        receives raw dicts rather than Tool objects.
        """
        return [
            {
                "toolSpec": {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "inputSchema": {
                        "json": func.get("parameters", {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        })
                    },
                }
            }
            for func in functions
        ]

    def adapt_response(
        self, response: dict[str, Any], model: str | None = None
    ) -> LLMResponse:
        """Parse a Converse response dict into an ``LLMResponse``.

        Args:
            response: The ``converse`` response dict.
            model: The model id used for the request. The Converse response
                body does not echo it, so the provider supplies it (also
                used for best-effort cost estimation).

        Returns:
            Standard ``LLMResponse`` with content, tool_calls, usage,
            finish_reason, and best-effort ``cost_usd``.
        """
        message = response.get("output", {}).get("message", {})
        content = ""
        tool_calls: list[ToolCall] = []
        for block in message.get("content", []):
            if "text" in block:
                content += block["text"]
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                tool_input = tool_use.get("input")
                tool_calls.append(ToolCall(
                    name=tool_use.get("name", ""),
                    parameters=tool_input if isinstance(tool_input, dict) else {},
                    id=tool_use.get("toolUseId"),
                ))

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
            tool_calls=tool_calls or None,
            metadata=metadata,
            cost_usd=_estimate_cost(model or "", usage),
        )


# Default input type for Cohere embeddings when ``options["input_type"]`` is
# unset. ``search_document`` is the corpus/ingest side; query-time embeddings
# should pass ``search_query`` so retrieval scoring is not skewed (Cohere
# embeds the two asymmetrically).
_COHERE_DEFAULT_INPUT_TYPE = "search_document"


def _bool_option(
    options: dict[str, Any] | None, key: str, default: bool
) -> bool:
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
    """
    raw = (options or {}).get(key)
    if raw is None:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"Bedrock option {key!r} must be {cast.__name__}-coercible, "
            f"got {raw!r}"
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
) -> list[list[float]]:
    """Embed each text via a Titan ``invoke_model`` call (one per text).

    Titan has no batch endpoint, so a list of N texts issues N calls. They
    run concurrently but bounded by ``max_concurrency`` (an
    :class:`asyncio.Semaphore`) so a large ingest batch cannot fan out
    unbounded ``invoke_model`` calls and trip Bedrock throttling or exhaust
    the client's connection pool. ``normalize`` defaults to ``True`` and is
    overridable via ``options["normalize"]``.
    """
    dimensions = config.dimensions
    normalize = _bool_option(config.options, "normalize", True)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _one(text: str) -> list[float]:
        body: dict[str, Any] = {"inputText": text, "normalize": normalize}
        if dimensions:
            body["dimensions"] = dimensions
        async with semaphore:
            result = await client.invoke_model(
                modelId=model, body=json.dumps(body)
            )
            raw = await result["body"].read()
        parsed = json.loads(raw)
        return parsed["embedding"]

    return list(await asyncio.gather(*(_one(t) for t in texts)))


async def _embed_cohere(
    client: Any,
    model: str,
    texts: list[str],
    config: LLMConfig,
    *,
    max_concurrency: int,
) -> list[list[float]]:
    """Embed the whole list via one Cohere ``invoke_model`` call.

    Cohere embeds the full batch in a single request, so ``max_concurrency``
    is accepted for a uniform family signature but unused. ``input_type``
    defaults to :data:`_COHERE_DEFAULT_INPUT_TYPE` and is overridable via
    ``options["input_type"]`` (e.g. ``"search_query"`` at query time).
    """
    input_type = (config.options or {}).get(
        "input_type", _COHERE_DEFAULT_INPUT_TYPE
    )
    body = {"texts": texts, "input_type": input_type}
    result = await client.invoke_model(modelId=model, body=json.dumps(body))
    raw = await result["body"].read()
    parsed = json.loads(raw)
    return parsed["embeddings"]


_EMBED_FAMILIES: tuple[tuple[str, Any], ...] = (
    ("amazon.titan-embed", _embed_titan),
    ("cohere.embed", _embed_cohere),
)


class BedrockProvider(AsyncLLMProvider):
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

    async def initialize(self) -> None:
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

    def _client_kwargs(
        self, *, read_timeout: float | None = None
    ) -> dict[str, Any]:
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
        return _numeric_option(
            self.config.options, "stream_read_timeout", None, float
        )

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
    ) -> dict[str, Any]:
        """Build the shared ``converse`` / ``converse_stream`` request kwargs.

        Shared by :meth:`complete` and :meth:`stream_complete` so the two
        methods differ only in ``converse`` vs ``converse_stream`` and
        buffered-vs-streamed delivery (no parameter drift).
        """
        if isinstance(messages, str):
            msg_list = [LLMMessage(role="user", content=messages)]
        else:
            msg_list = list(messages)

        system_blocks, converse_messages = self.adapter.adapt_messages(
            msg_list, system_prompt=runtime_config.system_prompt
        )

        request = self.adapter.adapt_config(runtime_config)
        request["messages"] = converse_messages
        if system_blocks:
            request["system"] = system_blocks
        if tools:
            request["toolConfig"] = {"tools": self.adapter.adapt_tools(tools)}
        guardrail = self._guardrail_config(runtime_config)
        if guardrail:
            request["guardrailConfig"] = guardrail
        return request

    async def validate_model(self) -> bool:
        """Heuristic model-id validation (no control-plane call).

        Returns ``True`` for a recognised foundation-model or
        inference-profile prefix. Deliberately does not call
        ``ListFoundationModels`` so the task role needs only inference
        permissions.
        """
        return self.config.model.startswith(_KNOWN_MODEL_PREFIXES)

    def _detect_capabilities(self) -> list[ModelCapability]:
        """Auto-detect Bedrock model capabilities from the model id.

        Embedding models advertise **only** ``EMBEDDINGS`` — they cannot
        chat, stream, or call tools, so reporting those would let
        capability-driven routing send a chat request to an embed-only
        model. Chat / generation models advertise text generation, chat,
        streaming, and function calling (plus ``VISION`` for multimodal
        Claude 3+ / Nova).
        """
        model = _canonical_model_id(self.config.model.lower())

        if any(
            token in model
            for token in ("titan-embed", "cohere.embed", "-embed-")
        ):
            return [ModelCapability.EMBEDDINGS]

        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.FUNCTION_CALLING,
        ]
        # Multimodal Claude 3+ / Nova models accept image content.
        if any(
            token in model
            for token in (
                "claude-3",
                "claude-sonnet",
                "claude-opus",
                "claude-haiku",
                "nova",
            )
        ):
            capabilities.append(ModelCapability.VISION)
        return capabilities

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
            return self._dataknobs_error_for_status(
                status, f"Bedrock API error: {exc}"
            )
        if isinstance(exc, BotoCoreError):
            return self._dataknobs_error_for_status(
                None, f"Bedrock API error: {exc}"
            )
        return None

    def _raise_translated(self, exc: Exception) -> NoReturn:
        """Raise the dataknobs translation of *exc*, else re-raise it unchanged.

        The S4 choke point for the ``converse`` / ``converse_stream`` /
        ``invoke_model`` call sites. A non-botocore error is re-raised as-is; a
        botocore error is raised as its dataknobs type ``from`` the original.
        """
        translated = self._translate_api_error(exc)
        if translated is None:
            raise exc
        raise translated from exc

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
            **kwargs: Additional Converse request parameters (merged in).
        """
        if not self._is_initialized:
            await self.initialize()

        runtime_config = self._get_runtime_config(config_overrides)
        request = self._build_converse_request(messages, runtime_config, tools)
        request.update(kwargs)

        start = time.perf_counter()
        async with self._session.client(
            "bedrock-runtime",
            **self._client_kwargs(read_timeout=runtime_config.timeout),
        ) as client:
            try:
                response = await client.converse(**request)
            except Exception as exc:
                self._raise_translated(exc)

        result = self._analyze_response(
            self.adapter.adapt_response(response, model=runtime_config.model)
        )
        logger.debug(
            "Bedrock converse complete (model=%s, finish=%s, tokens=%s, "
            "latency_ms=%d)",
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
            **kwargs: Additional Converse request parameters (merged in).
        """
        if not self._is_initialized:
            await self.initialize()

        runtime_config = self._get_runtime_config(config_overrides)
        request = self._build_converse_request(messages, runtime_config, tools)
        request.update(kwargs)

        logger.debug(
            "Bedrock converse_stream start (model=%s)", runtime_config.model
        )
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

            async for event in response["stream"]:
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
                        yield LLMStreamResponse(
                            delta=delta["text"], is_final=False
                        )
                    elif "toolUse" in delta and idx in tool_accumulators:
                        tool_accumulators[idx]["input"] += delta[
                            "toolUse"
                        ].get("input", "")
                elif "messageStop" in event:
                    stop_reason = event["messageStop"].get("stopReason")
                elif "metadata" in event:
                    usage_raw = event["metadata"].get("usage")
                    if usage_raw:
                        usage = {
                            "prompt_tokens": usage_raw.get("inputTokens", 0),
                            "completion_tokens": usage_raw.get(
                                "outputTokens", 0
                            ),
                            "total_tokens": usage_raw.get("totalTokens", 0),
                        }

            tool_calls: list[ToolCall] | None = None
            if tool_accumulators:
                tool_calls = [
                    ToolCall(
                        name=acc["name"],
                        parameters=(
                            json.loads(acc["input"]) if acc["input"] else {}
                        ),
                        id=acc["id"],
                    )
                    for _, acc in sorted(tool_accumulators.items())
                ]

            logger.debug(
                "Bedrock converse_stream done (model=%s, finish=%s, "
                "tokens=%s, latency_ms=%d)",
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

        Raises:
            ValueError: If the model id does not match a supported embedding
                family (Titan ``amazon.titan-embed*`` or Cohere
                ``cohere.embed*``).
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
        start = time.perf_counter()
        async with self._session.client(
            "bedrock-runtime",
            **self._client_kwargs(read_timeout=self.config.timeout),
        ) as client:
            try:
                vectors = await embed_fn(
                    client,
                    model,
                    text_list,
                    self.config,
                    max_concurrency=max_concurrency,
                )
            except Exception as exc:
                self._raise_translated(exc)

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

    async def function_call(
        self,
        messages: list[LLMMessage],
        functions: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute function calling via Converse tools (deprecated).

        Retained for interface parity with the other providers. Prefer
        ``complete(tools=...)``.
        """
        warnings.warn(
            "function_call() is deprecated, use complete(tools=...) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self._is_initialized:
            await self.initialize()

        runtime_config = self._get_runtime_config(None)
        request = self._build_converse_request(messages, runtime_config, None)
        request["toolConfig"] = {
            "tools": self.adapter.adapt_raw_functions(functions)
        }

        async with self._session.client(
            "bedrock-runtime",
            **self._client_kwargs(read_timeout=runtime_config.timeout),
        ) as client:
            try:
                response = await client.converse(**request)
            except Exception as exc:
                self._raise_translated(exc)

        parsed = self.adapter.adapt_response(
            response, model=runtime_config.model
        )

        # Surface the first tool call as the legacy function_call dict and route
        # through the shared _analyze_response choke point, so a truncated
        # tool-call turn on this path fires the truncation warning — exactly
        # like complete().
        return self._analyze_response(
            self._attach_legacy_function_call(parsed)
        )
