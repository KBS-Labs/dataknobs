"""Anthropic Claude LLM provider implementation.

This module provides Anthropic Claude API integration for dataknobs-llm, supporting:
- Claude 3 (Opus, Sonnet, Haiku) and Claude 2 models
- Native tools API for function calling
- Vision capabilities (Claude 3+)
- Streaming responses
- Long context windows (up to 200k tokens)
- Advanced reasoning and coding capabilities

The AnthropicProvider uses the official Anthropic Python SDK and supports
all standard Anthropic API parameters including system prompts, temperature,
and token limits.

Example:
    ```python
    from dataknobs_llm.llm.providers import AnthropicProvider
    from dataknobs_llm.llm.base import LLMConfig

    # Create provider
    config = LLMConfig(
        provider="anthropic",
        model="claude-3-sonnet-20240229",
        api_key="sk-ant-...",  # or set ANTHROPIC_API_KEY env var
        temperature=0.7,
        max_tokens=1024
    )

    async with AnthropicProvider(config) as llm:
        # Simple completion
        response = await llm.complete("Explain quantum computing")
        print(response.content)

        # Streaming for real-time output
        async for chunk in llm.stream_complete("Write a story"):
            print(chunk.delta, end="", flush=True)

        # Tool use (Claude 3+)
        tools = [{
            "name": "calculator",
            "description": "Perform arithmetic",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"}
                }
            }
        }]

        response = await llm.function_call(messages, tools)
    ```

See Also:
    - Anthropic API Documentation: https://docs.anthropic.com/
    - anthropic Python package: https://github.com/anthropics/anthropic-sdk-python
"""

import os
import json
from typing import TYPE_CHECKING, Any, Dict, List, Union, AsyncIterator

from ..base import (
    LLMConfig, LLMMessage, LLMResponse, LLMStreamResponse,
    AsyncLLMProvider, ModelCapability,
    normalize_llm_config
)
from dataknobs_llm.prompts import AsyncPromptBuilder

if TYPE_CHECKING:
    from dataknobs_config.config import Config


class AnthropicProvider(AsyncLLMProvider):
    r"""Anthropic Claude LLM provider with full API support.

    Provides async access to Anthropic's Claude models including Claude 3
    (Opus, Sonnet, Haiku) and Claude 2. Supports advanced features like
    native tool use, vision, and extended context windows.

    Features:
        - Claude 3 Opus/Sonnet/Haiku and Claude 2 models
        - Native tools API for function calling (Claude 3+)
        - Vision capabilities for image understanding (Claude 3+)
        - Streaming responses for real-time output
        - Long context windows (up to 200k tokens)
        - Advanced reasoning and coding capabilities
        - System prompts for behavior control
        - JSON output mode

    Example:
        ```python
        from dataknobs_llm.llm.providers import AnthropicProvider
        from dataknobs_llm.llm.base import LLMConfig, LLMMessage

        # Basic usage
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-sonnet-20240229",
            api_key="sk-ant-...",
            temperature=0.7,
            max_tokens=1024
        )

        async with AnthropicProvider(config) as llm:
            # Simple completion
            response = await llm.complete("Explain machine learning")
            print(response.content)

            # With system prompt
            messages = [
                LLMMessage(
                    role="system",
                    content="You are an expert Python tutor"
                ),
                LLMMessage(
                    role="user",
                    content="How do I use decorators?"
                )
            ]
            response = await llm.complete(messages)

        # Long context processing (Claude 3+)
        long_config = LLMConfig(
            provider="anthropic",
            model="claude-3-opus-20240229",
            max_tokens=4096
        )

        llm = AnthropicProvider(long_config)
        await llm.initialize()

        # Process large document
        with open("large_doc.txt") as f:
            long_text = f.read()  # Up to 200k tokens!

        response = await llm.complete(
            f"Summarize this document:\n\n{long_text}"
        )

        # Tool use / function calling (Claude 3+)
        tools = [
            {
                "name": "web_search",
                "description": "Search the web for information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

        messages = [
            LLMMessage(
                role="user",
                content="Search for latest AI news"
            )
        ]

        response = await llm.function_call(messages, tools)
        if response.function_call:
            import json
            tool_input = json.loads(response.function_call["arguments"])
            print(f"Tool: {response.function_call['name']}")
            print(f"Input: {tool_input}")
        ```

    Args:
        config: LLMConfig, dataknobs Config, or dict with provider settings
        prompt_builder: Optional AsyncPromptBuilder for prompt rendering

    Attributes:
        _client: Anthropic AsyncAnthropic client instance

    See Also:
        LLMConfig: Configuration options
        AsyncLLMProvider: Base provider interface
        Anthropic API Docs: https://docs.anthropic.com/
    """

    def __init__(
        self,
        config: Union[LLMConfig, "Config", Dict[str, Any]],
        prompt_builder: AsyncPromptBuilder | None = None
    ):
        # Normalize config first
        llm_config = normalize_llm_config(config)
        super().__init__(llm_config, prompt_builder=prompt_builder)

    async def initialize(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic

            api_key = self.config.api_key or os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key not provided")

            self._client = anthropic.AsyncAnthropic(
                api_key=api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout
            )
            self._is_initialized = True
        except ImportError as e:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic") from e

    async def close(self) -> None:
        """Close Anthropic client."""
        if self._client:
            await self._client.close()  # type: ignore[unreachable]
        self._is_initialized = False

    async def validate_model(self) -> bool:
        """Validate model availability."""
        valid_models = [
            'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku',
            'claude-2.1', 'claude-2.0', 'claude-instant-1.2'
        ]
        return any(m in self.config.model for m in valid_models)

    def get_capabilities(self) -> List[ModelCapability]:
        """Get Anthropic model capabilities."""
        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.CODE
        ]

        # Claude 3+ models support vision and tools
        if 'claude-3' in self.config.model or 'claude-sonnet' in self.config.model or 'claude-opus' in self.config.model:
            capabilities.extend([
                ModelCapability.VISION,
                ModelCapability.FUNCTION_CALLING
            ])

        return capabilities

    async def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ) -> LLMResponse:
        """Generate completion."""
        if not self._is_initialized:
            await self.initialize()

        # Convert to Anthropic format
        if isinstance(messages, str):
            prompt = messages
        else:
            # Build prompt from messages
            prompt = ""
            for msg in messages:
                if msg.role == 'system':
                    prompt = msg.content + "\n\n" + prompt
                elif msg.role == 'user':
                    prompt += f"\n\nHuman: {msg.content}"
                elif msg.role == 'assistant':
                    prompt += f"\n\nAssistant: {msg.content}"
            prompt += "\n\nAssistant:"

        # Make API call
        response = await self._client.messages.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens or 1024,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stop_sequences=self.config.stop_sequences
        )

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            finish_reason=response.stop_reason,
            usage={
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            } if hasattr(response, 'usage') else None
        )

    async def stream_complete(
        self,
        messages: Union[str, List[LLMMessage]],
        **kwargs
    ) -> AsyncIterator[LLMStreamResponse]:
        """Generate streaming completion."""
        if not self._is_initialized:
            await self.initialize()

        # Convert to Anthropic format
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = self._build_prompt(messages)

        # Stream API call
        async with self._client.messages.stream(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens or 1024,
            temperature=self.config.temperature
        ) as stream:
            async for chunk in stream:
                if chunk.type == 'content_block_delta':
                    yield LLMStreamResponse(
                        delta=chunk.delta.text,
                        is_final=False
                    )

            # Final message
            message = await stream.get_final_message()
            yield LLMStreamResponse(
                delta='',
                is_final=True,
                finish_reason=message.stop_reason
            )

    async def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Anthropic doesn't provide embeddings."""
        raise NotImplementedError("Anthropic doesn't provide embedding models")

    async def function_call(
        self,
        messages: List[LLMMessage],
        functions: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """Execute function calling with native Anthropic tools API (Claude 3+)."""
        if not self._is_initialized:
            await self.initialize()

        # Convert to Anthropic message format
        anthropic_messages = []
        system_content = self.config.system_prompt or ''

        for msg in messages:
            if msg.role == 'system':
                # Anthropic uses system parameter, not system messages
                system_content = msg.content if not system_content else f"{system_content}\n\n{msg.content}"
            else:
                anthropic_messages.append({
                    'role': msg.role,
                    'content': msg.content
                })

        # Convert functions to Anthropic tools format
        tools = []
        for func in functions:
            tool = {
                'name': func.get('name', ''),
                'description': func.get('description', ''),
                'input_schema': func.get('parameters', {
                    'type': 'object',
                    'properties': {},
                    'required': []
                })
            }
            tools.append(tool)

        # Make API call with tools
        try:
            response = await self._client.messages.create(
                model=self.config.model,
                messages=anthropic_messages,
                system=system_content if system_content else None,
                tools=tools,
                max_tokens=self.config.max_tokens or 1024,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )

            # Extract response content and tool use
            content = ''
            tool_use = None

            for block in response.content:
                if block.type == 'text':
                    content += block.text
                elif block.type == 'tool_use':
                    tool_use = {
                        'name': block.name,
                        'arguments': block.input
                    }

            llm_response = LLMResponse(
                content=content,
                model=response.model,
                finish_reason=response.stop_reason,
                usage={
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                },
                function_call=tool_use
            )

            return llm_response

        except Exception as e:
            # Fallback to prompt-based approach for older models
            import logging
            logging.warning(f"Anthropic native tools failed, falling back to prompt-based: {e}")

            function_descriptions = "\n".join([
                f"- {f['name']}: {f['description']}"
                for f in functions
            ])

            system_prompt = f"""You have access to the following functions:
{function_descriptions}

When you need to call a function, respond with:
FUNCTION_CALL: {{
    "name": "function_name",
    "arguments": {{...}}
}}"""

            messages_with_system = [
                LLMMessage(role='system', content=system_prompt)
            ] + list(messages)

            response = await self.complete(messages_with_system, **kwargs)

            # Parse function call from response
            if 'FUNCTION_CALL:' in response.content:
                try:
                    func_json = response.content.split('FUNCTION_CALL:')[1].strip()
                    function_call = json.loads(func_json)
                    response.function_call = function_call
                except (json.JSONDecodeError, IndexError):
                    pass

            return response

    def _build_prompt(self, messages: List[LLMMessage]) -> str:
        """Build Anthropic-style prompt from messages."""
        prompt = ""
        for msg in messages:
            if msg.role == 'system':
                prompt = msg.content + "\n\n" + prompt
            elif msg.role == 'user':
                prompt += f"\n\nHuman: {msg.content}"
            elif msg.role == 'assistant':
                prompt += f"\n\nAssistant: {msg.content}"
        prompt += "\n\nAssistant:"
        return prompt
