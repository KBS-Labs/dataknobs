"""Anthropic Claude LLM provider implementation."""

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
    """Anthropic Claude LLM provider.

    Supports latest Anthropic features including:
    - Native tools API (Claude 3+)
    - Vision capabilities (Claude 3+)
    - Streaming responses
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
