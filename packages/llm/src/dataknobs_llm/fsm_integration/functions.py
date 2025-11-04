"""Built-in LLM functions for FSM.

This module provides LLM-related functions that can be referenced
in FSM configurations for AI-powered workflows.

Note: This module was migrated from dataknobs_fsm.functions.library.llm to
consolidate all LLM functionality in the dataknobs-llm package.
"""

import asyncio
import json
from typing import Any, Callable, Dict, List

from dataknobs_fsm.functions.base import (
    ITransformFunction,
    IValidationFunction,
    TransformError,
    ValidationError,
)
from dataknobs_llm.fsm_integration.resources import LLMResource


class PromptBuilder(ITransformFunction):
    """Build prompts for LLM calls."""

    def __init__(
        self,
        template: str,
        system_prompt: str | None = None,
        variables: List[str] | None = None,
        format_spec: str | None = None,  # "json", "markdown", "plain"
    ):
        """Initialize the prompt builder.
        
        Args:
            template: Prompt template with {variable} placeholders.
            system_prompt: Optional system prompt.
            variables: List of variable names to extract from data.
            format_spec: Output format specification.
        """
        self.template = template
        self.system_prompt = system_prompt
        self.variables = variables or []
        self.format_spec = format_spec

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by building prompt.
        
        Args:
            data: Input data containing variables for prompt.
            
        Returns:
            Data with built prompt.
        """
        # Extract variables
        variables = {}
        for var in self.variables:
            if var in data:
                variables[var] = data[var]
            else:
                # Try nested access
                parts = var.split(".")
                value = data
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                if value is not None:
                    variables[var] = value
        
        # Build prompt
        try:
            prompt = self.template.format(**variables)
        except KeyError as e:
            raise TransformError(f"Missing variable for prompt: {e}") from e
        
        # Add format specification if provided
        if self.format_spec:
            if self.format_spec == "json":
                prompt += "\n\nPlease respond with valid JSON only."
            elif self.format_spec == "markdown":
                prompt += "\n\nPlease format your response using Markdown."
        
        result = {
            **data,
            "prompt": prompt,
        }

        if self.system_prompt:
            result["system_prompt"] = self.system_prompt

        return result

    def get_transform_description(self) -> str:
        """Get a description of the transformation.

        Returns:
            String describing what this transform does.
        """
        return f"Build prompt from template: {self.template}"


class LLMCaller(ITransformFunction):
    """Call an LLM with a prompt."""

    def __init__(
        self,
        resource_name: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
        response_field: str = "llm_response",
    ):
        """Initialize the LLM caller.
        
        Args:
            resource_name: Name of the LLM resource to use.
            model: Model to use (if None, use resource default).
            temperature: Temperature for generation.
            max_tokens: Maximum tokens to generate.
            stream: Whether to stream the response.
            response_field: Field to store response in.
        """
        self.resource_name = resource_name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.response_field = response_field

    async def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by calling LLM.
        
        Args:
            data: Input data containing prompt.
            
        Returns:
            Data with LLM response.
        """
        # Get resource from context
        resource = data.get("_resources", {}).get(self.resource_name)
        if not resource or not isinstance(resource, LLMResource):
            raise TransformError(f"LLM resource '{self.resource_name}' not found")
        
        # Get prompt
        prompt = data.get("prompt")
        if not prompt:
            raise TransformError("No prompt found in data")
        
        system_prompt = data.get("system_prompt")
        
        try:
            # Call LLM
            response = await resource.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=self.stream,
            )
            
            if self.stream:
                # For streaming, return an async generator
                return {
                    **data,
                    self.response_field: response,  # Async generator
                    "is_streaming": True,
                }
            else:
                # For non-streaming, return the full response
                return {
                    **data,
                    self.response_field: response,
                    "tokens_used": response.get("usage", {}).get("total_tokens"),
                }
        
        except Exception as e:
            raise TransformError(f"LLM call failed: {e}") from e

    def get_transform_description(self) -> str:
        """Get a description of the transformation.

        Returns:
            String describing what this transform does.
        """
        return f"Call LLM resource '{self.resource_name}' with prompt"


class ResponseValidator(IValidationFunction):
    """Validate LLM responses."""

    def __init__(
        self,
        response_field: str = "llm_response",
        format: str | None = None,  # "json", "markdown", etc.
        schema: Dict[str, Any] | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        required_fields: List[str] | None = None,
    ):
        """Initialize the response validator.
        
        Args:
            response_field: Field containing LLM response.
            format: Expected response format.
            schema: JSON schema for validation (if format is JSON).
            min_length: Minimum response length.
            max_length: Maximum response length.
            required_fields: Required fields in parsed response.
        """
        self.response_field = response_field
        self.format = format
        self.schema = schema
        self.min_length = min_length
        self.max_length = max_length
        self.required_fields = required_fields or []

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate LLM response.
        
        Args:
            data: Data containing LLM response.
            
        Returns:
            True if valid.
            
        Raises:
            ValidationError: If validation fails.
        """
        response = data.get(self.response_field)
        if response is None:
            raise ValidationError(f"Response field '{self.response_field}' not found")
        
        # Extract text from response object if needed
        if isinstance(response, dict):
            text = response.get("text", response.get("content", str(response)))
        else:
            text = str(response)
        
        # Check length constraints
        if self.min_length and len(text) < self.min_length:  # type: ignore
            raise ValidationError(
                f"Response too short: {len(text)} < {self.min_length}"  # type: ignore
            )
        
        if self.max_length and len(text) > self.max_length:  # type: ignore
            raise ValidationError(
                f"Response too long: {len(text)} > {self.max_length}"  # type: ignore
            )
        
        # Validate format
        if self.format == "json":
            try:
                parsed = json.loads(text)  # type: ignore
                
                # Validate against schema if provided
                if self.schema:
                    from pydantic import create_model, ValidationError as PydanticValidationError
                    model = create_model("ResponseSchema", **self.schema)
                    try:
                        model(**parsed)
                    except PydanticValidationError as e:
                        raise ValidationError(f"Schema validation failed: {e}") from e
                
                # Check required fields
                for field in self.required_fields:
                    if field not in parsed:
                        raise ValidationError(f"Required field missing: {field}")
                
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON response: {e}") from e

        return True

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules this function implements.

        Returns:
            Dictionary describing the validation rules.
        """
        return {
            "response_field": self.response_field,
            "format": self.format,
            "schema": self.schema,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "required_fields": self.required_fields,
        }


class FunctionCaller(ITransformFunction):
    """Call functions/tools based on LLM output."""

    def __init__(
        self,
        response_field: str = "llm_response",
        function_registry: Dict[str, Callable] | None = None,
        result_field: str = "function_result",
    ):
        """Initialize the function caller.
        
        Args:
            response_field: Field containing LLM response with function call.
            function_registry: Registry of available functions.
            result_field: Field to store function result.
        """
        self.response_field = response_field
        self.function_registry = function_registry or {}
        self.result_field = result_field

    async def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by calling function from LLM response.
        
        Args:
            data: Input data containing LLM response with function call.
            
        Returns:
            Data with function result.
        """
        response = data.get(self.response_field)
        if not response:
            return data
        
        # Parse function call from response
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                # Not a JSON response, no function to call
                return data
        
        # Extract function call
        function_name = response.get("function")
        function_args = response.get("arguments", {})
        
        if not function_name:
            return data
        
        # Look up function
        if function_name not in self.function_registry:
            raise TransformError(f"Function not found: {function_name}")
        
        func = self.function_registry[function_name]
        
        try:
            # Call function
            if asyncio.iscoroutinefunction(func):
                result = await func(**function_args)
            else:
                result = func(**function_args)
            
            return {
                **data,
                self.result_field: result,
                "function_called": function_name,
            }
        
        except Exception as e:
            raise TransformError(f"Function call failed: {e}") from e

    def get_transform_description(self) -> str:
        """Get a description of the transformation.

        Returns:
            String describing what this transform does.
        """
        available_funcs = ", ".join(self.function_registry.keys()) if self.function_registry else "none"
        return f"Call function from LLM response (available: {available_funcs})"


class ConversationManager(ITransformFunction):
    """Manage conversation history for multi-turn interactions."""

    def __init__(
        self,
        max_history: int = 10,
        history_field: str = "conversation_history",
        role_field: str = "role",
        content_field: str = "content",
    ):
        """Initialize the conversation manager.
        
        Args:
            max_history: Maximum number of messages to keep.
            history_field: Field to store conversation history.
            role_field: Field for message role.
            content_field: Field for message content.
        """
        self.max_history = max_history
        self.history_field = history_field
        self.role_field = role_field
        self.content_field = content_field

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by managing conversation history.
        
        Args:
            data: Input data with new message.
            
        Returns:
            Data with updated conversation history.
        """
        # Get existing history
        history = data.get(self.history_field, [])
        
        # Add user message if present
        if "prompt" in data:
            history.append({
                self.role_field: "user",
                self.content_field: data["prompt"],
            })
        
        # Add assistant message if present
        if "llm_response" in data:
            response = data["llm_response"]
            if isinstance(response, dict):
                content = response.get("text", response.get("content", str(response)))
            else:
                content = str(response)
            
            history.append({
                self.role_field: "assistant",
                self.content_field: content,
            })
        
        # Trim history if needed
        if len(history) > self.max_history:
            history = history[-self.max_history:]
        
        return {
            **data,
            self.history_field: history,
        }

    def get_transform_description(self) -> str:
        """Get a description of the transformation.

        Returns:
            String describing what this transform does.
        """
        return f"Manage conversation history (max {self.max_history} messages)"


class EmbeddingGenerator(ITransformFunction):
    """Generate embeddings for text using LLM."""

    def __init__(
        self,
        resource_name: str,
        text_field: str = "text",
        embedding_field: str = "embedding",
        model: str | None = None,
        batch_size: int = 100,
    ):
        """Initialize the embedding generator.
        
        Args:
            resource_name: Name of the LLM resource to use.
            text_field: Field containing text to embed.
            embedding_field: Field to store embeddings.
            model: Embedding model to use.
            batch_size: Batch size for embedding generation.
        """
        self.resource_name = resource_name
        self.text_field = text_field
        self.embedding_field = embedding_field
        self.model = model
        self.batch_size = batch_size

    async def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by generating embeddings.
        
        Args:
            data: Input data containing text.
            
        Returns:
            Data with embeddings.
        """
        # Get resource from context
        resource = data.get("_resources", {}).get(self.resource_name)
        if not resource or not isinstance(resource, LLMResource):
            raise TransformError(f"LLM resource '{self.resource_name}' not found")
        
        # Get text to embed
        text = data.get(self.text_field)
        if not text:
            return data
        
        try:
            # Generate embedding(s)
            if isinstance(text, list):
                # Batch processing
                embeddings = []
                for i in range(0, len(text), self.batch_size):
                    batch = text[i:i + self.batch_size]
                    batch_embeddings = await resource.embed(batch, model=self.model)
                    embeddings.extend(batch_embeddings)
            else:
                # Single text
                embeddings = await resource.embed(text, model=self.model)
            
            return {
                **data,
                self.embedding_field: embeddings,
            }
        
        except Exception as e:
            raise TransformError(f"Embedding generation failed: {e}") from e

    def get_transform_description(self) -> str:
        """Get a description of the transformation.

        Returns:
            String describing what this transform does.
        """
        return f"Generate embeddings using resource '{self.resource_name}'"


# Convenience functions for creating LLM functions
def build_prompt(template: str, **kwargs) -> PromptBuilder:
    """Create a PromptBuilder."""
    return PromptBuilder(template, **kwargs)


def call_llm(resource: str, **kwargs) -> LLMCaller:
    """Create an LLMCaller."""
    return LLMCaller(resource, **kwargs)


def validate_response(**kwargs) -> ResponseValidator:
    """Create a ResponseValidator."""
    return ResponseValidator(**kwargs)


def call_function(**kwargs) -> FunctionCaller:
    """Create a FunctionCaller."""
    return FunctionCaller(**kwargs)


def manage_conversation(**kwargs) -> ConversationManager:
    """Create a ConversationManager."""
    return ConversationManager(**kwargs)


def generate_embeddings(resource: str, **kwargs) -> EmbeddingGenerator:
    """Create an EmbeddingGenerator."""
    return EmbeddingGenerator(resource, **kwargs)
