"""Configuration schema definitions for FSM using Pydantic.

This module defines the schema for FSM configuration files, including:
- Data mode configuration
- Transaction configuration
- Resource definitions
- Streaming configuration
- FSM definition
- Network definition
- State definition
- Arc definition
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.core.transactions import TransactionStrategy


class ResourceType(str, Enum):
    """Available resource types."""

    DATABASE = "database"
    FILESYSTEM = "filesystem"
    HTTP = "http"
    LLM = "llm"
    VECTOR_STORE = "vector_store"
    CUSTOM = "custom"


class ExecutionStrategy(str, Enum):
    """Available execution strategies."""

    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    RESOURCE_OPTIMIZED = "resource_optimized"
    STREAM_OPTIMIZED = "stream_optimized"


class FunctionReference(BaseModel):
    """Reference to a function."""

    type: Literal["builtin", "custom", "inline", "registered"]
    name: str | None = None
    module: str | None = None
    code: str | None = None
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reference(self) -> "FunctionReference":
        """Validate that the reference has required fields based on type."""
        if self.type == "builtin" and not self.name:
            raise ValueError("Builtin functions require a 'name'")
        if self.type == "custom" and not (self.module and self.name):
            raise ValueError("Custom functions require both 'module' and 'name'")
        if self.type == "inline" and not self.code:
            raise ValueError("Inline functions require 'code'")
        if self.type == "registered" and not self.name:
            raise ValueError("Registered functions require a 'name'")
        return self


class DataModeConfig(BaseModel):
    """Configuration for data handling modes."""

    default: DataHandlingMode = DataHandlingMode.COPY
    state_overrides: Dict[str, DataHandlingMode] = Field(default_factory=dict)
    copy_config: Dict[str, Any] = Field(default_factory=dict)
    reference_config: Dict[str, Any] = Field(default_factory=dict)
    direct_config: Dict[str, Any] = Field(default_factory=dict)


class TransactionConfig(BaseModel):
    """Configuration for transaction management."""

    strategy: TransactionStrategy = TransactionStrategy.SINGLE
    batch_size: int = Field(default=100, ge=1)
    commit_triggers: List[str] = Field(default_factory=list)
    rollback_on_error: bool = True
    timeout_seconds: int | None = Field(default=None, ge=1)


class StreamConfig(BaseModel):
    """Configuration for streaming support."""

    enabled: bool = False
    chunk_size: int = Field(default=1000, ge=1)
    parallelism: int = Field(default=1, ge=1)
    memory_limit_mb: int | None = Field(default=None, ge=1)
    backpressure_threshold: float = Field(default=0.8, ge=0, le=1)
    format: str | None = None


class ResourceConfig(BaseModel):
    """Configuration for a resource."""

    name: str
    type: ResourceType
    config: Dict[str, Any] = Field(default_factory=dict)
    connection_pool_size: int = Field(default=10, ge=1)
    timeout_seconds: int = Field(default=30, ge=1)
    retry_attempts: int = Field(default=3, ge=0)
    retry_delay_seconds: float = Field(default=1.0, ge=0)
    health_check_interval: int | None = Field(default=None, ge=1)


class ArcConfig(BaseModel):
    """Configuration for an arc."""

    target: str
    condition: FunctionReference | None = None
    transform: FunctionReference | None = None
    resources: List[str] = Field(default_factory=list)
    priority: int = Field(default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PushArcConfig(ArcConfig):
    """Configuration for a push arc to another network."""

    target_network: str
    return_state: str | None = None
    data_isolation: DataHandlingMode = DataHandlingMode.COPY


class StateConfig(BaseModel):
    """Configuration for a state."""

    name: str
    data_schema: Dict[str, Any] | None = Field(default=None, alias="schema")
    pre_validators: List[FunctionReference] = Field(default_factory=list)
    validators: List[FunctionReference] = Field(default_factory=list)
    transforms: List[FunctionReference] = Field(default_factory=list)
    arcs: List[Union[ArcConfig, PushArcConfig]] = Field(default_factory=list)
    resources: List[str] = Field(default_factory=list)
    data_mode: DataHandlingMode | None = None
    is_start: bool = False
    is_end: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"populate_by_name": True}  # Allow both 'schema' and 'data_schema'

    @classmethod
    @field_validator("arcs", mode="before")
    def validate_arcs(cls, v: List[Any]) -> List[Union[ArcConfig, PushArcConfig]]:
        """Validate and convert arc configurations."""
        result = []
        for arc in v:
            if isinstance(arc, dict):
                if "target_network" in arc:
                    result.append(PushArcConfig(**arc))
                else:
                    result.append(ArcConfig(**arc))  # type: ignore
            else:
                result.append(arc)
        return result  # type: ignore


class NetworkConfig(BaseModel):
    """Configuration for a state network."""

    name: str
    states: List[StateConfig]
    resources: List[str] = Field(default_factory=list)
    streaming: StreamConfig | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_network(self) -> "NetworkConfig":
        """Validate network consistency."""
        state_names = {state.name for state in self.states}
        
        # Validate arc targets exist
        for state in self.states:
            for arc in state.arcs:
                if isinstance(arc, ArcConfig) and arc.target not in state_names:
                    raise ValueError(f"Arc target '{arc.target}' not found in network")
        
        # Validate at least one start state
        start_states = [s for s in self.states if s.is_start]
        if not start_states:
            raise ValueError("Network must have at least one start state")
        
        return self


class FSMConfig(BaseModel):
    """Complete FSM configuration."""

    name: str
    version: str = "1.0.0"
    description: str | None = None
    
    # Data handling
    data_mode: DataModeConfig = Field(default_factory=DataModeConfig)
    transaction: TransactionConfig = Field(default_factory=TransactionConfig)
    
    # Resources
    resources: List[ResourceConfig] = Field(default_factory=list)
    
    # Networks
    networks: List[NetworkConfig]
    main_network: str
    
    # Execution
    execution_strategy: ExecutionStrategy = ExecutionStrategy.DEPTH_FIRST
    max_transitions: int = Field(default=1000, ge=1)
    timeout_seconds: int | None = Field(default=None, ge=1)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode="after")
    def validate_fsm(self) -> "FSMConfig":
        """Validate FSM configuration consistency."""
        # Validate main network exists
        network_names = {net.name for net in self.networks}
        if self.main_network not in network_names:
            raise ValueError(f"Main network '{self.main_network}' not found")
        
        # Validate resource references
        resource_names = {res.name for res in self.resources}
        for network in self.networks:
            for res_name in network.resources:
                if res_name not in resource_names:
                    raise ValueError(f"Resource '{res_name}' not found in FSM resources")
            
            for state in network.states:
                for res_name in state.resources:
                    if res_name not in resource_names:
                        raise ValueError(f"Resource '{res_name}' not found in FSM resources")
        
        return self


class UseCaseTemplate(str, Enum):
    """Pre-defined use case templates."""

    DATABASE_ETL = "database_etl"
    FILE_PROCESSING = "file_processing"
    API_ORCHESTRATION = "api_orchestration"
    LLM_WORKFLOW = "llm_workflow"
    DATA_VALIDATION = "data_validation"
    STREAM_PROCESSING = "stream_processing"


class TemplateConfig(BaseModel):
    """Configuration for using a template."""

    template: UseCaseTemplate
    params: Dict[str, Any] = Field(default_factory=dict)
    overrides: Dict[str, Any] = Field(default_factory=dict)


def generate_json_schema() -> Dict[str, Any]:
    """Generate JSON schema for FSM configuration.
    
    Returns:
        JSON schema as a dictionary.
    """
    return FSMConfig.model_json_schema()


def validate_config(config: Dict[str, Any]) -> FSMConfig:
    """Validate a configuration dictionary.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Validated FSMConfig instance.
        
    Raises:
        ValidationError: If configuration is invalid.
    """
    return FSMConfig(**config)


# Template definitions
TEMPLATES: Dict[UseCaseTemplate, Dict[str, Any]] = {
    UseCaseTemplate.DATABASE_ETL: {
        "data_mode": {
            "default": DataHandlingMode.COPY,
        },
        "transaction": {
            "strategy": TransactionStrategy.BATCH,
            "batch_size": 1000,
        },
        "execution_strategy": ExecutionStrategy.RESOURCE_OPTIMIZED,
    },
    UseCaseTemplate.FILE_PROCESSING: {
        "data_mode": {
            "default": DataHandlingMode.REFERENCE,
        },
        "transaction": {
            "strategy": TransactionStrategy.SINGLE,
        },
        "execution_strategy": ExecutionStrategy.STREAM_OPTIMIZED,
    },
    UseCaseTemplate.API_ORCHESTRATION: {
        "data_mode": {
            "default": DataHandlingMode.COPY,
        },
        "transaction": {
            "strategy": TransactionStrategy.MANUAL,
        },
        "execution_strategy": ExecutionStrategy.DEPTH_FIRST,
    },
    UseCaseTemplate.LLM_WORKFLOW: {
        "data_mode": {
            "default": DataHandlingMode.COPY,
        },
        "transaction": {
            "strategy": TransactionStrategy.SINGLE,
        },
        "execution_strategy": ExecutionStrategy.RESOURCE_OPTIMIZED,
    },
    UseCaseTemplate.DATA_VALIDATION: {
        "data_mode": {
            "default": DataHandlingMode.DIRECT,
        },
        "transaction": {
            "strategy": TransactionStrategy.SINGLE,
        },
        "execution_strategy": ExecutionStrategy.DEPTH_FIRST,
    },
    UseCaseTemplate.STREAM_PROCESSING: {
        "data_mode": {
            "default": DataHandlingMode.REFERENCE,
        },
        "transaction": {
            "strategy": TransactionStrategy.BATCH,
            "batch_size": 5000,
        },
        "execution_strategy": ExecutionStrategy.STREAM_OPTIMIZED,
    },
}


def apply_template(
    template: UseCaseTemplate,
    params: Dict[str, Any] | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Apply a use case template to generate configuration.
    
    Args:
        template: Template to apply.
        params: Template parameters.
        overrides: Configuration overrides.
        
    Returns:
        Configuration dictionary.
    """
    import copy
    
    config = copy.deepcopy(TEMPLATES[template])
    
    # Apply parameters (template-specific logic can go here)
    if params:
        # This would contain template-specific parameter application
        pass
    
    # Apply overrides
    if overrides:
        def deep_merge(base: Dict, updates: Dict) -> Dict:
            for key, value in updates.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        deep_merge(config, overrides)
    
    return config
