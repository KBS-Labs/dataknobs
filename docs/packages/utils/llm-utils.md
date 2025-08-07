# LLM Utilities API Documentation

The `llm_utils` module provides utilities for working with Large Language Models (LLMs), including prompt management, message formatting, and conversation handling.

## Overview

This module includes:

- Utilities for deep dictionary value retrieval
- Prompt message management classes
- Conversation and thread management
- Integration with tree structures for hierarchical data

## Utility Functions

### get_value_by_key()
```python
def get_value_by_key(
    d: Optional[Dict[str, Any]],
    pathkey: str,
    default_value: Any = None,
) -> Any
```

Get a "deep" value from a nested dictionary using dot-delimited path notation.

**Parameters:**
- `d` (Optional[Dict[str, Any]]): The (possibly nested) dictionary
- `pathkey` (str): The dot-delimited path key (e.g., "foo.bar.baz")
- `default_value` (Any, default=None): Value to return when path doesn't exist

**Returns:** The retrieved value or the default_value

**Example:**
```python
from dataknobs_utils import llm_utils

# Simple nested dictionary access
data = {
    "user": {
        "profile": {
            "name": "Alice",
            "email": "alice@example.com"
        },
        "preferences": {
            "theme": "dark",
            "language": "en"
        }
    }
}

# Get nested values
name = llm_utils.get_value_by_key(data, "user.profile.name")
print(name)  # "Alice"

email = llm_utils.get_value_by_key(data, "user.profile.email")
print(email)  # "alice@example.com"

theme = llm_utils.get_value_by_key(data, "user.preferences.theme")
print(theme)  # "dark"

# Handle missing keys with default value
age = llm_utils.get_value_by_key(data, "user.profile.age", 25)
print(age)  # 25 (default value)

# Handle None input safely
result = llm_utils.get_value_by_key(None, "any.path", "fallback")
print(result)  # "fallback"
```

## Classes

### PromptMessage
```python
class PromptMessage:
    def __init__(
        self, 
        role: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    )
```

Wrapper for a prompt message with role-based content and optional metadata.

**Parameters:**
- `role` (str): The message role (e.g., "system", "user", "assistant")
- `content` (str): The message content
- `metadata` (Optional[Dict[str, Any]], default=None): Additional metadata

**Properties:**
- `role` (str): Message role
- `content` (str): Message content  
- `metadata` (Dict[str, Any]): Message metadata

**Metadata Structure:**
The metadata dictionary can contain:
- `generation_args`: Arguments used for generation
- `execution_data`: Model name, start time, end time, etc.
- `user_comments`: List of user comments with user and comment fields

**Example:**
```python
from dataknobs_utils import llm_utils
from datetime import datetime

# Create system message
system_msg = llm_utils.PromptMessage(
    "system",
    "You are a helpful AI assistant specialized in data analysis."
)

# Create user message with metadata
user_msg = llm_utils.PromptMessage(
    "user",
    "Analyze this dataset and provide insights.",
    metadata={
        "generation_args": {
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "user_comments": [
            {
                "user": "alice",
                "comment": "This is a priority analysis"
            }
        ]
    }
)

# Create assistant response with execution metadata
assistant_msg = llm_utils.PromptMessage(
    "assistant",
    "Based on the dataset analysis, I found the following patterns...",
    metadata={
        "execution_data": {
            "model_name": "gpt-4",
            "starttime": datetime.now().isoformat(),
            "endtime": datetime.now().isoformat(),
            "tokens_used": 250
        }
    }
)

# Access message properties
print(f"Role: {user_msg.role}")
print(f"Content: {user_msg.content}")
print(f"Temperature: {user_msg.metadata['generation_args']['temperature']}")
```

## Usage Patterns

### Building Conversation Flows
```python
from dataknobs_utils import llm_utils
from dataknobs_structures import Tree

class ConversationManager:
    def __init__(self):
        self.messages = []
    
    def add_system_message(self, content: str) -> None:
        """Add a system message to set context."""
        msg = llm_utils.PromptMessage("system", content)
        self.messages.append(msg)
    
    def add_user_message(self, content: str, **metadata) -> None:
        """Add a user message with optional metadata."""
        msg = llm_utils.PromptMessage("user", content, metadata)
        self.messages.append(msg)
    
    def add_assistant_response(self, content: str, model_info: dict) -> None:
        """Add an assistant response with execution metadata."""
        metadata = {"execution_data": model_info}
        msg = llm_utils.PromptMessage("assistant", content, metadata)
        self.messages.append(msg)
    
    def get_conversation_context(self) -> list:
        """Get conversation as list of role-content dictionaries."""
        return [
            {"role": msg.role, "content": msg.content} 
            for msg in self.messages
        ]
    
    def get_metadata_summary(self) -> dict:
        """Summarize metadata across all messages."""
        summary = {
            "total_messages": len(self.messages),
            "roles": {},
            "models_used": set(),
            "total_tokens": 0
        }
        
        for msg in self.messages:
            # Count roles
            summary["roles"][msg.role] = summary["roles"].get(msg.role, 0) + 1
            
            # Extract model info
            if msg.metadata:
                exec_data = msg.metadata.get("execution_data", {})
                if "model_name" in exec_data:
                    summary["models_used"].add(exec_data["model_name"])
                if "tokens_used" in exec_data:
                    summary["total_tokens"] += exec_data["tokens_used"]
        
        summary["models_used"] = list(summary["models_used"])
        return summary

# Usage example
conversation = ConversationManager()

# Set up conversation
conversation.add_system_message(
    "You are a data scientist helping with analysis tasks."
)

conversation.add_user_message(
    "What's the best approach for analyzing customer churn?",
    user_id="user123",
    priority="high"
)

conversation.add_assistant_response(
    "For customer churn analysis, I recommend starting with...",
    {
        "model_name": "gpt-4",
        "tokens_used": 150,
        "response_time": 2.3
    }
)

print(conversation.get_metadata_summary())
```

### Hierarchical Conversation Trees
```python
from dataknobs_utils import llm_utils
from dataknobs_structures import Tree

class ConversationTree:
    """Manage branching conversations using Tree structure."""
    
    def __init__(self, initial_message: str = None):
        self.root = Tree("conversation_root")
        if initial_message:
            self.add_message(initial_message, "system", parent=self.root)
    
    def add_message(self, content: str, role: str, parent=None, **metadata) -> Tree:
        """Add a message to the conversation tree."""
        if parent is None:
            parent = self.root
        
        message = llm_utils.PromptMessage(role, content, metadata)
        message_node = parent.add_child(message)
        return message_node
    
    def get_conversation_path(self, node: Tree) -> list:
        """Get conversation path from root to specific node."""
        path = node.get_path()[1:]  # Skip root
        return [
            {"role": n.data.role, "content": n.data.content}
            for n in path
        ]
    
    def branch_conversation(self, from_node: Tree, new_content: str, role: str) -> Tree:
        """Create a new branch in the conversation."""
        return self.add_message(new_content, role, parent=from_node)
    
    def find_messages_by_role(self, role: str) -> list:
        """Find all messages with specific role."""
        return self.root.find_nodes(
            lambda n: hasattr(n.data, 'role') and n.data.role == role
        )
    
    def get_all_paths(self) -> list:
        """Get all conversation paths (to leaf nodes)."""
        leaves = self.root.collect_terminal_nodes()
        return [self.get_conversation_path(leaf) for leaf in leaves]

# Usage example
conv_tree = ConversationTree(
    "You are an expert in machine learning and data analysis."
)

# Main conversation path
user_q1 = conv_tree.add_message(
    "How do I improve model accuracy?", 
    "user", 
    conv_tree.root.children[0]
)

assist_a1 = conv_tree.add_message(
    "There are several strategies: feature engineering, hyperparameter tuning...",
    "assistant",
    user_q1
)

# Branch 1: Follow up on feature engineering
user_q2a = conv_tree.add_message(
    "Tell me more about feature engineering techniques.",
    "user",
    assist_a1
)

assist_a2a = conv_tree.add_message(
    "Feature engineering involves creating new features from existing data...",
    "assistant",
    user_q2a
)

# Branch 2: Follow up on hyperparameters
user_q2b = conv_tree.add_message(
    "What's the best approach for hyperparameter tuning?",
    "user",
    assist_a1
)

assist_a2b = conv_tree.add_message(
    "For hyperparameter tuning, consider using grid search or random search...",
    "assistant",
    user_q2b
)

# Get all conversation paths
paths = conv_tree.get_all_paths()
for i, path in enumerate(paths):
    print(f"\nConversation Path {i + 1}:")
    for msg in path:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
```

### Configuration and Settings Management
```python
from dataknobs_utils import llm_utils

class LLMConfig:
    """Manage LLM configuration with nested settings."""
    
    def __init__(self, config_dict: dict):
        self.config = config_dict
    
    def get_setting(self, path: str, default=None):
        """Get setting using dot notation."""
        return llm_utils.get_value_by_key(self.config, path, default)
    
    def get_model_config(self, model_name: str) -> dict:
        """Get complete configuration for a specific model."""
        model_path = f"models.{model_name}"
        return self.get_setting(model_path, {})
    
    def get_generation_params(self, model_name: str) -> dict:
        """Get generation parameters for a model."""
        params_path = f"models.{model_name}.generation"
        default_params = self.get_setting("defaults.generation", {})
        model_params = self.get_setting(params_path, {})
        
        # Merge default and model-specific parameters
        return {**default_params, **model_params}

# Example configuration
config_data = {
    "defaults": {
        "generation": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9
        }
    },
    "models": {
        "gpt-4": {
            "api_key": "sk-...",
            "base_url": "https://api.openai.com/v1",
            "generation": {
                "temperature": 0.8,
                "max_tokens": 2000
            }
        },
        "claude": {
            "api_key": "sk-ant-...",
            "base_url": "https://api.anthropic.com",
            "generation": {
                "temperature": 0.6,
                "max_tokens": 1500
            }
        }
    },
    "features": {
        "conversation_memory": True,
        "auto_save": {
            "enabled": True,
            "interval": 300
        }
    }
}

config = LLMConfig(config_data)

# Get various settings
print(config.get_setting("defaults.generation.temperature"))  # 0.7
print(config.get_setting("models.gpt-4.api_key"))  # "sk-..."
print(config.get_setting("features.auto_save.enabled"))  # True
print(config.get_setting("nonexistent.path", "fallback"))  # "fallback"

# Get model-specific configurations
gpt4_config = config.get_model_config("gpt-4")
print(gpt4_config)

# Get generation parameters (with inheritance)
gpt4_params = config.get_generation_params("gpt-4")
print(gpt4_params)  # Merged default + model-specific params
```

## Error Handling

```python
from dataknobs_utils import llm_utils

def safe_config_access(config_data, path, expected_type=None):
    """Safely access configuration with type checking."""
    try:
        value = llm_utils.get_value_by_key(config_data, path)
        
        if value is None:
            print(f"Configuration path '{path}' not found")
            return None
        
        if expected_type and not isinstance(value, expected_type):
            print(f"Expected {expected_type.__name__} for '{path}', got {type(value).__name__}")
            return None
        
        return value
        
    except Exception as e:
        print(f"Error accessing configuration path '{path}': {e}")
        return None

# Usage
config = {"api": {"timeout": "30"}}

# This will warn about type mismatch
timeout = safe_config_access(config, "api.timeout", int)

# Safe message creation
try:
    msg = llm_utils.PromptMessage("user", "Hello world")
    print(f"Created message: {msg.role} - {msg.content}")
except Exception as e:
    print(f"Failed to create message: {e}")
```

## Integration Examples

### With Tree Structures
```python
from dataknobs_utils import llm_utils
from dataknobs_structures import Tree

# Build prompt template tree
def build_prompt_tree():
    root = Tree("prompt_templates")
    
    # Analysis templates
    analysis = root.add_child("analysis")
    analysis.add_child(llm_utils.PromptMessage(
        "system",
        "You are a data analyst. Analyze the provided data and give insights."
    ))
    
    # Creative templates
    creative = root.add_child("creative")
    creative.add_child(llm_utils.PromptMessage(
        "system", 
        "You are a creative writer. Help generate engaging content."
    ))
    
    return root

# Use templates
template_tree = build_prompt_tree()
analysis_templates = template_tree.find_nodes(
    lambda n: hasattr(n.data, 'content') and 'analyst' in n.data.content.lower()
)
```

### With File Processing
```python
from dataknobs_utils import llm_utils, file_utils
import json

# Load conversation history from files
def load_conversations(directory):
    conversations = []
    
    for filepath in file_utils.filepath_generator(directory):
        if filepath.endswith(".json"):
            for line in file_utils.fileline_generator(filepath):
                try:
                    data = json.loads(line)
                    role = data.get("role", "unknown")
                    content = data.get("content", "")
                    metadata = data.get("metadata", {})
                    
                    msg = llm_utils.PromptMessage(role, content, metadata)
                    conversations.append(msg)
                except json.JSONDecodeError:
                    continue
    
    return conversations

# Save conversations
def save_conversations(conversations, output_file):
    lines = []
    for msg in conversations:
        data = {
            "role": msg.role,
            "content": msg.content,
            "metadata": msg.metadata
        }
        lines.append(json.dumps(data))
    
    file_utils.write_lines(output_file, lines)
```

## Performance Considerations

- Use `get_value_by_key()` for safe nested dictionary access instead of chained `.get()` calls
- Store frequently accessed configuration paths in constants
- Consider caching configuration values for repeated access
- Use metadata efficiently - avoid storing large objects in message metadata

## Best Practices

- Always provide default values when accessing nested configuration
- Include meaningful metadata in PromptMessage objects for debugging
- Use consistent role names ("system", "user", "assistant")
- Structure metadata with clear categories (generation_args, execution_data, user_comments)
- Validate message content and roles before creating PromptMessage instances