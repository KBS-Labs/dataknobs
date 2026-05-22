# ConfigurableBase (deprecated)

`dataknobs_config.ConfigurableBase` is soft-deprecated in favor of
`dataknobs_common.structured_config.StructuredConfigConsumer`. Existing
consumers continue to work — no runtime warning is raised so the
transition stays quiet across multiple release cycles. Removal is
scheduled for a future release once the in-tree migration is
complete.

## What `ConfigurableBase` does

It is a kwarg-splat base class with one classmethod:

```python
class ConfigurableBase:
    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)
```

Adopters inherit the class and provide an `__init__` that names each
config key as a kwarg.

## Why it's deprecated

Three structural drift modes the kwarg-splat pattern can't catch:

- A factory enumerates only some of the ctor kwargs — drift between
  the factory allowlist and the ctor surface.
- The ctor accepts a kwarg with no corresponding config dataclass
  field — drift between the ctor and the documented config schema.
- A consumer needs pre-projection normalization (e.g., assembling a
  Postgres connection string from `DATABASE_URL` and `POSTGRES_*`
  env-var fallbacks) but the only override point is `from_config`,
  which mixes normalization with construction.

`StructuredConfigConsumer[ConfigT]` addresses all three by making the
typed `StructuredConfig` dataclass the single source of truth and
providing a dedicated `_normalize_dict` override hook for
pre-projection normalization.

## What to use instead

Move from kwarg-splat to typed-dispatch:

### Before

```python
from dataknobs_config import ConfigurableBase

class Widget(ConfigurableBase):
    def __init__(self, *, name: str = "default", size: int = 1):
        self.name = name
        self.size = size
        self.area = size ** 2

# Usage
Widget.from_config({"name": "x", "size": 4})
```

### After

```python
from dataclasses import dataclass
from typing import ClassVar
from dataknobs_common.structured_config import (
    StructuredConfig,
    StructuredConfigConsumer,
)


@dataclass(frozen=True)
class WidgetConfig(StructuredConfig):
    name: str = "default"
    size: int = 1


class Widget(StructuredConfigConsumer[WidgetConfig]):
    CONFIG_CLS: ClassVar[type[WidgetConfig]] = WidgetConfig

    def _setup(self) -> None:
        self.area = self._config.size ** 2

# Usage — same registry path:
Widget.from_config({"name": "x", "size": 4})

# Plus typed access and additional construction shapes:
w = Widget(WidgetConfig(name="x", size=4))
print(w.config.name)
```

See [Structured Configuration](../common/structured-config.md) for the
full API reference.
