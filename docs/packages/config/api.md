# Config Package API Reference

::: dataknobs_config.config
    options:
      show_source: true
      show_bases: true
      members:
        - Config

::: dataknobs_config.references
    options:
      show_source: true
      show_bases: true
      members:
        - ReferenceResolver

::: dataknobs_config.environment
    options:
      show_source: true
      show_bases: true
      members:
        - EnvironmentOverrides

::: dataknobs_config.settings
    options:
      show_source: true
      show_bases: true
      members:
        - SettingsManager

::: dataknobs_config.builders
    options:
      show_source: true
      show_bases: true
      members:
        - ObjectBuilder
        - ConfigurableBase
        - FactoryBase

::: dataknobs_config.exceptions
    options:
      show_source: true
      show_bases: true
      members:
        - ConfigError
        - ConfigNotFoundError
        - InvalidReferenceError
        - ValidationError
        - FileNotFoundError