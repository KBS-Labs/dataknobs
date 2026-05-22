"""Shared mixin for vector configuration across all backends."""

import logging
from typing import Any

from ..vector.types import DistanceMetric

logger = logging.getLogger(__name__)


class VectorConfigMixin:
    """Mixin to provide consistent vector configuration across all backends.
    
    This mixin ensures that all backends handle vector-related configuration
    parameters in a consistent way, including:
    - vector_enabled: Whether vector support is enabled
    - vector_metric: The distance metric to use for vector operations
    
    Usage:
        Include this mixin in your backend class and call _parse_vector_config()
        during initialization to extract vector configuration parameters.
    """

    def _parse_vector_config(self, config: dict[str, Any] | None = None) -> None:
        """Parse vector-related configuration parameters from a dict.

        Legacy dict-construction entry point. Backends constructed via
        ``StructuredConfigConsumer`` call :meth:`_apply_vector_config`
        with the typed config's fields directly instead.

        Args:
            config: Configuration dictionary (uses self.config if not provided)
        """
        # Use provided config or fall back to self.config
        config = config if config is not None else getattr(self, 'config', {})
        self._apply_vector_config(
            config.get("vector_enabled", False),
            config.get("vector_metric", "cosine"),
        )

    def _apply_vector_config(
        self,
        vector_enabled: bool,
        vector_metric: str | DistanceMetric,
    ) -> None:
        """Set vector state from already-resolved config values.

        Shared by :meth:`_parse_vector_config` (legacy dict path) and the
        ``_setup`` of backends migrated to ``StructuredConfigConsumer``
        (typed path). An unrecognized metric string falls back to cosine
        with a warning.

        Args:
            vector_enabled: Whether vector operations are enabled.
            vector_metric: Distance metric as a name or ``DistanceMetric``.
        """
        self._vector_enabled = vector_enabled

        if isinstance(vector_metric, DistanceMetric):
            self.vector_metric = vector_metric
        elif isinstance(vector_metric, str):
            try:
                self.vector_metric = DistanceMetric(vector_metric.lower())
            except ValueError:
                logger.warning(
                    f"Invalid vector metric '{vector_metric}', using cosine"
                )
                self.vector_metric = DistanceMetric.COSINE
        else:
            self.vector_metric = DistanceMetric.COSINE

        # Log vector configuration
        if self.vector_enabled:
            logger.debug(
                f"Vector support enabled with metric: {self.vector_metric.value}"
            )

    def _init_vector_state(self) -> None:
        """Initialize vector-related state variables.
        
        This method should be called after _parse_vector_config() to set up
        any additional state needed for vector operations.
        """
        # Track vector field dimensions (using underscore for consistency with mixins)
        self._vector_dimensions: dict[str, int] = {}
        self._vector_fields: dict[str, Any] = {}

        # For backends that support native vector operations
        self._has_native_vector_support = False

    @property
    def vector_enabled(self) -> bool:
        """Check if vector support is enabled.
        
        Returns:
            True if vector operations are enabled for this backend
        """
        return getattr(self, '_vector_enabled', False)

    @vector_enabled.setter
    def vector_enabled(self, value: bool) -> None:
        """Set vector support enabled state.
        
        Args:
            value: Whether to enable vector support
        """
        self._vector_enabled = value
