"""File system resource provider."""

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, TextIO, Union

from dataknobs_fsm.functions.base import ResourceError
from dataknobs_fsm.resources.base import (
    BaseResourceProvider,
    ResourceHealth,
    ResourceStatus,
)


class FileHandle:
    """Wrapper for file handles with metadata."""
    
    def __init__(self, path: Path, handle: Union[TextIO, BinaryIO], mode: str):
        """Initialize file handle.
        
        Args:
            path: File path.
            handle: The file handle.
            mode: File open mode.
        """
        self.path = path
        self.handle = handle
        self.mode = mode
        self.closed = False
    
    def close(self) -> None:
        """Close the file handle."""
        if not self.closed and self.handle:
            self.handle.close()
            self.closed = True
    
    def __enter__(self):
        """Enter context manager."""
        return self.handle
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()


class FileSystemResource(BaseResourceProvider):
    """File system resource provider for file I/O operations."""
    
    def __init__(
        self,
        name: str,
        base_path: str | None = None,
        temp_dir: str | None = None,
        **config
    ):
        """Initialize file system resource.
        
        Args:
            name: Resource name.
            base_path: Base directory for file operations.
            temp_dir: Directory for temporary files.
            **config: Additional configuration.
        """
        super().__init__(name, config)
        
        # Set up base path
        if base_path:
            self.base_path = Path(base_path).resolve()
            self.base_path.mkdir(parents=True, exist_ok=True)
        else:
            self.base_path = Path.cwd()
        
        # Set up temp directory
        self.temp_dir = temp_dir
        self._temp_files = []
        self._open_handles = {}
        
        self.status = ResourceStatus.IDLE
    
    def acquire(
        self,
        path: str | None = None,
        mode: str = "r",
        encoding: str | None = "utf-8",
        temp: bool = False,
        **kwargs
    ) -> FileHandle:
        """Acquire a file handle.
        
        Args:
            path: File path (relative to base_path).
            mode: File open mode.
            encoding: Text encoding (None for binary modes).
            temp: If True, create a temporary file.
            **kwargs: Additional open() parameters.
            
        Returns:
            FileHandle wrapper with the open file.
            
        Raises:
            ResourceError: If file operation fails.
        """
        try:
            if temp:
                # Create temporary file
                suffix = Path(path).suffix if path else ""
                prefix = Path(path).stem if path else "tmp_"
                
                if "b" in mode:
                    handle = tempfile.NamedTemporaryFile(
                        mode=mode,
                        suffix=suffix,
                        prefix=prefix,
                        dir=self.temp_dir,
                        delete=False
                    )
                else:
                    handle = tempfile.NamedTemporaryFile(
                        mode=mode,
                        suffix=suffix,
                        prefix=prefix,
                        dir=self.temp_dir,
                        delete=False,
                        encoding=encoding
                    )
                
                file_path = Path(handle.name)
                self._temp_files.append(file_path)
            else:
                if not path:
                    raise ResourceError(
                        "Path required for non-temporary files",
                        resource_name=self.name,
                        operation="acquire"
                    )
                
                # Resolve path relative to base
                file_path = self.base_path / path
                
                # Create parent directories if writing
                if any(m in mode for m in ["w", "a", "x"]):
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Open file
                if "b" in mode:
                    handle = open(file_path, mode, **kwargs)
                else:
                    handle = open(file_path, mode, encoding=encoding, **kwargs)
            
            # Create wrapper
            file_handle = FileHandle(file_path, handle, mode)  # type: ignore
            self._open_handles[id(file_handle)] = file_handle
            self._resources.append(file_handle)
            
            if self._resources:
                self.status = ResourceStatus.ACTIVE
            
            return file_handle
            
        except Exception as e:
            self.status = ResourceStatus.ERROR
            raise ResourceError(
                f"Failed to acquire file resource: {e}",
                resource_name=self.name,
                operation="acquire"
            ) from e
    
    def release(self, resource: Any) -> None:
        """Release a file handle.
        
        Args:
            resource: The FileHandle to release.
        """
        if isinstance(resource, FileHandle):
            resource.close()
            
            # Remove from tracking
            handle_id = id(resource)
            if handle_id in self._open_handles:
                del self._open_handles[handle_id]
            
            if resource in self._resources:
                self._resources.remove(resource)
        
        if not self._resources:
            self.status = ResourceStatus.IDLE
    
    def validate(self, resource: Any) -> bool:
        """Validate a file handle.
        
        Args:
            resource: The FileHandle to validate.
            
        Returns:
            True if the handle is valid and not closed.
        """
        if not isinstance(resource, FileHandle):
            return False
        
        return not resource.closed and resource.handle and not resource.handle.closed  # type: ignore
    
    def health_check(self) -> ResourceHealth:
        """Check file system health.
        
        Returns:
            Health status.
        """
        try:
            # Try to create and delete a temp file
            test_file = self.base_path / ".health_check"
            test_file.write_text("test")
            test_file.unlink()
            
            self.metrics.record_health_check(True)
            return ResourceHealth.HEALTHY
        except Exception:
            self.metrics.record_health_check(False)
            return ResourceHealth.UNHEALTHY
    
    @contextmanager
    def open(
        self,
        path: str,
        mode: str = "r",
        encoding: str | None = "utf-8",
        **kwargs
    ):
        """Context manager for file operations.
        
        Args:
            path: File path.
            mode: File mode.
            encoding: Text encoding.
            **kwargs: Additional parameters.
            
        Yields:
            The file handle.
        """
        handle = self.acquire(path, mode, encoding, **kwargs)
        try:
            yield handle.handle
        finally:
            self.release(handle)
    
    @contextmanager
    def temp_file(
        self,
        suffix: str = "",
        prefix: str = "tmp_",
        mode: str = "w",
        encoding: str | None = "utf-8"
    ):
        """Context manager for temporary files.
        
        Args:
            suffix: File suffix.
            prefix: File prefix.
            mode: File mode.
            encoding: Text encoding.
            
        Yields:
            The temporary file handle.
        """
        handle = self.acquire(
            path=f"{prefix}file{suffix}",
            mode=mode,
            encoding=encoding,
            temp=True
        )
        try:
            yield handle.handle
        finally:
            self.release(handle)
    
    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text from a file.
        
        Args:
            path: File path.
            encoding: Text encoding.
            
        Returns:
            File contents as string.
        """
        with self.open(path, "r", encoding) as f:
            return f.read()
    
    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """Write text to a file.
        
        Args:
            path: File path.
            content: Content to write.
            encoding: Text encoding.
        """
        with self.open(path, "w", encoding) as f:
            f.write(content)
    
    def read_bytes(self, path: str) -> bytes:
        """Read binary data from a file.
        
        Args:
            path: File path.
            
        Returns:
            File contents as bytes.
        """
        with self.open(path, "rb", encoding=None) as f:
            return f.read()
    
    def write_bytes(self, path: str, content: bytes) -> None:
        """Write binary data to a file.
        
        Args:
            path: File path.
            content: Binary content to write.
        """
        with self.open(path, "wb", encoding=None) as f:
            f.write(content)
    
    def exists(self, path: str) -> bool:
        """Check if a file exists.
        
        Args:
            path: File path.
            
        Returns:
            True if file exists.
        """
        file_path = self.base_path / path
        return file_path.exists()
    
    def delete(self, path: str) -> bool:
        """Delete a file.
        
        Args:
            path: File path.
            
        Returns:
            True if file was deleted.
        """
        try:
            file_path = self.base_path / path
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception:
            return False
    
    def list_files(self, pattern: str = "*") -> list[str]:
        """List files matching a pattern.
        
        Args:
            pattern: Glob pattern.
            
        Returns:
            List of file paths.
        """
        files = []
        for path in self.base_path.glob(pattern):
            if path.is_file():
                files.append(str(path.relative_to(self.base_path)))
        return files
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        import logging
        logger = logging.getLogger(__name__)
        
        cleanup_errors = []
        for temp_file in self._temp_files[:]:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                self._temp_files.remove(temp_file)
            except PermissionError as e:
                cleanup_errors.append(f"Permission denied cleaning up {temp_file}: {e}")
                logger.warning(f"Could not delete temporary file {temp_file}: {e}")
            except OSError as e:
                cleanup_errors.append(f"OS error cleaning up {temp_file}: {e}")
                logger.warning(f"OS error deleting temporary file {temp_file}: {e}")
            except Exception as e:
                cleanup_errors.append(f"Unexpected error cleaning up {temp_file}: {e}")
                logger.error(f"Unexpected error cleaning up {temp_file}: {e}")
        
        if cleanup_errors:
            # Store cleanup errors in metadata for debugging
            if not hasattr(self, '_cleanup_errors'):
                self._cleanup_errors = []
            self._cleanup_errors.extend(cleanup_errors)
    
    def close(self) -> None:
        """Close all open handles and clean up."""
        # Close all open handles
        for handle in list(self._open_handles.values()):
            self.release(handle)
        
        # Clean up temp files
        self.cleanup_temp_files()
        
        # Call parent close
        super().close()
