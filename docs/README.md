# Dataknobs Documentation

This directory contains the source files for the Dataknobs documentation site.

## Structure

```
docs/
├── api/                 # Auto-generated API documentation
├── development/         # Developer guides
├── examples/           # Usage examples
├── overrides/          # MkDocs Material theme overrides
├── packages/           # Package-specific documentation
├── stylesheets/        # Custom CSS
├── user-guide/         # User guides and tutorials
├── changelog.md        # Version history
├── getting-started.md  # Quick start guide
├── index.md           # Home page
├── installation.md    # Installation instructions
├── license.md         # License information
└── migration-guide.md # Migration from legacy package
```

## Building Documentation

### Prerequisites

Install MkDocs and required plugins:

```bash
uv pip install mkdocs mkdocs-material mkdocstrings[python] \
  mkdocs-monorepo-plugin mkdocs-awesome-pages-plugin \
  mkdocs-git-revision-date-localized-plugin
```

### Local Development

Serve documentation locally with hot reload:

```bash
# Using the provided script
./bin/docs-serve.sh

# Or directly with mkdocs
mkdocs serve
```

The documentation will be available at http://localhost:8000

### Building for Production

Build the static site:

```bash
# Using the provided script
./bin/docs-build.sh

# Or directly with mkdocs
mkdocs build
```

The built site will be in the `site/` directory.

### Deploying to GitHub Pages

Deploy directly to GitHub Pages:

```bash
mkdocs gh-deploy
```

Or use the GitHub Actions workflow which automatically deploys on push to main.

## Writing Documentation

### Markdown Extensions

The documentation supports various markdown extensions:

- **Admonitions**: `!!! note "Title"`
- **Code blocks with syntax highlighting**: ` ```python`
- **Tabs**: `=== "Tab 1"`
- **Tables**: Standard markdown tables
- **Task lists**: `- [x] Completed task`
- **Footnotes**: `[^1]`
- **Abbreviations**: `*[HTML]: Hyper Text Markup Language`

### API Documentation

API documentation is auto-generated from docstrings using mkdocstrings.

To document a module/class/function, use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of the function.
    
    Longer description providing more details about what
    the function does and how to use it.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
        
    Examples:
        >>> example_function("test", 42)
        True
    """
    pass
```

Then reference it in markdown:

```markdown
::: module.submodule.example_function
```

### Adding Examples

Create example files in `docs/examples/` with clear, runnable code:

```markdown
# Example: Working with Trees

This example demonstrates how to create and manipulate tree structures.

\```python
from dataknobs_structures import Tree

# Create a tree
root = Tree("root")
child = root.add_child("child")

# Print the tree structure
print(root)
\```

## Expected Output

\```
root
└── child
\```
```

## Style Guide

1. **Use clear, concise language**
2. **Include code examples for all features**
3. **Add type hints in examples**
4. **Use admonitions for important notes**
5. **Keep line length under 80 characters for code blocks**
6. **Use semantic versioning in changelog**

## Contributing

When adding new features or making changes:

1. Update relevant documentation
2. Add/update examples
3. Update API documentation docstrings
4. Update changelog if needed
5. Test documentation build locally before committing