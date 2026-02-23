# Developer Workflow with dk Command

The `dk` (DataKnobs) command is a unified developer interface that simplifies common development tasks with intuitive shortcuts and smart defaults. It's designed to make your development workflow faster and more enjoyable.

## Quick Start

### Installation

```bash
# From the project root, run:
./setup-dk.sh

# Or manually create a symlink:
ln -sf ./bin/dk ~/.local/bin/dk
```

### Essential Commands

```bash
dk pr          # Prepare for PR (full quality checks)
dk check data  # Quick quality check for data package
dk test        # Run tests with smart defaults
dk fix         # Auto-fix style issues
dk diagnose    # Find out what failed
```

## Command Reference

### Super Short Aliases

For maximum speed, use single-letter aliases:

| Alias | Command | Description |
|-------|---------|-------------|
| `dk p` | `dk pr` | Prepare for PR |
| `dk q` | `dk check` | Quick quality check |
| `dk t` | `dk test` | Run tests |
| `dk tf` | `dk test --last` | Test failures only |
| `dk f` | `dk fix` | Fix style issues |
| `dk d` | `dk diagnose` | Diagnose failures |
| `dk y` | `dk why` | Why did it fail? |

### PR Preparation

Commands for preparing code for pull requests:

```bash
dk pr              # Full quality checks (lint, style, tests, coverage)
dk prq             # Quick PR check (skip slow linting)
dk pr data         # PR checks for specific package
```

### Quality Checks

Fast development-mode quality checks:

```bash
dk check           # Quick check current directory
dk check data      # Check specific package
dk checkall        # Check all packages
dk lint            # Run linting only
dk style           # Run style checks only
```

### Testing

Smart testing with various modes:

```bash
dk test            # Run tests (auto-detects what to test)
dk test data       # Test specific package
dk unit            # Run unit tests only
dk int             # Run integration tests only
dk testlast        # Re-run only failed tests
dk testquick       # Tests without coverage
dk test -- -x      # Pass pytest args (stop on first failure)
dk test -- -vvs    # Verbose output with stdout
```

### Fixing Issues

Auto-fix and format code:

```bash
dk fix             # Auto-fix style issues
dk fixlint         # Try to auto-fix lint issues
dk format          # Format code with ruff
```

### Diagnostics

Understand what went wrong:

```bash
dk diagnose        # Analyze last failure
dk diagnose -v     # Verbose diagnostic output
dk diagnose -t     # Focus on test failures
dk diagnose -f     # Show fix commands
dk coverage        # Open coverage report in browser
dk why             # Alias for diagnose
```

### Documentation

Work with MkDocs documentation:

```bash
dk docs            # Serve docs locally with live reload
dk docs-build      # Build documentation to site/
dk docs-check      # Check for documentation issues
dk docs-open       # Open built docs in browser
```

### Service Management

Manage development services (PostgreSQL, Elasticsearch, LocalStack):

```bash
dk up              # Start development services
dk down            # Stop services
dk restart         # Restart services
dk logs            # Show all service logs
dk logs postgres   # Show specific service logs
```

### Dependencies

Compare dependency versions across branches:

```bash
dk deps            # Show dependency changes vs main
dk deps <ref>      # Compare vs any git ref (branch, tag, commit)
dk deps --staged   # Compare staged uv.lock vs HEAD
```

See [Dependency Updates](dependency-updates.md) for the full update and review workflow.

### Cleanup

Clean artifacts and reset state:

```bash
dk clean           # Clean artifacts and caches
dk cleanall        # Deep clean (includes .venv)
dk reset           # Clean and reinstall packages
```

## Common Workflows

### Before Creating a PR

```bash
# Full quality check
dk pr

# If failures, diagnose and fix
dk diagnose        # See what failed
dk fix             # Auto-fix style issues
dk test --last     # Re-run failed tests
dk pr              # Verify everything passes
```

### Quick Development Cycle

```bash
# Check specific package during development
dk check data      # Quick quality check
dk fix             # Fix any style issues
dk test data       # Run package tests

# Or even shorter:
dk q data && dk f && dk t data
```

### After Test Failures

```bash
# Understand and fix failures
dk d               # Diagnose what failed
dk tf              # Re-run only failed tests
dk t -- -x -vvs    # Debug with verbose output
```

### Service Management

```bash
# Start work session
dk up              # Start services

# During development
dk logs postgres   # Check database logs
dk restart         # Restart if needed

# End work session
dk down            # Stop services
```

### Documentation Development

```bash
# Write documentation with live preview
dk docs            # Start server at http://localhost:8000

# Check documentation quality
dk docs-check      # Find broken links and issues
dk docs-build      # Build final documentation
dk docs-open       # View built docs
```

### Release Management

```bash
# Interactive release process
dk release         # Full guided release workflow

# Individual release steps
dk release-check   # See what changed since last release
dk release-bump    # Update package versions
dk release-notes   # Generate changelog entries

# After creating release PR and merging
bin/tag-releases.sh      # Create git tags
bin/build-packages.sh    # Build distributions
bin/publish-pypi.sh      # Publish to PyPI
```

## Advanced Usage

### Passing Arguments to Underlying Tools

You can pass arguments through to pytest:

```bash
dk test -- -k test_name     # Run specific test
dk test -- --lf             # Run last failed
dk test -- -x               # Stop on first failure
dk test -- --pdb            # Drop into debugger on failure
```

### Package-Specific Operations

Most commands accept package names:

```bash
dk check data config    # Check multiple packages
dk test data            # Test specific package
dk fix                  # Fix all packages
```

### PR Mode vs Dev Mode

The quality check system has two modes:

- **PR Mode**: Full checks with artifacts, separate unit/integration test runs
- **Dev Mode**: Quick checks without artifacts, combined test runs

```bash
dk pr              # Force PR mode
dk check           # Force dev mode
```

## Tips and Tricks

### Quick Reference

Display the quick reference card:

```bash
dk cheat           # Show cheatsheet
dk cs              # Short alias for cheatsheet
```

### Command Discovery

Not sure what command to use?

```bash
dk help            # Full help with all commands
dk                 # Same as dk help
```

### Shell Completion

Add to your `.bashrc` or `.zshrc`:

```bash
# Bash completion for dk
_dk_complete() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local commands="pr check test fix diagnose clean up down help"
    COMPREPLY=($(compgen -W "${commands}" -- ${cur}))
}
complete -F _dk_complete dk
```

### Project-Specific Aliases

Create your own aliases in `.bashrc` or `.zshrc`:

```bash
alias dkp='dk pr'
alias dkq='dk check'
alias dkt='dk test'
alias dkf='dk fix'
```

## Troubleshooting

### Command Not Found

If `dk` is not found after installation:

1. Check if the symlink exists:
   ```bash
   ls -la ~/.local/bin/dk
   ```

2. Ensure the directory is in your PATH:
   ```bash
   echo $PATH | grep .local/bin
   ```

3. Add to PATH if needed:
   ```bash
   echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc
   source ~/.bashrc
   ```

### Services Won't Start

If development services fail to start:

```bash
# Check if ports are in use
dk down            # Stop any running services
docker ps          # Check for conflicting containers
dk up              # Try starting again
```

### Tests Failing Mysteriously

```bash
# Clean everything and start fresh
dk clean           # Clean artifacts
dk reset           # Reinstall packages
dk up              # Restart services
dk test            # Run tests again
```

## Philosophy

The `dk` command follows these principles:

1. **Intuitive**: Commands match what you're thinking ("prepare for PR" → `dk pr`)
2. **Fast**: Single-letter aliases for common operations
3. **Smart**: Defaults that make sense (auto-detect PR vs dev mode)
4. **Helpful**: Clear output and actionable error messages
5. **Workflow-oriented**: Commands match development patterns

## See Also

- [Quality Checks](quality-checks.md) - Detailed quality check documentation
- [Testing Guide](testing-guide.md) - Comprehensive testing documentation
- [Contributing](contributing.md) - How to contribute to DataKnobs