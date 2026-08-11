#!/usr/bin/env bash
# Validate code before commits or releases

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Source the package discovery utility
source "$ROOT_DIR/bin/package-discovery.sh"

# Default values
TARGETS=()
QUICK=false
FIX=false
STATS=false
WORKSPACE_ONLY=false
PRINT_TARGETS=false
PRINT_FORMAT_TARGETS=false

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS] [TARGETS...]"
    echo ""
    echo "Validate code quality and catch common errors"
    echo ""
    echo "Arguments:"
    echo "  TARGETS               Packages, directories, or files to validate"
    echo "                        Can be:"
    echo "                        - Package name (e.g., 'common', 'utils')"
    echo "                        - Directory path (e.g., 'packages/utils/src')"
    echo "                        - File path (e.g., 'packages/utils/src/dataknobs_utils/file_utils.py')"
    echo "                        If not specified, validates all packages"
    echo ""
    echo "Options:"
    echo "  -q, --quick           Quick validation (skip slow checks)"
    echo "  -f, --fix             Attempt to auto-fix issues"
    echo "  -s, --stats           Show detailed error statistics"
    echo "  -w, --workspace       Also validate the code belonging to no package"
    echo "                        (tests/, bin/, src/, conftest.py). Additive: with"
    echo "                        no other target it validates that set alone, and"
    echo "                        alongside packages it adds it to them"
    echo "      --print-targets   Print the resolved target list and exit, running"
    echo "                        no checks. What the target set IS, for callers"
    echo "                        that would otherwise have to re-derive it"
    echo "      --print-format-targets"
    echo "                        The same, for the formatter. A separate list"
    echo "                        because the formatter's declared coverage is"
    echo "                        wider than the linter's — see the format step"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                        # Validate all packages"
    echo "  $0 utils                                  # Validate only utils package"
    echo "  $0 packages/utils/src                     # Validate specific directory"
    echo "  $0 packages/utils/src/dataknobs_utils/*.py  # Validate specific files"
    echo "  $0 -f                                     # Validate and fix issues"
    echo "  $0 -s data                                # Show error statistics for data package"
    echo "  $0 -w                                     # Validate only the code belonging to no package"
    echo "  $0 data -w                                # Validate the data package and that code"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -q|--quick)
            QUICK=true
            shift
            ;;
        -f|--fix)
            FIX=true
            shift
            ;;
        -s|--stats)
            STATS=true
            shift
            ;;
        -w|--workspace)
            WORKSPACE_ONLY=true
            shift
            ;;
        --print-targets)
            PRINT_TARGETS=true
            shift
            ;;
        --print-format-targets)
            PRINT_FORMAT_TARGETS=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            # Add to targets list
            TARGETS+=("$1")
            shift
            ;;
    esac
done

# Get all packages dynamically
# Collected with a read loop rather than the mapfile shellcheck suggests,
# because mapfile is bash 4+ and these scripts run on the stock macOS bash 3.2.
# The loop is also glob-safe, which the bare arr=($(cmd)) form is not.
#
# Captured into a variable first, then fed as a here-string. The obvious
# `done < <(discover_packages)` never examines the producer's exit status --
# a process substitution's status is not reported anywhere -- so a failing
# discovery yielded an empty list and the run carried on. The floor below
# would have caught the empty case only because --workspace targets are
# appended unconditionally; without them it validates nothing and reports
# success. A bare assignment propagates the status under `set -e`, which is
# the whole reason for the extra line.
ALL_PACKAGES=()
_discovered=$(discover_packages)
while IFS= read -r _pkg; do
    [[ -n "$_pkg" ]] && ALL_PACKAGES+=("$_pkg")
done <<< "${_discovered// /$'\n'}"

# Determine what to validate
VALIDATE_TARGETS=()
VALIDATE_PACKAGES=()

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    # No targets specified, validate all packages
    if [[ "$WORKSPACE_ONLY" != true ]]; then
        VALIDATE_PACKAGES=("${ALL_PACKAGES[@]}")
        for package in "${ALL_PACKAGES[@]}"; do
            if [[ -d "packages/$package/src" ]]; then
                VALIDATE_TARGETS+=("packages/$package/src")
            fi
        done
    fi
else
    # Process specified targets
    for target in "${TARGETS[@]}"; do
        if [[ -d "packages/$target" ]]; then
            # It's a package name
            VALIDATE_PACKAGES+=("$target")
            if [[ -d "packages/$target/src" ]]; then
                VALIDATE_TARGETS+=("packages/$target/src")
            fi
        elif [[ -d "$target" ]]; then
            # It's a directory
            VALIDATE_TARGETS+=("$target")
        elif [[ -f "$target" ]]; then
            # It's a file
            VALIDATE_TARGETS+=("$target")
        else
            # Try glob expansion
            shopt -s nullglob
            # Unquoted on purpose: $target is a glob and this line is what
            # expands it. nullglob above turns a non-match into zero words.
            # shellcheck disable=SC2206
            files=($target)
            shopt -u nullglob
            if [[ ${#files[@]} -gt 0 ]]; then
                VALIDATE_TARGETS+=("${files[@]}")
            else
                # stderr, because stdout is a data channel under --print-targets:
                # a caller splits that output into a target list, and prose on the
                # same stream parses into targets that do not exist.
                echo -e "${YELLOW}Warning: Target '$target' not found${NC}" >&2
            fi
        fi
    done
fi

# The code belonging to no package, so a loop over packages/* reached none of
# it: the workspace guards asserting the toolchain is coherent, the root conftest
# every test run imports, the workspace shim, and bin/ — which holds the checkers
# deciding whether a pull request passes, among them the documentation-mirror and
# internal-label guards. Nothing linted the linters.
#
# Declared once in package-discovery.sh because fix.sh and dk need the same
# answer. tests/test_toolchain_consistency.py compares it against every tracked
# *.py, so a new directory of first-party Python fails there rather than quietly
# joining the set nothing checks; what stays out is declared in that file, with
# its size.
#
# Outside the branch above, and that placement is the point. This half used to be
# appended only when no target was named, so naming one dropped it — and the gate
# narrows by package name, so every pull request touching a package validated
# packages/*/src alone. That is every pull request that edits the ruff config,
# because the config is a global trigger that marks all ten packages changed. The
# change that started linting bin/ was itself one of them: bin/ went unlinted on
# the run that was supposed to prove it clean. So --workspace adds this set to
# whatever was named rather than replacing it, and with nothing named it is the
# whole answer.
if [[ ${#TARGETS[@]} -eq 0 || "$WORKSPACE_ONLY" == true ]]; then
    for workspace_target in $(workspace_targets); do
        VALIDATE_TARGETS+=("$workspace_target")
    done
fi

if [[ ${#VALIDATE_TARGETS[@]} -eq 0 ]]; then
    echo -e "${RED}No valid targets found to validate${NC}"
    exit 1
fi

# Resolved, before any check runs. A caller that needs to know what this script
# validates can ask it instead of re-deriving the answer from its source — which
# is what the coverage guard in tests/test_toolchain_consistency.py did, reading
# the appends as text and so crediting targets that a conditional could skip.
if [[ "$PRINT_TARGETS" == true ]]; then
    printf '%s\n' "${VALIDATE_TARGETS[@]}"
    exit 0
fi

# The formatter's population, resolved the same way and kept separate.
#
# It is not VALIDATE_TARGETS. That set is the *linter's*, and it deliberately
# omits every cell whose ruff tier is deferred — packages/*/tests among them.
# The quality contract enforces `format` at ceiling 0 on all ten of its cells,
# so borrowing the linter's list here checked 597 of 1,471 files and printed a
# clean verdict over the other 874.
#
# Composed from format_subdirs rather than restated, so a directory added to
# the formatter's coverage arrives here, in fix.sh and in `dk format` at once.
#
# Derived from VALIDATE_TARGETS rather than resolved a second time, so the two
# cannot disagree about which packages are in scope: each `packages/<pkg>/src`
# the package loop contributed widens to that package's whole format set, and
# everything else — a directory or file the caller named, the workspace set —
# passes through as given. A path the caller named directly is not widened,
# because naming one is the statement that it is what should be read.
FORMAT_TARGETS=()
for target in "${VALIDATE_TARGETS[@]}"; do
    _widened=false
    for package in ${VALIDATE_PACKAGES[@]+"${VALIDATE_PACKAGES[@]}"}; do
        if [[ "$target" == "packages/$package/src" ]]; then
            for _subdir in $(format_subdirs); do
                if [[ -d "packages/$package/$_subdir" ]]; then
                    FORMAT_TARGETS+=("packages/$package/$_subdir")
                fi
            done
            _widened=true
            break
        fi
    done
    # `if` rather than `[[ ... ]] && ...`: as the last command in the loop body
    # that form makes the loop's status the test's, so a final iteration that
    # widened would leave the `for` returning 1 and errexit would abort a run
    # scoped to a single package.
    if [[ "$_widened" == false ]]; then
        FORMAT_TARGETS+=("$target")
    fi
done

if [[ "$PRINT_FORMAT_TARGETS" == true ]]; then
    printf '%s\n' "${FORMAT_TARGETS[@]}"
    exit 0
fi

# Stats mode - show error statistics and exit
if [[ "$STATS" == true ]]; then
    echo -e "${BLUE}Error Statistics for targets:${NC}"
    echo -e "${YELLOW}==============================${NC}"
    
    # Ruff statistics
    echo -e "\n${BLUE}Ruff Linting Statistics:${NC}"
    for target in "${VALIDATE_TARGETS[@]}"; do
        echo -e "${YELLOW}  $target:${NC}"
        uv run ruff check "$target" --statistics --config "$ROOT_DIR/pyproject.toml" 2>/dev/null || true
    done

    # MyPy statistics: a breakdown by error code, which is a different product
    # from a verdict and so is produced here rather than by the contract.
    #
    # There used to be a "Total MyPy Errors" block below it, counting with
    # `grep -c "error:"`. That was a third implementation of a number the
    # contract already produces from `measure_mypy`, and two counters of the
    # same thing are two answers waiting to disagree — an unanchored substring
    # match against an anchored `path:line: error:` one. The count belongs to
    # whoever compares it against a ceiling.
    if [[ "$QUICK" != true ]]; then
        echo -e "\n${BLUE}MyPy Type Checking Statistics:${NC}"
        for target in "${VALIDATE_TARGETS[@]}"; do
            echo -e "${YELLOW}  $target:${NC}"
            uv run mypy "$target" --config-file "$ROOT_DIR/pyproject.toml" 2>&1 | \
                grep "error:" | \
                sed 's/.*error: //' | \
                sed 's/  \[/\n[/' | \
                grep '^\[' | \
                sed 's/\[//' | \
                sed 's/\]//' | \
                sort | uniq -c | sort -rn || echo "    No type errors found"
        done

        echo -e "\n${BLUE}Per-cell totals against their ceilings:${NC}"
        echo -e "  ${YELLOW}bin/quality-contract.py check --tool mypy${NC}"
    fi

    # TODO/FIXME count
    echo -e "\n${BLUE}TODO/FIXME Comments:${NC}"
    for target in "${VALIDATE_TARGETS[@]}"; do
        if [[ -f "$target" ]]; then
            count=$(grep -c "TODO\|FIXME" "$target" 2>/dev/null || echo 0)
            echo -e "  ${YELLOW}$target:${NC} $count"
        elif [[ -d "$target" ]]; then
            # Temporarily disable pipefail since grep -c returns exit 1 when count is 0
            set +o pipefail
            count=$(find "$target" -name "*.py" -exec grep -c "TODO\|FIXME" {} + 2>/dev/null | awk -F: '{sum += $2} END {print sum ? sum : 0}')
            set -o pipefail
            echo -e "  ${YELLOW}$target:${NC} $count"
        fi
    done
    
    exit 0
fi

echo -e "${YELLOW}Validating targets...${NC}"

# Track overall status
FAILED=false

# 1. Check Python syntax
echo -e "\n${BLUE}1. Checking Python syntax...${NC}"
for target in "${VALIDATE_TARGETS[@]}"; do
    echo -e "${YELLOW}  Checking $target...${NC}"
    
    if [[ -f "$target" ]]; then
        # Single file
        if ! uv run python -m py_compile "$target" 2>/dev/null; then
            echo -e "${RED}    ✗ Syntax error in $target${NC}"
            FAILED=true
        fi
    elif [[ -d "$target" ]]; then
        # Directory - find all Python files
        while IFS= read -r -d '' file; do
            if ! uv run python -m py_compile "$file" 2>/dev/null; then
                echo -e "${RED}    ✗ Syntax error in $file${NC}"
                FAILED=true
            fi
        done < <(find "$target" -name "*.py" -print0)
    fi
done

if [[ "$FAILED" == false ]]; then
    echo -e "${GREEN}  ✓ All Python files have valid syntax${NC}"
fi

# 2. Run ruff linting
echo -e "\n${BLUE}2. Running ruff linting...${NC}"

for target in "${VALIDATE_TARGETS[@]}"; do
    echo -e "${YELLOW}  Checking $target...${NC}"
    
    if [[ "$FIX" == true ]]; then
        # Run ruff with auto-fix (matching fix.sh behavior)
        if uv run ruff check "$target" --fix --no-unsafe-fixes --config "$ROOT_DIR/pyproject.toml"; then
            echo -e "${GREEN}    ✓ Ruff checks passed${NC}"
        else
            echo -e "${YELLOW}    ⚠ Some issues remain that need manual fixing${NC}"
            FAILED=true
        fi
    else
        if uv run ruff check "$target" --no-fix --config "$ROOT_DIR/pyproject.toml"; then
            echo -e "${GREEN}    ✓ Ruff checks passed${NC}"
        else
            echo -e "${RED}    ✗ Ruff found issues${NC}"
            FAILED=true
        fi
    fi
done

# 3. Check formatting
#
# The formatter has been configured in pyproject.toml and named in the published
# docs since the beginning, and until now nothing ran it: 1,128 of 1,471 tracked
# files were unformatted while the docs told contributors it was the standard.
# This is the step that makes the declaration true.
#
# Always with --config, and no --all-errors branch. The formatter has no
# equivalent of "show me the suppressed findings too" -- there is one formatted
# form -- so the two-branch shape the linter carries above would offer a choice
# between the real answer and a differently-configured one.
echo -e "\n${BLUE}3. Checking code formatting...${NC}"

for target in "${FORMAT_TARGETS[@]}"; do
    echo -e "${YELLOW}  Checking $target...${NC}"

    if [[ "$FIX" == true ]]; then
        if uv run ruff format "$target" --config "$ROOT_DIR/pyproject.toml"; then
            echo -e "${GREEN}    ✓ Formatting applied${NC}"
        else
            echo -e "${RED}    ✗ Formatter failed${NC}"
            FAILED=true
        fi
    else
        if uv run ruff format --check "$target" --config "$ROOT_DIR/pyproject.toml"; then
            echo -e "${GREEN}    ✓ Formatting is clean${NC}"
        else
            echo -e "${RED}    ✗ Files need formatting — run with -f, or bin/fix.sh${NC}"
            FAILED=true
        fi
    fi
done

# 4. Check imports (only for packages)
if [[ ${#VALIDATE_PACKAGES[@]} -gt 0 ]]; then
    echo -e "\n${BLUE}4. Checking imports...${NC}"
    for package in "${VALIDATE_PACKAGES[@]}"; do
        echo -e "${YELLOW}  Checking $package...${NC}"

        # Try to import the package
        # Special case: legacy package is named "dataknobs" not "dataknobs_legacy"
        if [[ "$package" == "legacy" ]]; then
            PACKAGE_NAME="dataknobs"
        else
            PACKAGE_NAME="dataknobs_${package//-/_}"
        fi

        if uv run python -c "import $PACKAGE_NAME" 2>/dev/null; then
            echo -e "${GREEN}    ✓ Package imports successfully${NC}"
        else
            echo -e "${RED}    ✗ Failed to import $PACKAGE_NAME${NC}"
            FAILED=true
        fi
    done
fi

# Run mypy over one target and return ITS exit status, echoing whatever it said.
#
# The verdict used to come from `mypy … 2>&1 | grep -E "(error|Error)"`, which
# reports the status of the pipeline. This script sets `pipefail` (line 4), and
# mypy exits non-zero exactly when it has findings — so a real type error made
# the pipeline non-zero, the `if` took the *else* branch, and the run printed
# "Type checks passed" directly beneath the errors grep had just echoed. FAILED
# was never set, so mypy could not fail a validation run. Reading the exit status
# is the whole fix; the output is printed rather than matched.
#
# Now reached only for a path outside the contract's population, where there is
# no ceiling and so any finding is a breach. Everything inside a cell goes
# through the contract — see the step below.
run_mypy() {
    local output rc
    output=$(uv run mypy "$@" --config-file "$ROOT_DIR/pyproject.toml" 2>&1) && rc=0 || rc=$?
    if [[ -n "$output" ]]; then
        printf '%s\n' "$output"
    fi
    return "$rc"
}

# Whether a value is already in the rest of the arguments.
_contains() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

# 5. Type checking with mypy (unless quick mode)
#
# The verdict is the quality contract's, not this script's. There is one mypy
# configuration now, and under it the tree has thousands of findings — so "any
# error means FAILED", which is what this step used to say, would fail every
# pull request touching a package that has not been adopted yet. The number that
# decides is `measured <= ceiling`, per cell, and that comparison already exists
# in bin/quality-contract.py. Calling it is what keeps the two from disagreeing.
#
# They *did* disagree, in the direction that matters least and hurts most: a new
# finding in a transitional package passed here and failed the gate, because
# this step read a second configuration under which the code was clean. A
# developer validated locally, saw green, pushed, and learned about it from CI.
#
# A ceiling is a whole-cell property, so a caller naming one file inside a cell
# gets that whole cell measured — a partial count compared against a whole-cell
# ceiling is not a verdict. That is also what retires --follow-imports=skip for
# this case: its reason for existing was to stop a single-file run dragging in
# the rest of the tree, and the cell is now the unit either way.
#
# Which paths are in which cell is asked, never matched here: a second copy of
# the contract's matcher is a second answer waiting to disagree with the one the
# ceilings were measured under.
if [[ "$QUICK" != true ]]; then
    echo -e "\n${BLUE}5. Running mypy type checking...${NC}"

    MYPY_CELLS=()
    MYPY_OUTSIDE=()
    mypy_ok=true

    # Captured into a variable rather than read from a process substitution, for
    # the reason recorded at the package-discovery loop above: a substitution's
    # exit status is reported nowhere, so a failing classifier would yield an
    # empty list and this step would check nothing and pass.
    _mypy_scope=$(uv run python "$ROOT_DIR/bin/quality-contract.py" \
        scope --tool mypy "${VALIDATE_TARGETS[@]}")
    while IFS=$'\t' read -r _kind _path _cell; do
        case "$_kind" in
            cell)
                if ! _contains "$_cell" ${MYPY_CELLS[@]+"${MYPY_CELLS[@]}"}; then
                    MYPY_CELLS+=("$_cell")
                fi
                ;;
            unmeasured)
                # Reported rather than dropped. Silently skipping it is how a
                # caller comes to believe a directory was type-checked.
                echo -e "${YELLOW}  Skipping $_path — $_cell is not type-checked by the contract${NC}"
                ;;
            outside)
                MYPY_OUTSIDE+=("$_path")
                ;;
            *)
                # Including the empty kind, which is what a truncated
                # classification looks like. Failing is the point: a target the
                # classifier did not place is one nothing below type-checks, and
                # skipping it would report success over unread code.
                echo -e "${RED}  Could not place '$_path' — the contract reported scope '$_kind'${NC}"
                mypy_ok=false
                ;;
        esac
    done <<< "$_mypy_scope"

    if [[ ${#MYPY_CELLS[@]} -gt 0 ]]; then
        echo -e "${YELLOW}  Checking ${MYPY_CELLS[*]}...${NC}"
        _cell_args=()
        for _cell in "${MYPY_CELLS[@]}"; do
            _cell_args+=(--cell "$_cell")
        done
        if ! uv run python "$ROOT_DIR/bin/quality-contract.py" \
            check --tool mypy --show-findings "${_cell_args[@]}"; then
            mypy_ok=false
        fi
    fi

    # Outside every cell there is no ceiling to be within, so any finding is a
    # breach. Imports are not followed: the path is outside the population, and
    # following them would put the population's backlog into its verdict.
    for _path in ${MYPY_OUTSIDE[@]+"${MYPY_OUTSIDE[@]}"}; do
        echo -e "${YELLOW}  Checking $_path (outside the contract, so it must be clean)...${NC}"
        run_mypy "$_path" --follow-imports=skip || mypy_ok=false
    done

    if [[ "$mypy_ok" == true ]]; then
        echo -e "${GREEN}    ✓ Type checks passed${NC}"
    else
        echo -e "${RED}    ✗ Type errors found${NC}"
        FAILED=true
    fi
fi

# 6. Check for common issues
echo -e "\n${BLUE}6. Checking for common issues...${NC}"

# Check for print statements (with exceptions for legitimate uses)
echo -e "${YELLOW}  Checking for print statements...${NC}"
HAS_PRINTS=false
PRINT_RESULTS=()

# Files that are allowed to have print statements
# CLI tools and debuggers need to output to users
PRINT_EXCEPTIONS=(
    "*/cli/main.py"                 # CLI interface uses Rich console.print
    "*/api/advanced.py"            # Debugger class needs user output
    "*/prompts/syntax.py"          # CLI tool for prompt syntax conversion/detection
    "*/tooling/model_limits.py"    # CLI maintainer tool: stdout is its product (drift report / status)
    # Every script in bin/ is a developer tool whose stdout is its product, and
    # for three it is the *return value*: run-quality-checks.sh captures
    # changed-packages.py and package-hashes.py into shell variables, and
    # validate.sh reads find_print_statements.py line by line a few dozen lines
    # below this comment. Routing those through logging would not improve the
    # output, it would empty it and break the caller.
    #
    # The directory rather than those three, and that is the loose part of this
    # entry: it exempts every future file here too, including one that turns out
    # not to be a CLI at all. It stands because bin/ is the scripts directory by
    # definition and all ten current files are entry points, so enumerating them
    # would be a list to forget rather than a constraint to meet. The narrower
    # form is available the day that stops being true.
    "bin/*.py"
    # Imported by pytest before any logging is configured, so a logger call here
    # goes nowhere. print is the mechanism conftest has.
    "conftest.py"
)

# A test file, by name — the naming convention pytest collects on.
#
# One predicate for both branches below, because there were two and they
# disagreed: files were skipped on "*test*" and directory walks on "*/test*",
# which is the same question asked two ways. Both were wider than the question,
# and the directory form exempted every shipped module beneath a testing/
# package — dataknobs_common.testing and its siblings are library code consumers
# import, the constructs the house rules point at instead of mocks — from the
# print check, silently and for as long as the check has existed. Eleven files.
# None of them print today, so this closes a latent hole rather than a live one.
#
# conftest.py is deliberately not matched here. It is exempt, but through
# PRINT_EXCEPTIONS, where the reason is written down — under the old glob its
# entry was unreachable, describing a suppression the glob had already applied.
is_test_file() {
    local base
    base="$(basename "$1")"
    [[ "$base" == test_*.py || "$base" == *_test.py ]]
}

# Function to check if a file should be excluded
should_exclude_file() {
    local file="$1"
    for exception in "${PRINT_EXCEPTIONS[@]}"; do
        # The right-hand side is a PATTERN, not a string — PRINT_EXCEPTIONS
        # holds globs such as "bin/*.py". Quoting it turns every entry into a
        # literal filename match, so every exception silently stops applying
        # and the check reports findings the gate does not have.
        # shellcheck disable=SC2053
        if [[ "$file" == $exception ]]; then
            return 0  # Should exclude (true)
        fi
    done
    return 1  # Should not exclude (false)
}

# Use Python AST parser to find print statements in actual code
# (ignoring comments, docstrings, and string literals)
for target in "${VALIDATE_TARGETS[@]}"; do
    # Collect files to check
    if [[ -f "$target" ]]; then
        if [[ "$(basename "$target")" != "__init__.py" ]] && \
           ! is_test_file "$target" && \
           ! should_exclude_file "$target"; then
            check_files=("$target")
        else
            check_files=()
        fi
    elif [[ -d "$target" ]]; then
        # Find Python files excluding __init__.py and test files
        check_files=()
        while IFS= read -r -d '' file; do
            if ! is_test_file "$file" && ! should_exclude_file "$file"; then
                check_files+=("$file")
            fi
        done < <(find "$target" -name "*.py" ! -name "__init__.py" -print0)
    fi

    # Run the print finder on collected files
    if [[ ${#check_files[@]} -gt 0 ]]; then
        while IFS= read -r line; do
            if [[ -n "$line" ]]; then
                PRINT_RESULTS+=("$line")
                HAS_PRINTS=true
            fi
        done < <(uv run python "$ROOT_DIR/bin/find_print_statements.py" "${check_files[@]}" 2>/dev/null || true)
    fi
done

if [[ "$HAS_PRINTS" == false ]]; then
    echo -e "${GREEN}    ✓ No print statements found${NC}"
else
    echo -e "${RED}    ✗ Found print statements (use logging instead):${NC}"

    # Group results by file (simple approach without associative arrays)
    current_file=""
    shown=0
    count_in_file=0

    for result in "${PRINT_RESULTS[@]}"; do
        # Parse the result: filepath:line:col:content
        file="${result%%:*}"
        rest="${result#*:}"
        line="${rest%%:*}"
        rest="${rest#*:}"
        # The column field is stepped over rather than captured: the file and
        # line are what the report shows, and content is what follows it.
        content="${rest#*:}"

        # Check if this is a new file
        if [[ "$file" != "$current_file" ]]; then
            current_file="$file"
            count_in_file=0

            # Only show up to 10 files
            if [[ $shown -lt 10 ]]; then
                echo -e "${RED}      - $file:${NC}"
                ((shown++))
            fi
        fi

        # Show up to 3 occurrences per file
        if [[ $count_in_file -lt 3 ]] && [[ $shown -le 10 ]]; then
            # Trim and truncate line content
            trimmed=$(echo "$content" | cut -c1-60)
            echo -e "${RED}        Line $line: $trimmed${NC}"
            ((count_in_file++))
        fi
    done

    # Count unique files
    total_files=$(echo "${PRINT_RESULTS[@]}" | tr ' ' '\n' | cut -d: -f1 | sort -u | wc -l | tr -d ' ')
    if [[ $total_files -gt 10 ]]; then
        echo -e "${RED}      ... and $(($total_files - 10)) more files${NC}"
    fi
    FAILED=true
fi

# Check for TODO/FIXME comments
echo -e "${YELLOW}  Checking for TODO/FIXME comments...${NC}"
TODO_COUNT=0
TODO_FILES=()

for target in "${VALIDATE_TARGETS[@]}"; do
    if [[ -f "$target" ]]; then
        # grep returns exit 1 when no match, handle gracefully
        # Disable pipefail temporarily for this command
        set +o pipefail
        count=$(grep -E "TODO|FIXME" "$target" 2>/dev/null | wc -l | tr -d ' ')
        set -o pipefail
        # Ensure count is a number (default to 0 if empty)
        count=${count:-0}
        if [[ $count -gt 0 ]]; then
            TODO_FILES+=("$target:$count")
            TODO_COUNT=$((TODO_COUNT + count))
        fi
    elif [[ -d "$target" ]]; then
        # Find files with TODO/FIXME and their counts
        while IFS= read -r -d '' file; do
            # Disable pipefail temporarily for this command
            set +o pipefail
            count=$(grep -E "TODO|FIXME" "$file" 2>/dev/null | wc -l | tr -d ' ')
            set -o pipefail
            # Ensure count is a number (default to 0 if empty)
            count=${count:-0}
            if [[ $count -gt 0 ]]; then
                TODO_FILES+=("$file:$count")
                TODO_COUNT=$((TODO_COUNT + count))
            fi
        done < <(find "$target" -name "*.py" -print0)
    fi
done

if [[ "$TODO_COUNT" -eq 0 ]]; then
    echo -e "${GREEN}    ✓ No TODO/FIXME comments found${NC}"
elif [[ "$TODO_COUNT" -gt 0 ]]; then
    echo -e "${YELLOW}    ⚠ Found $TODO_COUNT TODO/FIXME comments:${NC}"
    # Show up to 10 files with TODO/FIXME
    shown=0
    for file_info in "${TODO_FILES[@]}"; do
        if [[ $shown -lt 10 ]]; then
            file="${file_info%:*}"
            count="${file_info##*:}"
            echo -e "${YELLOW}      - $file ($count occurrences):${NC}"
            # Show first 3 TODO/FIXME comments with line numbers
            grep -E -n "TODO|FIXME" "$file" 2>/dev/null | head -3 | while IFS=: read -r line_num line_content; do
                # Trim whitespace and show a preview
                trimmed=$(echo "$line_content" | sed 's/^[[:space:]]*//' | cut -c1-60)
                echo -e "${YELLOW}        Line $line_num: $trimmed${NC}"
            done || true
            shown=$((shown + 1))
        fi
    done
    if [[ ${#TODO_FILES[@]} -gt 10 ]]; then
        echo -e "${YELLOW}      ... and $((${#TODO_FILES[@]} - 10)) more files${NC}"
    fi
fi

# Check for internal-tracking-label leakage (Item NNN, RCN, PR #NNN, etc.)
# This is a tree-wide invariant: a single hard-fail invocation over the
# full packages/*/src + packages/*/tests scope (fast; runs in all modes),
# independent of the per-target loop above.  Allowlist of genuine
# fixture/data values lives in bin/internal-label-allowlist.txt.
echo -e "${YELLOW}  Checking for internal-tracking-label leakage...${NC}"
if uv run python "$ROOT_DIR/bin/check-internal-labels.py"; then
    : # script prints its own success line
else
    FAILED=true
fi

# Summary
echo -e "\n${YELLOW}Validation Summary:${NC}"
echo -e "${YELLOW}==================${NC}"

if [[ "$FAILED" == true ]]; then
    echo -e "${RED}❌ Validation failed!${NC}"
    echo -e "\nTo fix issues:"
    echo -e "  1. Run: ./bin/validate.sh -f"
    echo -e "  2. Run: ./bin/fix.sh"
    echo -e "  3. Fix remaining issues manually"
    exit 1
else
    echo -e "${GREEN}✅ All validations passed!${NC}"
    exit 0
fi
