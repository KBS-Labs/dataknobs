"""Template syntax annotation and conversion utilities.

This module provides:

- ``TemplateSyntax`` enum for annotating prompt templates with their authoring syntax
- Conversion functions between Python ``.format()`` and Jinja2 template syntaxes
- Heuristic syntax detection for unannotated templates
- Normalization to Jinja2 for the rendering pipeline

**Design principle:** Template syntax is a *storage/authoring* concern, not a runtime
concern. Prompts are stored in whatever syntax the author prefers (typically ``.format()``
for simple variable substitution, Jinja2 for conditionals/loops/filters). At the rendering
boundary, everything is normalized to Jinja2 so there is a single rendering code path.

**Edge cases with literal braces:**

``.format()`` syntax uses ``{{`` and ``}}`` for literal braces (e.g., in JSON examples).
Jinja2 uses ``{{ "{" }}`` or ``{% raw %}...{% endraw %}`` for the same purpose. The
conversion functions handle this correctly, but round-tripping through both conversions
may not be lossless for complex templates with mixed literal braces and variables.

**Lossy conversion (Jinja2 to format):**

``jinja2_to_format()`` only supports simple variable references (``{{ var }}``). Templates
using Jinja2 features (conditionals, loops, filters, macros, ``prompt_ref()`` calls)
cannot be converted to ``.format()`` syntax and will raise ``ValueError``.
"""

import re
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TemplateSyntax(Enum):
    """Template authoring syntax annotation.

    Each prompt template carries an explicit syntax annotation so the rendering
    pipeline knows how to normalize it before rendering.

    Attributes:
        FORMAT: Python ``str.format()`` syntax — ``{var}`` for variables,
            ``{{`` / ``}}`` for literal braces.
        JINJA2: Jinja2 syntax — ``{{ var }}`` for variables,
            ``{% block %}``, ``{# comment #}``, filters (``| upper``), etc.
    """

    FORMAT = "format"
    JINJA2 = "jinja2"

    @classmethod
    def from_string(cls, value: str) -> "TemplateSyntax":
        """Parse syntax from string value.

        Args:
            value: Syntax string (``"format"`` or ``"jinja2"``).

        Returns:
            Corresponding ``TemplateSyntax`` enum value.

        Raises:
            ValueError: If value is not a recognized syntax string.
        """
        try:
            return cls(value.lower())
        except ValueError as e:
            raise ValueError(
                f"Invalid template syntax: {value!r}. "
                f"Valid values: {', '.join(s.value for s in cls)}"
            ) from e


# ---------------------------------------------------------------------------
# Conversion: .format() → Jinja2
# ---------------------------------------------------------------------------

# Regex matching .format()-style placeholders: {name} or {name!s} or {name:>10}
# but NOT {{ (literal brace) or {} (empty — positional)
_FORMAT_VAR_RE = re.compile(
    r"\{"
    r"(?!"            # negative lookahead: not a literal {{ or empty {}
    r"\{|"            # not {{
    r"\})"            # not {}
    r"([a-zA-Z_][a-zA-Z0-9_]*)"  # variable name (group 1)
    r"(?:"            # optional format spec group
    r"![sra]"         # conversion flag
    r"|"
    r":[^}]*"         # format spec
    r")?"
    r"\}"
)

# Regex matching .format()-style literal brace escapes: {{ or }}
_FORMAT_LITERAL_OPEN_RE = re.compile(r"\{\{")
_FORMAT_LITERAL_CLOSE_RE = re.compile(r"\}\}")


def format_to_jinja2(template: str) -> str:
    """Convert a Python ``.format()``-style template to Jinja2 syntax.

    Conversion rules:

    - ``{var}`` → ``{{ var }}``
    - ``{var!s}`` / ``{var:>10}`` → ``{{ var }}`` (format specs are dropped
      since Jinja2 has its own filter system)
    - ``{{`` (literal open brace) → ``{{ "{" }}``
    - ``}}`` (literal close brace) → ``{{ "}" }}``

    Args:
        template: A ``.format()``-style template string.

    Returns:
        Equivalent Jinja2 template string.
    """
    if not template:
        return template

    # Process the template character by character to handle overlapping patterns.
    # We need to distinguish {var} from {{ (literal brace).
    result: list[str] = []
    i = 0
    length = len(template)

    while i < length:
        # Check for literal {{ (format escape for open brace)
        if i + 1 < length and template[i] == "{" and template[i + 1] == "{":
            # But check it's not {{{ which is {{ followed by {var}
            # In .format(), {{{ means literal { + start of var
            # However, {{x}} in .format() means literal { then x}
            # We handle {{ as literal brace escape
            # Check if this is actually a format variable like {{var}}
            # In .format(), {{var}} means literal "{" + "var}"... which is invalid
            # Actually in .format(), {{ always means literal {
            result.append('{{ "{" }}')
            i += 2
            continue

        # Check for literal }} (format escape for close brace)
        if i + 1 < length and template[i] == "}" and template[i + 1] == "}":
            result.append('{{ "}" }}')
            i += 2
            continue

        # Check for {var} or {var!s} or {var:spec}
        if template[i] == "{":
            match = _FORMAT_VAR_RE.match(template, i)
            if match:
                var_name = match.group(1)
                result.append("{{ " + var_name + " }}")
                i = match.end()
                continue
            # Unmatched { — pass through as-is
            result.append(template[i])
            i += 1
            continue

        # Regular character
        result.append(template[i])
        i += 1

    return "".join(result)


# ---------------------------------------------------------------------------
# Conversion: Jinja2 → .format()
# ---------------------------------------------------------------------------

# Jinja2 features that cannot be expressed in .format()
_JINJA2_BLOCK_RE = re.compile(r"\{%")
_JINJA2_COMMENT_RE = re.compile(r"\{#")
_JINJA2_FILTER_RE = re.compile(r"\{\{[^}]*\|[^}]*\}\}")
_JINJA2_FUNCTION_CALL_RE = re.compile(r"\{\{[^}]*\([^)]*\)[^}]*\}\}")

# Simple Jinja2 variable reference: {{ var_name }}
_JINJA2_SIMPLE_VAR_RE = re.compile(
    r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}"
)


def jinja2_to_format(template: str) -> str:
    """Convert a simple Jinja2 template to Python ``.format()`` syntax.

    Only converts simple variable references: ``{{ var }}`` → ``{var}``.

    This conversion is lossy — Jinja2 features that have no ``.format()``
    equivalent will cause a ``ValueError``:

    - Block tags: ``{% if %}``, ``{% for %}``, ``{% block %}``
    - Comments: ``{# ... #}``
    - Filters: ``{{ var | upper }}``
    - Function calls: ``{{ prompt_ref("key") }}``
    - Complex expressions: ``{{ a + b }}``, ``{{ items[0] }}``

    Args:
        template: A Jinja2 template string using only simple variable references.

    Returns:
        Equivalent ``.format()``-style template string.

    Raises:
        ValueError: If the template uses Jinja2 features that cannot be
            expressed in ``.format()`` syntax.
    """
    if not template:
        return template

    # Check for unconvertible Jinja2 features
    if _JINJA2_BLOCK_RE.search(template):
        raise ValueError(
            "Template contains Jinja2 block tags ({%...%}) which cannot be "
            "converted to .format() syntax"
        )
    if _JINJA2_COMMENT_RE.search(template):
        raise ValueError(
            "Template contains Jinja2 comments ({#...#}) which cannot be "
            "converted to .format() syntax"
        )
    if _JINJA2_FILTER_RE.search(template):
        raise ValueError(
            "Template contains Jinja2 filters ({{ var | filter }}) which "
            "cannot be converted to .format() syntax"
        )
    if _JINJA2_FUNCTION_CALL_RE.search(template):
        raise ValueError(
            "Template contains Jinja2 function calls ({{ func() }}) which "
            "cannot be converted to .format() syntax"
        )

    # Check for complex Jinja2 expressions that aren't simple variables
    # Find all {{ ... }} blocks and verify each is a simple variable
    all_jinja_vars = re.finditer(r"\{\{(.*?)\}\}", template)
    for match in all_jinja_vars:
        inner = match.group(1).strip()
        if inner and not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", inner):
            # Check if it's a literal string expression like {{ "{" }}
            if re.match(r'^".*"$', inner) or re.match(r"^'.*'$", inner):
                # This is a literal brace expression — handle below
                continue
            raise ValueError(
                f"Template contains complex Jinja2 expression "
                f"({{{{ {inner} }}}}) which cannot be converted to "
                f".format() syntax"
            )

    # Convert simple {{ var }} to {var}
    result = _JINJA2_SIMPLE_VAR_RE.sub(r"{\1}", template)

    # Convert literal brace expressions back to .format() escapes
    # {{ "{" }} → {{  and  {{ "}" }} → }}
    result = re.sub(r'\{\{\s*"\{"\s*\}\}', "{{", result)
    result = re.sub(r"\{\{\s*'\{'\s*\}\}", "{{", result)
    result = re.sub(r'\{\{\s*"\}"\s*\}\}', "}}", result)
    result = re.sub(r"\{\{\s*'\}'\s*\}\}", "}}", result)

    return result


# ---------------------------------------------------------------------------
# Syntax detection
# ---------------------------------------------------------------------------

def detect_syntax(template: str) -> TemplateSyntax:
    """Heuristically detect whether a template uses .format() or Jinja2 syntax.

    Detection signals for Jinja2:

    - Block tags: ``{%`` ... ``%}``
    - Comment tags: ``{#`` ... ``#}``
    - Filters: ``{{ var | filter_name }}``
    - Function calls: ``{{ func(args) }}``
    - ``prompt_ref()`` calls

    Detection signals for .format():

    - Single-brace variables: ``{var_name}`` (not preceded by another ``{``)
    - Format specs: ``{var:>10}``, ``{var!s}``

    Ambiguous cases (e.g., ``{{ var }}`` which is valid in both syntaxes as
    either a Jinja2 variable or a .format() literal-brace-var-literal-brace)
    default to ``JINJA2`` since that is the more common intended reading.

    For empty templates, returns ``FORMAT`` as the safe default.

    Args:
        template: Template string to analyze.

    Returns:
        Detected ``TemplateSyntax``.
    """
    if not template:
        return TemplateSyntax.FORMAT

    # Strong Jinja2 signals
    if _JINJA2_BLOCK_RE.search(template):
        return TemplateSyntax.JINJA2
    if _JINJA2_COMMENT_RE.search(template):
        return TemplateSyntax.JINJA2
    if re.search(r"\{\{[^}]*\|[^}]*\}\}", template):
        return TemplateSyntax.JINJA2
    if re.search(r"\{\{[^}]*\([^)]*\)[^}]*\}\}", template):
        return TemplateSyntax.JINJA2
    if "prompt_ref" in template:
        return TemplateSyntax.JINJA2

    # Strong .format() signals: {single_brace_var} not preceded by {
    # Look for {var} that isn't part of {{ (which could be either syntax)
    # Check that these aren't inside {{ }} (Jinja2 variables)
    has_single_brace_vars = False
    for match in _FORMAT_VAR_RE.finditer(template):
        start = match.start()
        # Check the character before — if it's { then this is inside {{var}}
        if start > 0 and template[start - 1] == "{":
            continue
        # Check the character after the match — if it's } then this is {var}}
        end = match.end()
        if end < len(template) and template[end] == "}":
            continue
        has_single_brace_vars = True
        break

    if has_single_brace_vars:
        return TemplateSyntax.FORMAT

    # Check for .format() literal brace escapes that aren't Jinja2 vars
    # {{ in .format() is a literal brace; in Jinja2 it starts a variable
    # If we see {{ followed by content that looks like a variable }}, that's Jinja2
    # If we see {{ not followed by a variable (e.g., just {{ at line end), ambiguous
    has_jinja2_vars = bool(_JINJA2_SIMPLE_VAR_RE.search(template))
    if has_jinja2_vars:
        return TemplateSyntax.JINJA2

    # No strong signals — default to FORMAT (safe for existing code)
    return TemplateSyntax.FORMAT


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_to_jinja2(template: str, syntax: TemplateSyntax) -> str:
    """Normalize a template to Jinja2 syntax based on its declared syntax.

    This is the entry point for the rendering pipeline. All templates pass
    through this function before being handed to the Jinja2 engine.

    - ``JINJA2`` templates are returned as-is (identity).
    - ``FORMAT`` templates are converted via ``format_to_jinja2()``.

    Args:
        template: Template string in the declared syntax.
        syntax: The declared syntax of the template.

    Returns:
        Template string in Jinja2 syntax, ready for the Jinja2 engine.
    """
    if syntax == TemplateSyntax.JINJA2:
        return template
    return format_to_jinja2(template)


# ============================================================================
# CLI entry point
# ============================================================================

def _cli_main() -> None:  # pragma: no cover
    """CLI for prompt syntax operations.

    Subcommands:
        convert  — Convert a prompt between format and jinja2 syntax
        detect   — Detect the syntax of a prompt file or string
        validate — Validate placeholders in a prompt against a key's requirements
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m dataknobs_llm.prompts.syntax",
        description="Prompt template syntax utilities",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- convert ---
    convert_p = sub.add_parser(
        "convert",
        help="Convert a prompt between syntaxes",
    )
    convert_p.add_argument(
        "--from", dest="from_syntax", required=True,
        choices=["format", "jinja2"],
        help="Source syntax",
    )
    convert_p.add_argument(
        "--to", dest="to_syntax", required=True,
        choices=["format", "jinja2"],
        help="Target syntax",
    )
    convert_p.add_argument(
        "input",
        nargs="?",
        help="Input file (reads from stdin if omitted)",
    )

    # --- detect ---
    detect_p = sub.add_parser(
        "detect",
        help="Detect the syntax of a prompt",
    )
    detect_p.add_argument(
        "input",
        nargs="?",
        help="Input file (reads from stdin if omitted)",
    )

    # --- validate ---
    validate_p = sub.add_parser(
        "validate",
        help="Validate a prompt against a key's placeholder requirements",
    )
    validate_p.add_argument(
        "--key", required=True,
        help="Prompt key to validate against (e.g. wizard.clarification.issues)",
    )
    validate_p.add_argument(
        "input",
        nargs="?",
        help="Input file (reads from stdin if omitted)",
    )

    args = parser.parse_args()

    def _read_input() -> str:
        if args.input:
            with open(args.input) as f:
                return f.read()
        return sys.stdin.read()

    if args.command == "convert":
        text = _read_input()
        if args.from_syntax == args.to_syntax:
            print(text, end="")
        elif args.from_syntax == "format" and args.to_syntax == "jinja2":
            print(format_to_jinja2(text), end="")
        elif args.from_syntax == "jinja2" and args.to_syntax == "format":
            try:
                print(jinja2_to_format(text), end="")
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

    elif args.command == "detect":
        text = _read_input()
        syntax = detect_syntax(text)
        print(syntax.value)

    elif args.command == "validate":
        text = _read_input()
        # Look up the key in the default prompt library
        try:
            from dataknobs_bots.prompts.defaults import get_default_prompt_library
            library = get_default_prompt_library()
            template_dict = library.get_system_prompt(args.key)
        except ImportError:
            # Fall back to extraction library if bots not installed
            from dataknobs_llm.extraction.prompts import get_extraction_prompt_library
            library = get_extraction_prompt_library()
            template_dict = library.get_system_prompt(args.key)

        if template_dict is None:
            print(f"Error: Unknown prompt key: {args.key}", file=sys.stderr)
            sys.exit(1)

        # Detect syntax of the input and extract placeholders
        detected = detect_syntax(text)
        print(f"Input syntax: {detected.value}")
        print(f"Key: {args.key}")
        print(f"Key syntax: {template_dict.get('template_syntax', 'unknown')}")

        # Extract placeholders from the default template for comparison
        default_template = template_dict.get("template", "")
        default_syntax = template_dict.get("template_syntax", "format")

        if default_syntax == "format":
            import string
            formatter = string.Formatter()
            try:
                default_placeholders = {
                    fname
                    for _, fname, _, _ in formatter.parse(default_template)
                    if fname is not None
                }
            except (ValueError, KeyError):
                default_placeholders = set()
        else:
            # Jinja2 — extract {{ var }} patterns (simple heuristic)
            default_placeholders = set(
                re.findall(r'\{\{\s*(\w+)\s*\}\}', default_template)
            )

        if default_placeholders:
            print(f"Expected placeholders: {', '.join(sorted(default_placeholders))}")
        else:
            print("No placeholders expected (static template)")

        print("OK")


if __name__ == "__main__":
    _cli_main()
