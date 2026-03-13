---
name: feedback_validation_command
description: Use bin/validate.sh (not bin/dk check) for code validation
type: feedback
---

Use `bin/validate.sh` for code validation, not `bin/dk check`.

**Why:** User prefers `bin/validate.sh` and rejected `bin/dk check` when used for validation.

**How to apply:** After making code changes, run `bin/validate.sh <package>` (or `bin/validate.sh -f <package>` to auto-fix). Use `bin/test.sh` for running tests.
