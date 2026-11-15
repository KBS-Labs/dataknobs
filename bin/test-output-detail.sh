#!/usr/bin/env bash
# What the quality gate shows about a test suite that failed.
#
# Sourced by bin/run-quality-checks.sh, and by tests/test_gate_failure_detail.py,
# which feeds it real pytest output -- a file of its own so the code that
# decides what a developer sees about a red suite is code a test can run.
#
# The verdict is the caller's, never this file's. The gate knows whether a
# suite failed from its exit status, at the one place it runs it, and calls
# report_failed_suite there. Nothing here looks at an output file to guess
# whether it records a failure: that guess was a list of patterns, and every
# failure the list did not anticipate printed nothing.

# The report flags the gate hands every pytest run. -rf lists FAILED tests,
# -rE lists ERROR ones -- a fixture that raises and a collection that fails
# produce no FAILED line -- and -rs gives the skip reasons the gate's skip
# summary reads.
# shellcheck disable=SC2034  # read by the scripts that source this file
GATE_PYTEST_REPORT_FLAGS="-rfEs"

# How many lines of a failed suite's output to show when it names no failed
# test. Enough to reach past pytest's summary line into the section above it,
# which is where an xdist collection mismatch prints the ids that differ.
GATE_FAILURE_TAIL_LINES=15

# The useful part of a failed suite's output.
#
# With FAILED lines, those and any ERROR lines beside them: the names of what
# broke. Without any, the tail -- a collection error, a fixture error, an
# import error, a usage error and an internal error all fail a run without a
# FAILED line, and the tail is where each states its cause. Only ever asked
# about a suite that failed, so it never prints nothing.
print_test_output_detail() {
    local clean failed
    clean=$(sed 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null || true)
    if [ -z "$clean" ]; then
        echo "    (the run wrote no output to $1)"
        return 0
    fi
    failed=$(printf '%s\n' "$clean" | grep -E '^FAILED ' || true)
    if [ -n "$failed" ]; then
        printf '%s\n' "$clean" | grep -E '^(FAILED|ERROR) ' | sed 's/^/    /'
    else
        printf '%s\n' "$clean" | tail -n "$GATE_FAILURE_TAIL_LINES" | sed 's/^/    /'
    fi
}

# Called where a suite's failure is decided: show it, and record it for the
# summary at the end of the run.
#
# One line appended per failed suite. The unit suites run as background jobs,
# so this is how a verdict reached in a subshell gets back to the parent; a
# single short write to a file opened for appending is not interleaved.
report_failed_suite() {
    local output_file="$1" record="$2"
    print_test_output_detail "$output_file"
    printf '%s\n' "$output_file" >> "$record"
}

# The summary's failure details: each suite report_failed_suite recorded,
# labelled the way its output file is named. Prints nothing when none failed.
print_failure_details() {
    local record="$1" output_file name
    [ -s "$record" ] || return 0
    while IFS= read -r output_file; do
        [ -n "$output_file" ] || continue
        name=$(basename "$output_file" .txt)
        echo -e "  ${YELLOW:-}${name%%-test-output-*} (${name##*-output-}):${NC:-}"
        print_test_output_detail "$output_file"
    done < "$record"
}
