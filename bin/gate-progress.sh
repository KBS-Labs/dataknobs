#!/usr/bin/env bash
# How the quality gate spends and reports its time: the pool its unit suites
# run through, and the clock on the lines it prints.
#
# Sourced by bin/run-quality-checks.sh, bin/docs-checks.sh and bin/validate.sh,
# and by tests/test_gate_progress.py -- a file of its own so the scheduling is
# code a test can run. Works under `set -u`, which validate.sh sets, and under
# bash 3.2 (stock macOS), which has neither `wait -n` nor EPOCHREALTIME.

# This file's directory, for the programs beside it. Resolved when sourced,
# since the callers source it from different working directories.
_GATE_PROGRESS_BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Whole seconds as a reader would say them: 42s, 7m26s, 1h02m05s.
# bin/quality-summary.py has the same function for the banner's rows;
# tests/test_gate_progress.py holds the two to one answer.
format_elapsed() {
    local total="$1"
    if [ "$total" -ge 3600 ]; then
        printf '%dh%02dm%02ds\n' $(( total / 3600 )) $(( total % 3600 / 60 )) $(( total % 60 ))
    elif [ "$total" -ge 60 ]; then
        printf '%dm%02ds\n' $(( total / 60 )) $(( total % 60 ))
    else
        printf '%ds\n' "$total"
    fi
}

# The time of day, and the time since the given `date +%s` stamp.
# One `date` call, so the two cannot straddle a second.
progress_stamp() {
    local clock now
    read -r clock now <<< "$(date '+%H:%M:%S %s')"
    printf '[%s +%s]\n' "$clock" "$(format_elapsed $(( now - $1 )))"
}

# run_pool <limit> <function> <item>...
#
# Call the function once per item, each in the background, at most <limit> at a
# time. Returns 1 if any call failed; a failure does not stop the others.
#
# A slot is refilled when any job ends. Waiting for the oldest job instead left
# the other slots idle for as long as that one job took, and the packages
# queued behind the slowest suite started only once it had finished. Without
# `wait -n` that takes a poll; a tenth of a second is small beside a suite.
run_pool() {
    local limit="$1" job="$2" item pid running status=0
    local pids=()
    shift 2
    for item in "$@"; do
        while :; do
            running=0
            for pid in ${pids[@]+"${pids[@]}"}; do
                if kill -0 "$pid" 2>/dev/null; then
                    running=$(( running + 1 ))
                fi
            done
            [ "$running" -ge "$limit" ] || break
            sleep 0.1
        done
        # Job control for this one line, so the job gets a process group of its
        # own: stop_on_signals (bin/stop-on-signal.sh) ends it by that group.
        # Without it a job also starts with SIGINT ignored, and nothing it runs
        # ever sees a Ctrl-C. No stdin, since a read of the terminal from
        # outside its foreground group would stop the job for good.
        set -m
        "$job" "$item" < /dev/null &
        set +m
        pids+=("$!")
    done
    # bash keeps the status of a job that has already ended, so this reads
    # every verdict and not only those of the jobs still running.
    for pid in ${pids[@]+"${pids[@]}"}; do
        wait "$pid" || status=1
    done
    return "$status"
}

# longest_suite_first [--stage unit|integration] <root> <package>...
#
# The packages, one per line, by how long each one's suite at that stage (unit
# unless named) last took: a long suite started last runs alone at the end.
#
# The times are the stage's suite_seconds (unit_tests or integration_tests) in
# the committed history of quality-summary.json (bin/quality-summary.py
# last-seconds), so they travel with the branch. They used to be gitignored
# files beside the run, which a new checkout does not have, and every worktree
# is a new checkout.
#
# One with no recorded time goes ahead of those that have one, since it may be
# the longest, and by how much test code it has at that stage. That is only a
# stand-in: the package with the most code is not the one that takes longest.
longest_suite_first() {
    local stage="unit" root pkg seconds size times
    if [ "${1:-}" = "--stage" ]; then
        stage="$2"
        shift 2
    fi
    case "$stage" in
        unit|integration) ;;
        *) echo "longest_suite_first: unknown stage '$stage'" >&2; return 1 ;;
    esac
    root="$1"
    shift
    times=$(uv run python "$_GATE_PROGRESS_BIN/quality-summary.py" last-seconds \
        --repo "$root" --check "${stage}_tests" "$@") || return 1
    for pkg in "$@"; do
        seconds=$(printf '%s\n' "$times" | awk -v p="$pkg" '$1 == p { print $2 }')
        if [ -n "$seconds" ]; then
            printf '1 %d %s\n' "$((10#$seconds))" "$pkg"
            continue
        fi
        size=0
        if [ "$stage" = "integration" ]; then
            if [ -d "$root/packages/$pkg/tests/integration" ]; then
                size=$(find "$root/packages/$pkg/tests/integration" -name '*.py' \
                    -exec cat {} + | wc -c)
            fi
        elif [ -d "$root/packages/$pkg/tests" ]; then
            size=$(find "$root/packages/$pkg/tests" -name '*.py' \
                -not -path '*/integration/*' -exec cat {} + | wc -c)
        fi
        printf '0 %d %s\n' "$size" "$pkg"
    done | sort -k1,1n -k2,2rn -k3 | cut -d' ' -f3
}

# resolve_gate_settings
#
# How many suites the gate runs at once, read from the environment and checked.
# Sets:
#
#   GATE_UNIT_JOBS         DK_GATE_UNIT_JOBS, default 3: slots in the unit pool,
#                          which also holds the workspace guards and, unless
#                          GATE_DOCS says otherwise, the documentation checks.
#   GATE_INTEGRATION_JOBS  DK_GATE_INTEGRATION_JOBS, default 1: integration
#                          suites run at once. 1 is one after another.
#   GATE_DOCS              DK_GATE_DOCS, default pool: "pool" gives the
#                          documentation checks a unit slot; "beside" runs them
#                          alongside the pool, taking none.
#
# Settings rather than constants so the gate can be timed at other values
# without an edit, and tuned on a machine with a different number of cores.
# The defaults are the values the gate has always used.
#
# An unset or empty variable takes its default. Any other value that is not
# one the setting accepts fails, naming the variable: falling back to the
# default would time a configuration nobody asked for and report it as the
# one they did.
resolve_gate_settings() {
    GATE_UNIT_JOBS="${DK_GATE_UNIT_JOBS:-3}"
    GATE_INTEGRATION_JOBS="${DK_GATE_INTEGRATION_JOBS:-1}"
    GATE_DOCS="${DK_GATE_DOCS:-pool}"
    _gate_positive_count DK_GATE_UNIT_JOBS "$GATE_UNIT_JOBS" || return 1
    _gate_positive_count DK_GATE_INTEGRATION_JOBS "$GATE_INTEGRATION_JOBS" || return 1
    case "$GATE_DOCS" in
        pool|beside) ;;
        *)
            echo "DK_GATE_DOCS must be 'pool' or 'beside', not '$GATE_DOCS'" >&2
            return 1
            ;;
    esac
}

# _gate_positive_count <variable name> <value>
#
# Whole decimal numbers from 1 up. Leading zeros are refused rather than read:
# bash arithmetic would take 010 as octal 8.
_gate_positive_count() {
    case "$2" in
        [1-9]|[1-9]*[0-9])
            case "$2" in
                *[!0-9]*) ;;
                *) return 0 ;;
            esac
            ;;
    esac
    echo "$1 must be a whole number from 1 up, not '$2'" >&2
    return 1
}

# run_beside <function> <item>
#
# Start the function on the item in the background, outside any pool, and set
# BESIDE_PID to wait on. Started the way run_pool starts a job, so
# stop_on_signals can stop it and it never reads the terminal.
run_beside() {
    set -m
    "$1" "$2" < /dev/null &
    set +m
    # shellcheck disable=SC2034  # the output: the caller reads it to wait on
    BESIDE_PID=$!
}

# clear_coverage DIR
#
# Remove every coverage report and data file in DIR. The artifacts directory
# outlives a run, and a run that selects no package writes no coverage of its
# own: without this it would report whatever an earlier run, possibly on
# another branch, had left behind.
clear_coverage() {
    rm -f "$1"/coverage*.xml "$1"/.coverage "$1"/.coverage.*
}
