#!/usr/bin/env bash
# What a script that runs suites does when it is told to stop.
#
# Sourced by bin/run-quality-checks.sh, bin/test.sh and
# bin/run-integration-tests.sh, and by tests/test_stop_on_signal.py.
#
# A trap whose handler returns does not stop a script: bash runs the handler and
# goes on to the next statement. `trap cleanup EXIT INT TERM` therefore cleaned
# up on a Ctrl-C and carried on. Use `trap cleanup EXIT` for the cleanup, which
# runs however the script ends, and stop_on_signals for the stopping.
#
# Works under bash 3.2 (stock macOS).

# stop_on_signals
#
# Make INT (Ctrl-C), TERM and HUP (the terminal closing) end this script and the
# background jobs it started with job control on, as run_pool does.
#
# bash runs a trap only once the foreground command has ended. A Ctrl-C reaches
# that command too, so it ends at once; a TERM sent to the script alone waits.
stop_on_signals() {
    local name
    for name in INT TERM HUP; do
        # shellcheck disable=SC2064  # the name is fixed now, not when it fires
        trap "stop_jobs_and_die $name" "$name"
    done
}

# stop_jobs_and_die <signal name>
#
# Interrupt each job's process group, give it STOP_GRACE_SECONDS (default 5) to
# tear down, kill what is left, then die of the signal.
#
# INT rather than TERM for the jobs: pytest answers it by running the fixture
# teardowns, which drop the databases and indices a suite made.
#
# Dying of the signal rather than exiting 130: a bash that was running this
# script stops too only if it sees that.
stop_jobs_and_die() {
    local group groups running polls
    # A second signal must not cut the stopping short.
    trap '' INT TERM HUP
    # Taken once: a job's own shell may be gone while its suite tears down.
    groups=$(jobs -pr) || groups=""
    if [ -n "$groups" ]; then
        # The terminal may be what went away.
        { echo ""; echo "Interrupted: stopping the running jobs..."; } 2>/dev/null || true
    fi
    for group in $groups; do
        kill -INT -- "-$group" 2>/dev/null || true
    done
    polls=$(( ${STOP_GRACE_SECONDS:-5} * 10 ))
    while [ "$polls" -gt 0 ]; do
        running=""
        for group in $groups; do
            if kill -0 -- "-$group" 2>/dev/null; then
                running="yes"
            fi
        done
        [ -n "$running" ] || break
        sleep 0.1
        polls=$(( polls - 1 ))
    done
    for group in $groups; do
        kill -KILL -- "-$group" 2>/dev/null || true
    done
    # The EXIT trap still runs: bash runs it when a signal ends the script.
    trap - "$1"
    kill -s "$1" $$
}
