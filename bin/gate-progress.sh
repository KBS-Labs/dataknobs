#!/usr/bin/env bash
# How the quality gate spends and reports its time: the pool its unit suites
# run through, and the clock on the lines it prints.
#
# Sourced by bin/run-quality-checks.sh, bin/docs-checks.sh and bin/validate.sh,
# and by tests/test_gate_progress.py -- a file of its own so the scheduling is
# code a test can run. Works under `set -u`, which validate.sh sets, and under
# bash 3.2 (stock macOS), which has neither `wait -n` nor EPOCHREALTIME.

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

# longest_suite_first <root> <output dir> <package>...
#
# The packages, one per line, by how long each one's unit suite last took
# (unit-test-seconds-<package>.txt in the output directory): a long suite
# started last runs alone at the end.
#
# One with no such record goes ahead of those that have one, since it may be the
# longest, and by how much unit test code it has. That is only a stand-in: the
# package with the most code is not the one that takes longest.
longest_suite_first() {
    local root="$1" out="$2" pkg seconds size
    shift 2
    for pkg in "$@"; do
        seconds=""
        if [ -f "$out/unit-test-seconds-$pkg.txt" ]; then
            read -r seconds < "$out/unit-test-seconds-$pkg.txt" || true
        fi
        case "$seconds" in
            '' | *[!0-9]*) ;;
            *)
                printf '1 %d %s\n' "$((10#$seconds))" "$pkg"
                continue
                ;;
        esac
        size=0
        if [ -d "$root/packages/$pkg/tests" ]; then
            size=$(find "$root/packages/$pkg/tests" -name '*.py' \
                -not -path '*/integration/*' -exec cat {} + | wc -c)
        fi
        printf '0 %d %s\n' "$size" "$pkg"
    done | sort -k1,1n -k2,2rn -k3 | cut -d' ' -f3
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
