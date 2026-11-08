#!/usr/bin/env bash
# The battery's 'deprecation_gate' stage: the suites run the binaries, the drivers and the launcher under the
# canonical option and variable names, so a 'warning: <old> is deprecated, use <new>' line (the wording of the
# binaries, scripts/config.sh and dtfelib, once per old name) in a suite's temporary files means a suite still spells
# something the old way. Only the suites that run the old spellings on purpose may leave one: the allow-list below
# names their files, each with its reason. The line is matched at the start of a line, so a suite that echoes the
# wording inside a message of its own does not count. Only the files a program's output goes to are read, by name
# ('*.log', '*.err', '*.out', '*.txt': where the suites capture stdout and stderr; '*runlog*': the binaries' own),
# each as text, a NUL byte in it (the --serve logs hold some) included.
# The folder is read as the suites left it. The battery runs several suites more than once into one DTFE_TEST_TMP
# (ci_core and then ci_double; the *_d and *_np126 stages after the stages that ran the same suites), and a rerun
# clears its suite's folder or rewrites its logs by name: the files read are each suite's last run of the battery,
# with any file of an earlier run that the last one did not write again.
# Not read: the battery's own stage logs. Five python suites (ps_3d_test, ps_standard_cross_check,
# ps_convergence_test, ps_nonperiodic_test, ps_regression_test) run the binaries without capturing their stderr, so
# a deprecation line of theirs lands in the stage log, not here.
#
# Usage: tests/dev/deprecation_gate.sh <suites' temp folder> [<marker file>]
#   <suites' temp folder>   the DTFE_TEST_TMP the suites ran with: one folder per suite (a path's first component
#                           names its suite), a few loose files in the root
#   <marker file>           read only the files newer than it: tests/dev/battery.sh touches one before its first
#                           suite, since a reused scratch folder still holds the files of suites this battery did not
#                           run; with '--only deprecation_gate' it runs no suite and passes none (the whole folder)
# Prints one 'UNEXPECTED' line per deprecation line outside the allow-list, then what the allow-list covered.
# Exit 0 when every deprecation line is allow-listed, 1 when one is not, 2 on a usage error.
set -uo pipefail
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; exit 2; }
{ [ $# -ge 1 ] && [ $# -le 2 ]; } || usage
case "$1" in -h|--help) usage ;; esac
[ -d "$1" ] || { echo "deprecation gate: no folder '$1'"; exit 2; }
root="$(cd "$1" && pwd -P)"   # the physical path: find lists nothing under a symlinked start folder
mark="${2:-}"
newer=()
if [ -n "$mark" ]; then
    [ -e "$mark" ] || { echo "deprecation gate: no marker file '$mark'"; exit 2; }
    newer=(-newer "$mark")
fi
PAT='^warning: .* is deprecated'

# allowed <path relative to the folder>: prints the reason and returns 0 for a file of a suite that runs an old
# spelling on purpose
allowed() {
    case "$1" in
        flag_spelling_check/*)
            echo "the suite tests every old spelling of the options and variables" ;;
        dtfe_resume_check/k1.log|dtfe_resume_check/k4a.log)
            echo "D2, D4: the killed runs spell --lambda_th, so the resume under --lambda-th tests one settings key" ;;
        ps_resume_check/x1.log)
            echo "C15: a run killed under the old spellings resumes under the canonical ones" ;;
        ps_resume_check/g1.log)
            echo "C8: a killed --ps-gpu run resumes under --gpu" ;;
        ps_runtime_check/bad.log)
            echo "R7: the deprecated merge pair (--ordered-merge --completion-order-merge) is refused" ;;
        ps_fixed_merge_check/bad1.log)
            echo "X5: the deprecated --completion-order-merge beside --fixed-point-merge is refused" ;;
        dtfe_sampling_check/*|dtfe_mc_average_check/*)
            echo "the deprecated total --samples 20 is kept: no --subsamples k (k^3 points) gives 20" ;;
        run_scripts/*)
            echo "the old-name cases of the drivers and the launcher" ;;
        *) return 1 ;;
    esac
}

# the files with a deprecation line: the names keep a battery's ~10 GB of grids unread, and -a reads a log with a
# NUL byte as text where -I would skip it unread (serve_split/srv.log, window_check/serve.log);
# '/dev/null' never matches, and xargs's 123 (a batch without a match) is not an error here
nfiles="$(find "$root" -type f \( -name '*.log' -o -name '*.err' -o -name '*.out' -o -name '*.txt' -o -name '*runlog*' \) \
             ${newer[@]+"${newer[@]}"} | wc -l | tr -d ' ')"
files="$(find "$root" -type f \( -name '*.log' -o -name '*.err' -o -name '*.out' -o -name '*.txt' -o -name '*runlog*' \) \
             ${newer[@]+"${newer[@]}"} -print0 | xargs -0 grep -laE "$PAT" /dev/null 2>/dev/null || true)"

nok=0; nbad=0; seen=""; tab=$'\t'
while IFS= read -r f; do
    [ -n "$f" ] || continue
    rel="${f#"$root"/}"
    case "$rel" in */*) suite="${rel%%/*}" ;; *) suite="(root)" ;; esac
    if why="$(allowed "$rel")"; then
        k="$(grep -acE "$PAT" "$f")"
        nok=$((nok + k))
        seen="${seen}${suite}${tab}${k}${tab}${why}"$'\n'
    else
        while IFS= read -r hit; do
            nbad=$((nbad + 1))
            echo "UNEXPECTED deprecation line: suite=${suite} file=${rel}:${hit%%:*}: ${hit#*:}"
        done < <(grep -naE "$PAT" "$f")
    fi
done <<< "$files"

if [ -n "$seen" ]; then
    echo "allow-listed (suite: lines in files -- why):"
    # one row per suite and reason: the files of a suite summed, without bash 4's associative arrays
    printf '%s' "$seen" | awk -F'\t' '{ k = $1 "\t" $3; n[k] += $2; c[k]++ }
        END { for (k in n) { split(k, p, "\t"); printf "  %s: %d line(s) in %d file(s) -- %s\n", p[1], n[k], c[k], p[2] } }' | sort
fi
# the count of files read is in the summary, so a run that read nothing (a marker newer than every file) does not
# look like a clean scan
echo "deprecation gate over ${root}${mark:+ (files newer than ${mark})}: ${nfiles} file(s) read, ${nok} line(s) in allow-listed files, ${nbad} unexpected"
[ "$nfiles" -gt 0 ] || echo "NOTE: no file was read (nothing newer than the marker, or an empty folder)"
if [ "$nbad" -gt 0 ]; then
    echo "RESULT: FAIL (${nbad} deprecation line(s) outside the allow-list: spell the suite's option the new way, or allow-list the file in ${BASH_SOURCE[0]##*/} with a reason)"
    exit 1
fi
echo "RESULT: PASS"
exit 0
