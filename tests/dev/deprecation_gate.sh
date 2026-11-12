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
# With '--stage-logs' the battery's stage logs are read too: what a suite does not redirect lands there (a program's
# stderr, a data generator's, a python suite's own), which the temporary files never show. A line there belongs to the
# step named by the last '==== ' line above it (tests/ci_suite.sh's step(), one per suite), else to the stage itself
# (a stage that runs one suite); only the suites whose every case is an old spelling may leave one (the second
# allow-list below). Not read: the build stages' logs (the compilers' output, no suite) and the gate's own.
#
# Usage: tests/dev/deprecation_gate.sh [--stage-logs <folder>] <suites' temp folder> [<marker file>]
#   --stage-logs <folder>   also read the '<stage>.log' files in this folder (tests/dev/battery.sh's
#                           <scratch>/battery), the same marker applying
#   <suites' temp folder>   the DTFE_TEST_TMP the suites ran with: one folder per suite (a path's first component
#                           names its suite), a few loose files in the root
#   <marker file>           read only the files newer than it: tests/dev/battery.sh touches one before its first
#                           suite, since a reused scratch folder still holds the files of suites this battery did not
#                           run; with '--only deprecation_gate' it runs no suite and passes none (the whole folder)
# Prints one 'UNEXPECTED' line per deprecation line outside the allow-lists and one 'UNREADABLE' line per listed file
# it cannot read, then what the allow-lists covered.
# Exit 0 when every deprecation line is allow-listed and every listed file was read, 1 otherwise, 2 on a usage error.
set -uo pipefail
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; exit 2; }
folder=""; mark=""; stage_dir=""; nargs=0
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage ;;
        --stage-logs) [ $# -ge 2 ] || usage; stage_dir="$2"; shift 2 ;;
        -*) echo "deprecation gate: unknown option '$1'"; usage ;;
        *) nargs=$((nargs + 1))
           case "$nargs" in 1) folder="$1" ;; 2) mark="$1" ;; *) usage ;; esac
           shift ;;
    esac
done
[ "$nargs" -ge 1 ] || usage
[ -d "$folder" ] || { echo "deprecation gate: no folder '$folder'"; exit 2; }
root="$(cd "$folder" && pwd -P)"   # the physical path: find lists nothing under a symlinked start folder
sroot=""
if [ -n "$stage_dir" ]; then
    [ -d "$stage_dir" ] || { echo "deprecation gate: no stage-log folder '$stage_dir'"; exit 2; }
    sroot="$(cd "$stage_dir" && pwd -P)"
fi
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
        run_scripts/*)
            echo "the old-name cases of the drivers and the launcher" ;;
        *) return 1 ;;
    esac
}

# stage_allowed <stage> <step>: prints the reason and returns 0 for a stage-log line of a suite whose every case is an
# old spelling (by its whole step title in tests/ci_suite.sh: a new step that only starts with the same words is not
# allowed, and a retitled step fails here until this list follows); the suites that test a few old spellings among
# other things keep those runs' output in their files (above), so a line of theirs in a stage log is unexpected
stage_allowed() {
    case "$2" in
        "old and new flag spellings: one run, one settings key")
            echo "flag_spelling_check: the suite tests every old spelling of the options and variables" ;;
        "python CLI spellings: old names as warned aliases, no abbreviations")
            echo "py_cli_alias_test: the python CLIs' old names as warned aliases" ;;
        "batch drivers")
            echo "run_scripts_check: the old-name cases of the drivers and the launcher" ;;
        *) return 1 ;;
    esac
}

# step_of <stage log> <line number>: the step that line belongs to, the text of the last '==== ' line above it
# (empty in a stage that runs one suite itself)
step_of() { awk -v n="$2" 'NR > n { exit } /^==== / { s = substr($0, 6) } END { print s }' "$1"; }

# suite_files, stage_files: the files the gate reads, NUL-separated. The suites' files by name (the names keep a
# battery's ~10 GB of grids unread); of the battery's stage logs ('--stage-logs') every '<stage>.log' but the builds'
# and this gate's own (which the battery opens before the gate runs)
suite_files() {
    find "$root" -type f \( -name '*.log' -o -name '*.err' -o -name '*.out' -o -name '*.txt' -o -name '*runlog*' \) \
         ${newer[@]+"${newer[@]}"} -print0
}
stage_files() {
    find "$sroot" -maxdepth 1 -type f -name '*.log' ! -name 'build*.log' ! -name 'deprecation_gate.log' \
         ${newer[@]+"${newer[@]}"} -print0
}
# count_nul: the number of NUL-separated names on stdin
count_nul() { tr -cd '\0' | wc -c | tr -d ' '; }

# a listed file that cannot be read drops out of grep's list below with its error discarded, and its lines would
# count as clean: name it and fail
nunread=0
while IFS= read -r -d '' f; do
    [ -r "$f" ] || { nunread=$((nunread + 1)); echo "UNREADABLE (not scanned): ${f}"; }
done < <(suite_files; [ -z "$sroot" ] || stage_files)

# the files with a deprecation line: -a reads a log with a NUL byte as text where -I would skip it unread
# (serve_split/srv.log, window_check/serve.log); '/dev/null' never matches, and xargs's exit status for a batch
# without a match is not an error here
nfiles="$(suite_files | count_nul)"
files="$(suite_files | xargs -0 grep -laE "$PAT" /dev/null 2>/dev/null || true)"

nok=0; nbad=0; seen=""; sseen=""; tab=$'\t'
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

# the battery's stage logs ('--stage-logs')
nstage=0
if [ -n "$sroot" ]; then
    nstage="$(stage_files | count_nul)"
    sfiles="$(stage_files | xargs -0 grep -laE "$PAT" /dev/null 2>/dev/null || true)"
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        stage="$(basename "$f" .log)"
        while IFS= read -r hit; do
            ln="${hit%%:*}"
            step="$(step_of "$f" "$ln")"
            if why="$(stage_allowed "$stage" "$step")"; then
                nok=$((nok + 1))
                sseen="${sseen}${stage}${step:+ (${step})}${tab}${why}"$'\n'
            else
                nbad=$((nbad + 1))
                echo "UNEXPECTED deprecation line: stage=${stage}${step:+ step='${step}'} file=${f##*/}:${ln}: ${hit#*:}"
            fi
        done < <(grep -naE "$PAT" "$f")
    done <<< "$sfiles"
fi

if [ -n "$seen" ]; then
    echo "allow-listed (suite: lines in files -- why):"
    # one row per suite and reason: the files of a suite summed, without bash 4's associative arrays
    printf '%s' "$seen" | awk -F'\t' '{ k = $1 "\t" $3; n[k] += $2; c[k]++ }
        END { for (k in n) { split(k, p, "\t"); printf "  %s: %d line(s) in %d file(s) -- %s\n", p[1], n[k], c[k], p[2] } }' | sort
fi
if [ -n "$sseen" ]; then
    echo "allow-listed in the stage logs (stage (step): lines -- why):"
    printf '%s' "$sseen" | awk -F'\t' '{ n[$0]++ } END { for (k in n) { split(k, p, "\t"); printf "  %s: %d line(s) -- %s\n", p[1], n[k], p[2] } }' | sort
fi
# the counts of files read are in the summary, so a run that read nothing (a marker newer than every file) does not
# look like a clean scan
echo "deprecation gate over ${root}${mark:+ (files newer than ${mark})}: ${nfiles} file(s) read${sroot:+, ${nstage} stage log(s) of ${sroot}}, ${nok} allow-listed line(s), ${nbad} unexpected, ${nunread} unreadable"
[ "$nfiles" -gt 0 ] || echo "NOTE: no file was read (nothing newer than the marker, or an empty folder)"
[ -z "$sroot" ] || [ "$nstage" -gt 0 ] || echo "NOTE: no stage log was read (nothing newer than the marker, or no '<stage>.log' in ${sroot})"
if [ "$nbad" -gt 0 ]; then
    echo "RESULT: FAIL (${nbad} deprecation line(s) outside the allow-lists: spell the suite's option the new way, or allow-list the file or the step in ${BASH_SOURCE[0]##*/} with a reason)"
    exit 1
fi
if [ "$nunread" -gt 0 ]; then
    echo "RESULT: FAIL (${nunread} listed file(s) could not be read, see the UNREADABLE lines: their deprecation lines were not checked)"
    exit 1
fi
echo "RESULT: PASS"
exit 0
