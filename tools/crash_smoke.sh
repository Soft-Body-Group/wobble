#!/bin/sh
# Crash-capture smoke test: crashes a wobble binary on purpose and checks the
# reports land in an isolated outbox (never ~/.wobble, never uploaded).
#
#   tools/crash_smoke.sh target/release/wobble        # macOS / Linux
#   sh tools/crash_smoke.sh target/release/wobble.exe # Windows (Git Bash)
#
# CI runs it on all three (ci.yml) and on the Windows alpha build
# (alpha.yml). Every crash happens before a window exists, so it
# needs no display or GPU. What it proves:
#   everywhere  a panic writes a report; a native abort (no hook can see it)
#               is reported by the NEXT launch as an unclean exit.
#   Windows     an access violation goes through the exception filter: a
#               report naming the exception + module, a minidump, and (with
#               wobble.pdb beside the exe) a symbolised stack.
set -u
BIN=${1:?usage: crash_smoke.sh <wobble binary>}
DIR="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/wobble-crash-smoke"
rm -rf "$DIR"
export WOBBLE_REPORT_DIR="$DIR"
unset WOBBLE_REPORT_URL WOBBLE_REPORT_SELFTEST
OB="$DIR/crash"
fail() {
  echo "CRASH SMOKE FAIL: $*"
  ls -la "$OB" 2>/dev/null
  exit 1
}
crash() { # crash <mode>: run, expect a non-zero exit
  WOBBLE_CRASH_TEST=$1 "$BIN" >"$DIR.out" 2>&1 && fail "WOBBLE_CRASH_TEST=$1 exited 0"
  return 0
}
case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) WIN=1 ;; *) WIN=0 ;; esac
# On a CI runner, keep Windows Error Reporting from parking a dialog over the
# crashed process (it would hang the step). Never touched on a real machine.
if [ "$WIN" = 1 ] && [ -n "${CI:-}" ]; then
  MSYS_NO_PATHCONV=1 reg add "HKCU\\Software\\Microsoft\\Windows\\Windows Error Reporting" \
    /v DontShowUI /t REG_DWORD /d 1 /f >/dev/null 2>&1 || true
fi

# 1. panic
crash 1
grep -l '"title": "panic: deliberate test crash' "$OB"/crash-*.json >/dev/null 2>&1 \
  || fail "no panic report"
echo "ok   panic -> report"

# 2. Windows: access violation through the exception filter
if [ "$WIN" = 1 ]; then
  crash segv
  R=$(grep -l '"title": "native crash: 0xc0000005' "$OB"/crash-*.json 2>/dev/null | head -1)
  [ -n "$R" ] || fail "no native crash report for the access violation"
  echo "ok   access violation -> $(basename "$R")"
  D=$(ls "$OB"/*.dmp 2>/dev/null | head -1)
  [ -n "$D" ] && [ -s "$D" ] || fail "no minidump"
  echo "ok   minidump $(basename "$D") ($(wc -c <"$D") bytes)"
  if grep -q "maybe_crash_for_test" "$R"; then
    echo "ok   stack symbolised from wobble.pdb"
  else
    # Not fatal: the report + dump are what matter, but say so loudly.
    echo "WARN stack not symbolised (is wobble.pdb beside the exe?)"
    [ -n "${GITHUB_ACTIONS:-}" ] && echo "::warning::crash stack not symbolised"
  fi
  sed -n '1,60p' "$R"
fi

# 3. a native abort: no report from the dying process...
crash abort
# ...so the next launch must report it (the OS's record of it can take a
# few seconds to appear; the adopting run looks once, uploads look again).
sleep 8
crash 1
U=$(ls "$OB"/unclean-*.json 2>/dev/null | head -1)
[ -n "$U" ] || fail "the aborted session was not reported by the next launch"
echo "ok   abort -> next launch reported $(basename "$U"):"
grep -E '"(kind|title)"' "$U"

echo "CRASH SMOKE PASS"
