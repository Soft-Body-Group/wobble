#!/usr/bin/env python3
"""Blank home directories out of release binaries.

--remap-path-prefix catches almost every path rustc writes, but a few panic
locations inlined across crates (and rust-src's std paths) keep the real
one, FFmpeg's libraries record their configure line, and a committed blob
(geoplayer.wasm) carries the paths of whichever machine built IT. So the
user name in ANY /Users/<name>/, /home/<name>/ or X:\\Users\\<name>\\ is
overwritten IN PLACE with a filler of the same length, so no offset or
length in the file moves:

    /Users/name/.cargo/...  ->  /Users/____/.cargo/...

    python3 tools/scrub_paths.py FILE_OR_DIR...

Run it before signing (it changes the bytes a signature covers). Prints how
many it blanked; exits 1 if a home path is still there afterwards.
"""
import os
import re
import sys


HOME = re.compile(rb"(/Users/|/home/|[A-Za-z]:\\Users\\|[A-Za-z]:/Users/)"
                  rb"(?!Shared[/\\])([A-Za-z0-9._-]+)(?=[/\\])", re.IGNORECASE)


def blank(m):
    name = m.group(2)
    if set(name) == {ord("_")}:
        return m.group(0)
    return m.group(1) + b"_" * len(name)


def scrub(path):
    data = open(path, "rb").read()
    new, n = HOME.subn(blank, data)
    n -= sum(1 for m in HOME.finditer(data) if set(m.group(2)) == {ord("_")})
    if new != data:
        st = os.stat(path)
        with open(path, "wb") as f:
            f.write(new)
        os.chmod(path, st.st_mode)
    left = sum(1 for m in HOME.finditer(new) if set(m.group(2)) != {ord("_")})
    return n, left


def main():
    total = left = 0
    for arg in sys.argv[1:]:
        files = [arg] if os.path.isfile(arg) else [
            os.path.join(d, f) for d, _, fs in os.walk(arg) for f in fs]
        for f in files:
            if os.path.islink(f):
                continue
            n, l = scrub(f)
            total, left = total + n, left + l
            if n:
                print(f"scrub_paths: {n} in {f}")
    print(f"scrub_paths: blanked {total}")
    sys.exit(1 if left else 0)


if __name__ == "__main__":
    main()
