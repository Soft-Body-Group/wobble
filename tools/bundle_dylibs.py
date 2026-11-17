#!/usr/bin/env python3
"""Make a .app self-contained: copy every non-system dylib the binaries
(Contents/MacOS/wobble, and scrub when it is there) pull
in (FFmpeg from Homebrew, and everything IT pulls in) into
Contents/Frameworks and point the load commands there.

    tools/bundle_dylibs.py dist/Wobble.app

System libraries (/usr/lib, /System) stay as they are. Fails if any
reference outside the bundle survives, so a build that would only run on
a Mac with Homebrew never ships.
"""
import os
import shutil
import subprocess
import sys

SYSTEM = ("/usr/lib/", "/System/")


def deps(path):
    out = subprocess.run(["otool", "-L", path], capture_output=True, text=True, check=True).stdout
    # First line is the file itself; a dylib's next line is its own id.
    return [l.strip().split(" (")[0] for l in out.splitlines()[1:] if l.strip()]


def rpaths(path):
    out = subprocess.run(["otool", "-l", path], capture_output=True, text=True, check=True).stdout.splitlines()
    rp = []
    for i, l in enumerate(out):
        if l.strip() == "cmd LC_RPATH":
            rp.append(out[i + 2].strip().split(" ")[1])
    return rp


def resolve(ref, owner):
    """The file a load command names, or None for a system library."""
    if ref.startswith(SYSTEM):
        return None
    here = os.path.dirname(os.path.realpath(owner))
    if ref.startswith("@loader_path/"):
        return os.path.realpath(os.path.join(here, ref[len("@loader_path/"):]))
    if ref.startswith("@rpath/"):
        for rp in rpaths(owner):
            rp = rp.replace("@loader_path", here)
            cand = os.path.join(rp, ref[len("@rpath/"):])
            if os.path.exists(cand):
                return os.path.realpath(cand)
        # $WOBBLE_DYLIB_PATH: where a library built with @rpath ids lives
        # (the release's own FFmpeg), for a binary that carries no rpath.
        dirs = [here] + [d for d in os.environ.get("WOBBLE_DYLIB_PATH", "").split(":") if d]
        for d in dirs:
            cand = os.path.join(d, ref[len("@rpath/"):])
            if os.path.exists(cand):
                return os.path.realpath(cand)
        return None
    if ref.startswith("@"):
        return None
    return os.path.realpath(ref)


def main(app):
    exes = [os.path.join(app, "Contents/MacOS", n) for n in ("wobble", "scrub")]
    exes = [e for e in exes if os.path.exists(e)]
    fw = os.path.join(app, "Contents/Frameworks")
    os.makedirs(fw, exist_ok=True)
    copied = {}  # real source path -> bundled name
    todo = list(exes)
    while todo:
        f = todo.pop()
        # A dylib's first entry is its own id, not a dependency.
        for ref in deps(f)[0 if f in exes else 1:]:
            src = resolve(ref, f)
            if src is None or src.startswith(os.path.realpath(fw) + "/"):
                continue
            if src not in copied:
                name = os.path.basename(src)
                dst = os.path.join(fw, name)
                shutil.copy2(src, dst)
                os.chmod(dst, 0o755)
                subprocess.run(["install_name_tool", "-id", f"@rpath/{name}", dst],
                               check=True, capture_output=True)
                copied[src] = name
                todo.append(dst)
            subprocess.run(["install_name_tool", "-change", ref, f"@rpath/{copied[src]}", f],
                           check=True, capture_output=True)
    for f in exes + [os.path.join(fw, n) for n in copied.values()]:
        rp = "@executable_path/../Frameworks" if f in exes else "@loader_path"
        # A dylib's @rpath deps also resolve through the executable's rpath
        # (dyld walks the whole load chain), so its own is a nicety, and a
        # library linked without header padding has no room for one.
        if rp not in rpaths(f):
            subprocess.run(["install_name_tool", "-add_rpath", rp, f], check=f in exes, capture_output=True)
    bad = []
    for f in exes + [os.path.join(fw, n) for n in copied.values()]:
        for ref in deps(f)[0 if f in exes else 1:]:
            if not ref.startswith(SYSTEM) and not ref.startswith("@rpath/"):
                bad.append(f"{os.path.basename(f)} -> {ref}")
    if bad:
        sys.exit("bundle_dylibs: references outside the bundle remain:\n  " + "\n  ".join(bad))
    print(f"bundled {len(copied)} dylibs into {fw}")


if __name__ == "__main__":
    main(sys.argv[1])
