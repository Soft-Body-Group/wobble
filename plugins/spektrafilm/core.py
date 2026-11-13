"""The spektrafilm wobbin's logic, without the event loop: finding the baker,
recipes, baking, and reading a recipe back off a node. No wobble calls here,
so it tests on its own (test_core.py)."""

import json
import os
import re
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LUTGEN = os.path.join(HERE, "lutgen")
BAKER = os.path.join(LUTGEN, "target", "release", "spektra-lut")

# The recipe rides on the node, in its source, on this line.
RECIPE_RE = re.compile(r"^// spektrafilm: (\{.*\})$", re.M)

AUTO_PRINT = "paired paper"
SCAN = "no print (scan the film)"

# wobble working primaries (app_info's working_space Debug text) -> the
# spektrafilm colourspace that has them. Linear either way: both sides of the
# LUT are scene-linear.
SPACES = {
    "Ap1": "ACEScg",
    "Ap0": "ACES2065-1",
    "Rec709": "ITU-R BT.709",
    "Rec2020": "ITU-R BT.2020",
    "P3D65": "Display P3",
    "P3Dci": "DCI-P3",
}


def data_dir():
    """spektrafilm's data (profiles, spectral LUTs): $SPEKTRAFILM_DATA, else
    the spektrafilm-rs checkout next to this wobble checkout — the same one
    lutgen/Cargo.toml builds against."""
    env = os.environ.get("SPEKTRAFILM_DATA")
    if env:
        return env
    return os.path.normpath(os.path.join(HERE, "..", "..", "..", "spektrafilm-rs", "data"))


def space_of(working_space):
    m = re.search(r"primaries:\s*(\w+)", working_space or "")
    prim = m.group(1) if m else "Rec709"
    if prim not in SPACES:
        raise ValueError(f"spektrafilm has no colourspace with {prim} primaries; "
                         f"use a working space in {', '.join(SPACES)}")
    return SPACES[prim]


def baker_ready():
    return os.path.isfile(BAKER) and os.access(BAKER, os.X_OK)


def build_baker():
    """cargo build the baker. Minutes the first time (it builds spektrafilm-rs)."""
    r = subprocess.run(["cargo", "build", "--release"], cwd=LUTGEN,
                       capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.strip().splitlines()[-6:])
        raise RuntimeError(f"cargo build failed:\n{tail}")


def _run(args):
    if not baker_ready():
        raise RuntimeError("the baker is not built: press 'build the baker'")
    r = subprocess.run([BAKER, *args, "--data", data_dir()], capture_output=True, text=True)
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        raise RuntimeError(f"baker said nothing useful: {r.stderr.strip()[-300:]}")
    if not out.get("ok"):
        raise RuntimeError(out.get("error", "bake failed"))
    return out


def profiles():
    """{'films': [{id, label, print}], 'papers': [{id, label}]}"""
    return _run(["list"])


def default_recipe(film="kodak_portra_400"):
    return {
        "film": film,
        "print": "",          # "" = the film's paired paper
        "scan_film": None,    # None = scan only a stock that has no paper
        "size": 33,
        "space": "ACEScg",
        "params": {},
    }


def set_param(recipe, path, value):
    """Set a dotted spektrafilm RuntimeParams path in the recipe's overrides."""
    node = recipe.setdefault("params", {})
    keys = path.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def get_param(recipe, path, default):
    node = recipe.get("params", {})
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _clean(recipe):
    """What the baker is given: no Nones, and nothing it does not read."""
    r = {k: v for k, v in recipe.items() if v is not None and k != "baker"}
    return r


def bake(recipe):
    """Bake a recipe; returns the Shader node source."""
    with tempfile.TemporaryDirectory(prefix="spektrafilm-") as d:
        rp = os.path.join(d, "recipe.json")
        out = os.path.join(d, "film.wgsl")
        with open(rp, "w") as f:
            json.dump(_clean(recipe), f)
        _run(["bake", "--recipe", rp, "--out", out])
        with open(out) as f:
            return f.read()


def recipe_of(source):
    """The recipe a film node was baked from, or None for any other shader."""
    m = RECIPE_RE.search(source or "")
    if not m:
        return None
    try:
        r = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    r.pop("baker", None)
    return r


def label_of(recipe, films, papers):
    """A node name: 'Portra 400 / Portra Endura', or the stock alone for a scan."""
    def short(pid, table):
        for p in table:
            if p["id"] == pid:
                return re.sub(r"^(Kodak|Fujifilm|Fuji)\s+", "", p["label"] or pid)
        return pid
    film = short(recipe["film"], films)
    if recipe.get("scan_film"):
        return film
    paper = recipe.get("print") or next((f.get("print") for f in films if f["id"] == recipe["film"]), None)
    # A print FILM (2383, 2393) is a film profile, not a paper.
    return f"{film} / {short(paper, papers + films)}" if paper else film
