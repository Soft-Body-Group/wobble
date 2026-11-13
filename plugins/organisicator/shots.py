"""What a wobble comp needs from an organisicator shot, with no wobble in it.

``core.py`` is organisicator's own manifest reader, copied verbatim from its
Blender add-on (``organisicator/blender_addon/organisicator/core.py``) so the
two integrations can never disagree about where a shot's folders are. This
file adds the wobble side: where comps live, what they are called, where a
Write renders to, and which image sequences are plates.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import core
from core import Instance, Project

COMP_EXT = ".wbl"
SEQ_EXTS = {"exr", "dpx", "png", "jpg", "jpeg", "tif", "tiff", "mov", "mp4"}
MOVIE_EXTS = {"mov", "mp4"}

PREFS = os.path.join(os.path.expanduser("~"), ".wobble", "organisicator.json")
# Where organisicator puts managed projects by default; the panel's
# field overrides it and the choice is remembered.
DEFAULT_ROOTS = [
    "~/Library/CloudStorage/Dropbox/managed",
    "~/Dropbox/managed",
    "D:/Dropbox/managed",
]


# ---- prefs -----------------------------------------------------------------


def load_prefs() -> dict:
    try:
        with open(PREFS, "r", encoding="utf-8") as fh:
            d = json.load(fh)
            return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_prefs(d: dict) -> None:
    os.makedirs(os.path.dirname(PREFS), exist_ok=True)
    tmp = PREFS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)
    os.replace(tmp, PREFS)


def default_root() -> str:
    for r in DEFAULT_ROOTS:
        p = os.path.expanduser(r)
        if os.path.isdir(p):
            return p
    return ""


# ---- labels ----------------------------------------------------------------


def label_of(project: Project, inst: Optional[Instance]) -> str:
    """``dev / spb_010`` for a shot, ``dev`` for a sequence, ``(project)``."""
    if inst is None:
        return "(project)"
    return " / ".join(i.value for i in project.ancestors(inst) + [inst])


def shot_labels(project: Project) -> list[tuple[str, Optional[Instance]]]:
    return [(label_of(project, None), None)] + [(label_of(project, i), i) for i in project.tree()]


# ---- folders ---------------------------------------------------------------


def comp_dir(project: Project, inst: Optional[Instance]) -> Optional[str]:
    """Where this instance's comps live: a folder named wobble or comp, else
    one whose file types include .wbl, else its software folder (or that
    folder's ``wobble`` child when one exists) — the Blender add-on's rule
    with wobble's names."""

    def score(f: dict, path: str) -> int:
        name = core._segs(f.get("path", ""))[-1].lower()
        exts = (f.get("exts") or "").lower().split()
        kind = (f.get("type") or "").lower()
        if name == "wobble":
            return 5
        if "wbl" in exts:
            return 4
        if name in ("comp", "comps") or kind == "comp":
            return 3
        if kind == "software" or name in ("software", "scripts", "work"):
            return 2
        return 0

    found = project._pick(inst, score)
    if found and os.path.basename(found).lower() != "wobble":
        for child in ("wobble", "Wobble"):
            p = os.path.join(found, child)
            if os.path.isdir(p):
                return p
    return found


def plates_dir(project: Project, inst: Optional[Instance]) -> Optional[str]:
    def score(f: dict, path: str) -> int:
        name = core._segs(f.get("path", ""))[-1].lower()
        kind = (f.get("type") or "").lower()
        if kind in ("plate", "plates"):
            return 4
        if name in ("plates", "plate"):
            return 3
        if kind == "source":
            return 2
        return 0

    return project._pick(inst, score)


def comp_files(directory: Optional[str]) -> list[str]:
    """``.wbl`` files in a folder, newest first."""
    if not directory or not os.path.isdir(directory):
        return []
    out = []
    for e in os.scandir(directory):
        if e.is_file() and e.name.lower().endswith(COMP_EXT) and not e.name.startswith("."):
            try:
                out.append((e.stat().st_mtime, e.path))
            except OSError:
                pass
    out.sort(reverse=True)
    return [p for _, p in out]


# ---- names -----------------------------------------------------------------


def new_comp_path(project: Project, inst: Optional[Instance], task: str) -> Optional[str]:
    """``<comp dir>/<shot>_<task>_v001.wbl``, or the next version past any
    comp of that name already there."""
    d = comp_dir(project, inst)
    if not d:
        return None
    t = core.tokens_for(project, inst, task=task)
    stem = core.fill("{name}_{task}_v001" if task else "{name}_v001", t).replace("/", "_")
    path = os.path.join(d, stem + COMP_EXT)
    while os.path.exists(path):
        stem = core.next_version(stem)
        path = os.path.join(d, stem + COMP_EXT)
    return path


def version_up(path: str) -> str:
    """``…_v004.wbl`` -> ``…_v005.wbl``, skipping versions already on disk."""
    d, name = os.path.split(path)
    stem = os.path.splitext(name)[0]
    while True:
        stem = core.next_version(stem)
        nxt = os.path.join(d, stem + COMP_EXT)
        if not os.path.exists(nxt):
            return nxt


def write_targets(project: Project, inst: Optional[Instance], comp_stem: str, writes: list[str]) -> dict[str, str]:
    """Each Write's output, by node name: ``<renders>/<comp>/<comp>.####.exr``
    for a lone Write, ``<renders>/<comp>_<write>/…`` for several (so two
    Writes never land in one folder). Empty when the project has no render
    folder."""
    rdir = project.render_dir(inst)
    if not rdir or not writes:
        return {}
    out = {}
    for w in writes:
        stem = comp_stem if len(writes) == 1 else f"{comp_stem}_{re.sub(r'[^A-Za-z0-9_-]+', '_', w)}"
        out[w] = os.path.join(rdir, stem, f"{stem}.####.exr")
    return out


# ---- plates ----------------------------------------------------------------

_FRAME = re.compile(r"^(.*?)(\d+)\.([A-Za-z0-9]+)$")


def sequences(directory: Optional[str], depth: int = 3) -> list[str]:
    """Every image sequence and movie under ``directory``, as the paths a
    Read takes: ``dir/name.####.exr`` for frames, the file for a movie.
    Sorted, hidden folders skipped."""
    out: list[str] = []
    if not directory or not os.path.isdir(directory):
        return out

    def walk(d: str, left: int) -> None:
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name.lower())
        except OSError:
            return
        groups: dict[tuple[str, int, str], int] = {}
        for e in entries:
            if e.name.startswith("."):
                continue
            if e.is_dir(follow_symlinks=False):
                if left > 0:
                    walk(e.path, left - 1)
                continue
            ext = e.name.rsplit(".", 1)[-1].lower() if "." in e.name else ""
            if ext not in SEQ_EXTS:
                continue
            if ext in MOVIE_EXTS:
                out.append(e.path)
                continue
            m = _FRAME.match(e.name)
            if m:
                key = (m.group(1), len(m.group(2)), m.group(3))
                groups[key] = groups.get(key, 0) + 1
            else:
                out.append(e.path)
        for (head, pad, ext) in groups:
            out.append(os.path.join(d, f"{head}{'#' * pad}.{ext}"))

    walk(directory, depth)
    return sorted(out)



# ---- organisicator's API (asset ids) ----------------------------------------

API_FILE = os.path.join(os.path.expanduser("~"), ".organisicator", "api.json")


def api_get(path: str) -> dict:
    """GET from organisicator's local API (it must be running)."""
    import urllib.request

    try:
        with open(os.environ.get("ORGANISICATOR_API_FILE", API_FILE), "r", encoding="utf-8") as fh:
            d = json.load(fh)
        req = urllib.request.Request(
            f"http://127.0.0.1:{int(d['port'])}{path}",
            headers={"x-organisicator-token": d["token"]},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode("utf-8"))
    except (OSError, ValueError, KeyError) as e:
        raise RuntimeError("organisicator isn't running — start it to add plates by asset") from e


def asset_ids(project: Project, inst: Optional[Instance], paths: list[str], projects: list, listing: dict) -> dict[str, int]:
    """Each Read path in ``paths`` → organisicator's asset id, from its
    ``/v1/projects`` and ``/v1/projects/{id}/assets`` answers. Paths the
    index does not hold yet are left out."""
    root = os.path.realpath(project.root)
    if not any(os.path.realpath(p.get("root", "")) == root for p in projects):
        return {}
    base = project.path_of(inst)
    by_rel = {a["rel"]: a["id"] for a in listing.get("assets", [])}
    out = {}
    for p in paths:
        rel = os.path.relpath(p, base).replace(os.sep, "/")
        if rel in by_rel:
            out[p] = by_rel[rel]
    return out


def fetch_asset_ids(project: Project, inst: Optional[Instance], paths: list[str]) -> dict[str, int]:
    projects = api_get("/v1/projects")
    root = os.path.realpath(project.root)
    pid = next((p["id"] for p in projects if os.path.realpath(p.get("root", "")) == root), None)
    if pid is None:
        raise RuntimeError(f"organisicator doesn't know {project.name}")
    q = f"?instance={inst.id}" if inst is not None else ""
    return asset_ids(project, inst, paths, projects, api_get(f"/v1/projects/{pid}/assets{q}"))
