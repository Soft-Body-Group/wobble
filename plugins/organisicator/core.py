"""Managed-project logic with no Blender dependency (tested with plain Python).

organisicator publishes each project's structure to
``<project>/.organisicator/project.json`` (see ``src-tauri/src/manifest.rs``).
Every path in it is relative to the project root, so the same file works on
macOS, Windows and Linux wherever Dropbox (or a NAS) mounts the managed root.
This module finds those manifests under the managed root, answers "which
shot is this file in / where do its renders go", and maps absolute paths
between the per-OS roots.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

MANIFEST_DIR = ".organisicator"
MANIFEST_FILE = "project.json"
SUPPORTED_FORMAT = 1

OS_KEYS = ("mac", "windows", "linux")

_VAR_SEG = re.compile(r"^\{[A-Za-z0-9_]+\}$")


def current_os() -> str:
    if sys.platform == "darwin":
        return "mac"
    if sys.platform.startswith("win"):
        return "windows"
    return "linux"


def _segs(rel: str) -> list[str]:
    return [s for s in rel.replace("\\", "/").split("/") if s]


def is_var_segment(seg: str) -> bool:
    return bool(_VAR_SEG.match(seg))


# ---- the project model -----------------------------------------------------


@dataclass
class Instance:
    """A concrete folder behind a repeating level (a sequence, a shot…)."""

    id: int
    parent: Optional[int]
    decl: str
    value: str
    code: str
    rel: str
    depth: int = 0

    @property
    def var(self) -> str:
        last = _segs(self.decl)[-1]
        return last[1:-1]


@dataclass
class Project:
    name: str
    code: str
    root: str  # absolute, native separators
    anatomy: Optional[str] = None
    folders: list[dict] = field(default_factory=list)
    levels: list[dict] = field(default_factory=list)
    instances: list[Instance] = field(default_factory=list)

    # -- structure --

    def instance(self, iid: Optional[int]) -> Optional[Instance]:
        for i in self.instances:
            if i.id == iid:
                return i
        return None

    def children(self, parent: Optional[int]) -> list[Instance]:
        return [i for i in self.instances if i.parent == parent]

    def ancestors(self, inst: Instance) -> list[Instance]:
        """Enclosing instances, outermost first (``inst`` excluded)."""
        out = []
        p = self.instance(inst.parent)
        guard = 0
        while p is not None and guard < 32:
            out.append(p)
            p = self.instance(p.parent)
            guard += 1
        return list(reversed(out))

    def tree(self) -> list[Instance]:
        """Instances depth-first, siblings sorted by name, with ``depth`` set."""
        out: list[Instance] = []

        def walk(parent: Optional[int], depth: int) -> None:
            for c in sorted(self.children(parent), key=lambda i: i.value.lower()):
                c.depth = depth
                out.append(c)
                if depth < 16:
                    walk(c.id, depth + 1)

        walk(None, 0)
        return out

    def abs(self, rel: str) -> str:
        return os.path.join(self.root, *_segs(rel))

    def path_of(self, inst: Optional[Instance]) -> str:
        return self.root if inst is None else self.abs(inst.rel)

    def folders_of(self, inst: Optional[Instance]) -> list[tuple[dict, str]]:
        """The anatomy folders that sit inside ``inst`` (not inside a deeper
        repeating level), as (folder definition, absolute path). ``None``
        means the project's own static folders."""
        out = []
        for f in self.folders:
            s = _segs(f.get("path", ""))
            ks = [k for k, seg in enumerate(s) if is_var_segment(seg)]
            if inst is None:
                if ks:
                    continue
                out.append((f, self.abs("/".join(s))))
                continue
            if not ks:
                continue
            k = ks[-1]
            if "/".join(s[: k + 1]) != inst.decl or k + 1 >= len(s):
                continue
            out.append((f, self.abs(inst.rel + "/" + "/".join(s[k + 1 :]))))
        return out

    # -- where things go --

    def _pick(self, inst: Optional[Instance], score) -> Optional[str]:
        """The best-scoring folder of ``inst``; if it has none, of the
        nearest enclosing instance that does, then the project."""
        chain: list[Optional[Instance]] = []
        if inst is not None:
            chain.append(inst)
            chain.extend(reversed(self.ancestors(inst)))
        chain.append(None)
        for level in chain:
            best = None
            for f, path in self.folders_of(level):
                sc = score(f, path)
                if sc > 0 and (best is None or sc > best[0]):
                    best = (sc, path)
            if best:
                return best[1]
        return None

    def render_dir(self, inst: Optional[Instance]) -> Optional[str]:
        def score(f: dict, path: str) -> int:
            name = _segs(f.get("path", ""))[-1].lower()
            kind = (f.get("type") or "").lower()
            if kind in ("render", "renders"):
                return 3
            if name in ("renders", "render"):
                return 2
            if name in ("output", "out"):
                return 1
            return 0

        return self._pick(inst, score)

    def blender_dir(self, inst: Optional[Instance]) -> Optional[str]:
        """Where this instance's .blend files live: a folder named Blender,
        else one whose file types include .blend, else its software folder
        (or that folder's ``Blender``/``blender`` child when one exists)."""

        def score(f: dict, path: str) -> int:
            name = _segs(f.get("path", ""))[-1].lower()
            exts = (f.get("exts") or "").lower().split()
            kind = (f.get("type") or "").lower()
            if name == "blender":
                return 4
            if "blend" in exts:
                return 3
            if kind == "software" or name in ("software", "scripts", "work"):
                return 2
            return 0

        found = self._pick(inst, score)
        if found and os.path.basename(found).lower() != "blender":
            for child in ("Blender", "blender"):
                p = os.path.join(found, child)
                if os.path.isdir(p):
                    return p
        return found

    def locate(self, filepath: str) -> Optional[Instance]:
        """The deepest instance whose folder contains ``filepath``."""
        rel = relative_to(filepath, self.root)
        if rel is None:
            return None
        best = None
        for i in self.instances:
            ir = "/".join(_segs(i.rel))
            if rel == ir or rel.startswith(ir + "/"):
                if best is None or len(ir) > len(best.rel):
                    best = i
        return best


def load_manifest(project_root: str) -> Optional[Project]:
    path = os.path.join(project_root, MANIFEST_DIR, MANIFEST_FILE)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("app") != "organisicator":
        return None
    if int(data.get("format", 0)) > SUPPORTED_FORMAT:
        return None
    insts = []
    for d in data.get("instances") or []:
        try:
            insts.append(
                Instance(
                    id=int(d["id"]),
                    parent=None if d.get("parent") is None else int(d["parent"]),
                    decl=str(d["decl"]),
                    value=str(d["value"]),
                    code=str(d.get("code") or ""),
                    rel=str(d["rel"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return Project(
        name=str(data.get("name") or os.path.basename(project_root)),
        code=str(data.get("code") or ""),
        root=os.path.normpath(project_root),
        anatomy=data.get("anatomy"),
        folders=[f for f in data.get("folders") or [] if isinstance(f, dict)],
        levels=[lv for lv in data.get("levels") or [] if isinstance(lv, dict)],
        instances=insts,
    )


def find_projects(managed_root: str, max_depth: int = 6) -> list[Project]:
    """Every organisicator project under ``managed_root`` (a project's own
    subtree isn't searched further). Hidden folders are skipped."""
    out: list[Project] = []
    if not managed_root or not os.path.isdir(managed_root):
        return out

    def walk(d: str, depth: int) -> None:
        if os.path.isfile(os.path.join(d, MANIFEST_DIR, MANIFEST_FILE)):
            p = load_manifest(d)
            if p is not None:
                out.append(p)
                return
        if depth >= max_depth:
            return
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name.lower())
        except OSError:
            return
        for e in entries:
            if e.name.startswith(".") or e.name.startswith("_"):
                continue
            try:
                if e.is_dir(follow_symlinks=False):
                    walk(e.path, depth + 1)
            except OSError:
                continue

    walk(os.path.normpath(managed_root), 0)
    return out


def locate(projects: list[Project], filepath: str) -> tuple[Optional[Project], Optional[Instance]]:
    for p in projects:
        if relative_to(filepath, p.root) is not None:
            return p, p.locate(filepath)
    return None, None


# ---- paths across operating systems ----------------------------------------


def _norm(p: str) -> str:
    p = p.replace("\\", "/")
    while len(p) > 1 and p.endswith("/") and not p.endswith(":/"):
        p = p[:-1]
    return p


def _windowsy(p: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", p)) or p.startswith("\\\\")


def relative_to(path: str, root: str) -> Optional[str]:
    """``path`` relative to ``root`` ('/'-separated, '' for the root itself),
    or None when it isn't inside. Case-insensitive for Windows paths."""
    if not path or not root:
        return None
    a, b = _norm(path), _norm(root)
    fold = _windowsy(path) or _windowsy(root) or current_os() == "windows"
    ca, cb = (a.lower(), b.lower()) if fold else (a, b)
    if ca == cb:
        return ""
    prefix = cb if cb.endswith("/") else cb + "/"
    if ca.startswith(prefix):
        return a[len(prefix):]
    return None


def remap(path: str, roots: dict[str, str], target: Optional[str] = None) -> Optional[str]:
    """If ``path`` is absolute under another OS's managed root, the same
    file under ``target``'s root (native separators for that OS); else None.

    Blender-relative paths (``//…``) are left alone — they already travel."""
    target = target or current_os()
    dst_root = roots.get(target) or ""
    if not path or not dst_root or path.startswith("//"):
        return None
    for key in OS_KEYS:
        if key == target:
            continue
        src_root = roots.get(key) or ""
        if not src_root:
            continue
        rel = relative_to(path, src_root)
        if rel is None:
            continue
        sep = "\\" if target == "windows" else "/"
        base = _norm(dst_root)
        if target == "windows":
            base = base.replace("/", "\\")
        if not rel:
            return base
        return base.rstrip(sep) + sep + rel.replace("/", sep)
    return None


# ---- names -----------------------------------------------------------------

_VERSION = re.compile(r"^(.*?)([._-]?)v(\d+)$", re.IGNORECASE)


def next_version(stem: str) -> str:
    """``cf_010_light_v003`` → ``cf_010_light_v004``; no version → ``_v002``."""
    m = _VERSION.match(stem)
    if not m:
        return f"{stem}_v002"
    head, sep, num = m.groups()
    vchar = stem[len(head) + len(sep)]
    return f"{head}{sep}{vchar}{int(num) + 1:0{len(num)}d}"


def fill(template: str, tokens: dict[str, str]) -> str:
    """Replaces ``{token}``s; unknown ones are dropped. The result is made
    safe as a relative path (no spaces, no ``..``)."""

    def sub(m: re.Match) -> str:
        return tokens.get(m.group(1).lower(), "")

    out = re.sub(r"\{([A-Za-z_.]+)\}", sub, template)
    parts = []
    for seg in re.split(r"[\\/]", out):
        seg = re.sub(r"\s+", "_", seg.strip())
        seg = re.sub(r'[<>:"|?*]', "", seg)
        if seg in ("", ".", ".."):
            continue
        parts.append(seg)
    trailing = out.endswith(("/", "\\"))
    joined = "/".join(parts)
    return joined + ("/" if trailing and joined else "")


def tokens_for(project: Project, inst: Optional[Instance], blend_stem: str = "", task: str = "", scene: str = "") -> dict[str, str]:
    t = {
        "project": project.name,
        "project.code": project.code,
        "blend": blend_stem,
        "task": task,
        "scene": scene,
    }
    m = _VERSION.match(blend_stem or "")
    t["version"] = f"v{m.group(3)}" if m else ""
    chain = ([] if inst is None else project.ancestors(inst) + [inst])
    for i in chain:
        t[i.var] = i.value
        t[f"{i.var}.code"] = i.code or i.value
    if inst is not None:
        t["name"] = inst.value
        t["code"] = inst.code or inst.value
        t.setdefault("shot", inst.value)
    else:
        t["name"] = project.name
        t["code"] = project.code
    return t


def render_target(project: Project, inst: Optional[Instance], blend_stem: str, template: str, scene: str = "") -> Optional[str]:
    """Absolute render output for a file in ``inst`` (``None`` when the
    project defines no render folder). A template ending in ``/`` or ``_``
    lets Blender append the frame number and extension."""
    rdir = project.render_dir(inst)
    if not rdir:
        return None
    rel = fill(template, tokens_for(project, inst, blend_stem, scene=scene))
    if not rel:
        return rdir + os.sep
    trailing = rel.endswith("/")
    out = os.path.join(rdir, *rel.rstrip("/").split("/"))
    return out + (os.sep if trailing else "")


def blend_files(directory: Optional[str]) -> list[tuple[str, float]]:
    """``.blend`` files in a folder, newest first, as (path, mtime)."""
    if not directory or not os.path.isdir(directory):
        return []
    out = []
    try:
        for e in os.scandir(directory):
            if e.is_file() and e.name.lower().endswith(".blend") and not e.name.startswith("."):
                try:
                    out.append((e.path, e.stat().st_mtime))
                except OSError:
                    pass
    except OSError:
        return []
    out.sort(key=lambda t: t[1], reverse=True)
    return out
