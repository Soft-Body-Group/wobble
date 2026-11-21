"""The media browser: footage for a Load picked from organisicator's
database (its local API, ~/.organisicator/api.json) instead of the file
system. A shot lists its own footage and the footage linked to it from
elsewhere in the project (a sequence's assets, a library), which
organisicator keeps as footage links.

Pure functions over the API's answers, so they test without either app;
main.py does the calls."""

from typing import Optional

PROJECT = "(project)"

# What a Load opens: image sequences, movies, stills (the decoder's list).
LOADABLE = {
    "exr", "dpx", "cin", "png", "jpg", "jpeg", "tif", "tiff", "tga", "hdr", "webp", "bmp",
    "mov", "mp4", "m4v", "mxf", "mkv", "webm", "avi",
}


def shot_options(instances: list) -> list:
    """``[(label, instance id or None)]``: the project itself first, then
    every sequence / shot in tree order, labelled by its chain of names
    (``coffee › cf_010``)."""
    by_id = {i["id"]: i for i in instances}

    def chain(i):
        names, seen = [], set()
        while i is not None and i["id"] not in seen:
            seen.add(i["id"])
            names.append(i["value"])
            i = by_id.get(i.get("parent"))
        return list(reversed(names))

    rows = [(" › ".join(chain(i)), i["id"]) for i in instances]
    rows.sort(key=lambda r: [s.lower() for s in r[0].split(" › ")])
    return [(PROJECT, None)] + rows


def _loadable(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return "." in name and name.rsplit(".", 1)[-1].lower() in LOADABLE


def _span(a: dict) -> str:
    if a.get("kind") == "sequence":
        return f"{a['frames']}f" if a.get("frames") else "sequence"
    return ""


def footage_rows(answer: dict, instances: list, flt: str = "") -> list:
    """What the list shows for one shot, from
    ``/v1/projects/{id}/assets?instance=N``: its own footage, then what is
    linked to it. Each row is ``{label, id, path, rel, linked}``; ``id`` is
    None for a linked file organisicator hasn't indexed yet (it loads by
    path then). Labels are unique — the list picks by label."""
    names = {i.get("rel"): i.get("value") for i in instances}
    q = flt.strip().lower()
    rows = []
    for a in answer.get("assets", []):
        rel = a.get("rel", "")
        if not _loadable(rel):
            continue
        span = _span(a)
        rows.append({
            "label": rel + (f"  ·  {span}" if span else ""),
            "id": a.get("id"), "path": None, "rel": rel, "linked": False,
        })
    for a in answer.get("linked", []):
        rel = a.get("rel", "")
        if not _loadable(rel):
            continue
        via = a.get("via", "")
        where = "the project" if not via else names.get(via, via.rsplit("/", 1)[-1])
        bits = [b for b in (_span(a), f"linked via {where}", "" if a.get("exists", True) else "missing") if b]
        rows.append({
            "label": f"{rel}  ·  " + "  ·  ".join(bits),
            "id": a.get("id"), "path": a.get("path"), "rel": rel, "linked": True,
        })
    if q:
        rows = [r for r in rows if q in r["label"].lower()]
    return rows


def info(row: Optional[dict], answer: Optional[dict]) -> str:
    """The line under the list."""
    if answer is not None and not answer.get("indexed", True):
        return "not indexed yet — open this project's Work view in organisicator (Rescan)"
    if row is None:
        return "pick footage; double-click loads it"
    if row["id"] is None:
        return "linked, not indexed yet: loads by path (Rescan in organisicator to load by id)"
    return "loads by organisicator id: keeps working when the project moves or the shot is renamed"


def locate(asset_answer: dict) -> tuple:
    """``/v1/assets/{id}`` → (project id, innermost instance id or None):
    where a Load's current asset is, to open the browser on it."""
    pid = (asset_answer.get("project") or {}).get("id")
    shot = asset_answer.get("shot") or []
    return pid, (shot[-1].get("id") if shot else None)
