"""Organisicator for wobble: see addon.json for what it declares, shots.py
for where things go. Everything here is reacting to the panel."""

import os
import subprocess
import sys

import core
import shots
import wobble

addon = wobble.Addon()
P = "main"
state = {
    "root": "",
    "projects": [],  # [core.Project]
    "project": None,  # core.Project
    "shot": None,  # core.Instance or None (the project itself)
    "task": "comp",
}


def project_names():
    return [p.name for p in state["projects"]]


def show():
    """Push everything the panel shows from ``state``."""
    proj, inst = state["project"], state["shot"]
    values = {
        "root": state["root"],
        "project": {"options": project_names(), "value": proj.name if proj else ""},
    }
    if proj is None:
        values["shot"] = {"options": [], "value": ""}
        values["comp"] = {"options": [], "value": ""}
        values["where"] = "no projects under the managed root" if state["root"] else "set the managed root"
        addon.set(P, **values)
        return
    labels = [lbl for lbl, _ in shots.shot_labels(proj)]
    values["shot"] = {"options": labels, "value": shots.label_of(proj, inst)}
    cdir = shots.comp_dir(proj, inst)
    comps = [os.path.basename(p) for p in shots.comp_files(cdir)]
    values["comp"] = {"options": comps, "value": comps[0] if comps else ""}
    rdir = proj.render_dir(inst)
    pdir = shots.plates_dir(proj, inst)

    def rel(p):
        return os.path.relpath(p, proj.root) if p else "—"

    values["where"] = f"comps: {rel(cdir)}\nrenders: {rel(rdir)}\nplates: {rel(pdir)}"
    addon.set(P, **values)


def scan():
    state["projects"] = core.find_projects(state["root"]) if state["root"] else []
    names = project_names()
    if state["project"] is None or state["project"].name not in names:
        state["project"] = state["projects"][0] if state["projects"] else None
        state["shot"] = None
    else:
        # Reload the picked project so new shots show up.
        old = state["shot"]
        state["project"] = state["projects"][names.index(state["project"].name)]
        state["shot"] = state["project"].instance(old.id) if old else None


def follow_open_comp(w):
    """Pick the shot the open comp is saved in, as the Blender add-on picks
    the open .blend's."""
    setup = w.app_info().get("setup")
    if not setup:
        return False
    proj, inst = core.locate(state["projects"], setup)
    if proj is None:
        return False
    state["project"], state["shot"] = proj, inst
    return True


def sync(w):
    """Follow the open comp when it changed under us (opened or saved from
    the app), so an action always lands in the shot the comp belongs to."""
    setup = w.app_info().get("setup")
    if setup != state.get("last_setup"):
        state["last_setup"] = setup
        if follow_open_comp(w):
            show()


def need_shot(w):
    sync(w)
    if state["project"] is None:
        raise RuntimeError("pick a project first")
    return state["project"], state["shot"]


# ---- actions ---------------------------------------------------------------


def route(w, comp_stem=None):
    """Point every Write at the shot's render folder. Returns how many."""
    proj, inst = need_shot(w)
    if comp_stem is None:
        setup = w.app_info().get("setup")
        comp_stem = os.path.splitext(os.path.basename(setup))[0] if setup else None
    if not comp_stem:
        nxt = shots.new_comp_path(proj, inst, state["task"])
        comp_stem = os.path.splitext(os.path.basename(nxt))[0] if nxt else "comp"
    # By the parameter record, which names the kind whatever the menu calls it.
    writes = [n for n in w.graph_get(include_kind=True)["nodes"] if "Write" in (n.get("kind") or {})]
    names = [n.get("name") or f"write{n['id']}" for n in writes]
    targets = shots.write_targets(proj, inst, comp_stem, names)
    if writes and not targets:
        raise RuntimeError(f"{proj.name} has no render folder in its anatomy")
    if targets:
        w.call("batch", calls=[
            {"method": "node_patch", "params": {"id": n["id"], "kind": {"Write": {"path": targets[name]}}}}
            for n, name in zip(writes, names)
        ])
    return len(writes)


def save_as(w, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = route(w, os.path.splitext(os.path.basename(path))[0])
    w.setup_save(path=path)
    show()
    addon.status(f"saved {os.path.basename(path)}" + (f", {n} Write(s) to the shot's renders" if n else ""))


def save(w, _v=None):
    proj, inst = need_shot(w)
    # Already in this shot's comp folder: save over it, like ⌘S.
    setup = w.app_info().get("setup")
    cdir = shots.comp_dir(proj, inst)
    if setup and cdir and os.path.dirname(os.path.abspath(setup)) == os.path.abspath(cdir):
        return save_as(w, setup)
    path = shots.new_comp_path(proj, inst, state["task"])
    if not path:
        raise RuntimeError(f"{proj.name} has no software/comp folder for {shots.label_of(proj, inst)}")
    save_as(w, path)


def version_up(w, _v=None):
    setup = w.app_info().get("setup")
    if not setup:
        return save(w)
    save_as(w, shots.version_up(setup))


def open_comp(w, _v=None):
    proj, inst = need_shot(w)
    cdir = shots.comp_dir(proj, inst)
    files = shots.comp_files(cdir)
    pick = state.get("comp") or (os.path.basename(files[0]) if files else "")
    if not pick:
        raise RuntimeError("no comps in this shot yet: save one first")
    path = os.path.join(cdir, pick)
    w.setup_load(path=path)
    addon.status(f"opened {pick} (undo brings the old comp back)")


def add_plates(w, _v=None):
    proj, inst = need_shot(w)
    seqs = shots.sequences(shots.plates_dir(proj, inst))
    if not seqs:
        raise RuntimeError("no plates in this shot's plates folder")
    # By organisicator's asset id, so the Loads keep working after the
    # project moves or the shot is renamed.
    ids = shots.fetch_asset_ids(proj, inst, seqs)
    unindexed = [s for s in seqs if s not in ids]
    if unindexed and not ids:
        raise RuntimeError("organisicator hasn't indexed these plates yet — Rescan the project in its Work view")
    calls = [
        {"method": "node_create", "params": {"type": "Load", "asset": {"id": ids[s]}, "name": os.path.basename(os.path.dirname(s)) or "plate"}}
        for s in seqs if s in ids
    ]
    r = w.call("batch", calls=calls)
    ids = [x.get("id") for x in (r.get("results") or []) if isinstance(x, dict)]
    if ids and ids[0] is not None:
        w.view_set(id=ids[0])
    note = f" ({len(unindexed)} not indexed yet — Rescan in organisicator)" if unindexed else ""
    addon.status(f"added {len(calls)} Load(s){note}")


def reveal(w, _v=None):
    proj, inst = need_shot(w)
    path = proj.path_of(inst)
    opener = {"darwin": ["open"], "win32": ["explorer"]}.get(sys.platform, ["xdg-open"])
    subprocess.Popen(opener + [path])


def panel(w, _v=None):
    addon.set(P, open=True)


# ---- rows ------------------------------------------------------------------


@addon.row(P, "root")
def set_root(w, v):
    state["root"] = os.path.expanduser(str(v or "").strip())
    prefs = shots.load_prefs()
    prefs["root"] = state["root"]
    shots.save_prefs(prefs)
    scan()
    show()
    addon.status(f"{len(state['projects'])} project(s)")


@addon.row(P, "project")
def pick_project(w, v):
    for p in state["projects"]:
        if p.name == v:
            state["project"], state["shot"] = p, None
    show()


@addon.row(P, "shot")
def pick_shot(w, v):
    proj = state["project"]
    if proj is not None:
        for lbl, inst in shots.shot_labels(proj):
            if lbl == v:
                state["shot"] = inst
    show()


@addon.row(P, "comp")
def pick_comp(w, v):
    state["comp"] = v


@addon.row(P, "task")
def set_task(w, v):
    state["task"] = str(v or "").strip()


def rescan(w, _v=None):
    scan()
    show()
    addon.status(f"{len(state['projects'])} project(s)")


for row, fn in [("open", open_comp), ("save", save), ("version_up", version_up), ("route", lambda w, _v=None: addon.status(f"{route(w)} Write(s) pointed at the shot's renders")),
                ("plates", add_plates), ("reveal", reveal), ("rescan", rescan)]:
    addon.row(P, row)(fn)
for cmd, fn in [("panel", panel), ("save", save), ("version_up", version_up), ("route", lambda w: addon.status(f"{route(w)} Write(s) pointed at the shot's renders")), ("plates", add_plates)]:
    addon.command(cmd)(fn)


# ---- start -----------------------------------------------------------------

state["root"] = shots.load_prefs().get("root") or shots.default_root()
scan()
state["last_setup"] = addon.w.app_info().get("setup")
follow_open_comp(addon.w)
show()
addon.status(f"{len(state['projects'])} project(s) under {state['root'] or '(no root)'}")
addon.run()
