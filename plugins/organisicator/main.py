"""Organisicator for wobble: see addon.json for what it declares, shots.py
for where things go. Everything here is reacting to the panel."""

import os
import subprocess
import sys

import core
import media
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


# ---- load from project: the media browser ----------------------------------
#
# The Load panel's "from project…" button (a picker in addon.json) opens this
# with the Load it fills; "load from project…" in the menu opens it for a new
# Load. Everything comes from organisicator's database through its API.

M = "media"
mstate = {
    "node": None,  # the Load being filled, or None = make a new one
    "projects": [],  # /v1/projects
    "pid": None,
    "root": "",  # the project's root on this machine
    "instances": [],  # /v1/projects/{pid} instances
    "all": None,  # /v1/projects/{pid}/assets?recursive=1
    "counts": {},  # media.folder_counts(all)
    "expanded": {""},
    "tree": [],  # [(label, folder rel)]
    "folder": "",  # the picked folder ("" = the project)
    "recursive": False,
    "answer": None,  # what the footage list is built from
    "rows": [],
    "pick": None,  # the picked row
    "filter": "",
}


def m_rows():
    mstate["rows"] = media.footage_rows(mstate["answer"] or {}, mstate["instances"], mstate["filter"])
    if mstate["pick"] and not any(r["label"] == mstate["pick"]["label"] for r in mstate["rows"]):
        mstate["pick"] = None


def m_register(folder):
    """Ask organisicator to index ``folder`` (rel; None = the whole project)
    where it is, so renders written since its last scan — by Blender here,
    or on another machine and synced in by Dropbox — are listed."""
    pid = mstate["pid"]
    if pid is None:
        return
    body = {"project": pid} if folder is None else {"path": os.path.join(mstate["root"], *folder.split("/"))}
    try:
        shots.api_post("/v1/register", body)
    except RuntimeError as e:
        addon.status(f"listing what organisicator last indexed ({e})")


def m_fetch_all():
    pid = mstate["pid"]
    mstate["all"] = shots.api_get(f"/v1/projects/{pid}/assets?recursive=1") if pid is not None else None
    mstate["counts"] = media.folder_counts(mstate["all"] or {})
    if mstate["folder"] not in mstate["counts"]:
        mstate["folder"] = ""
    m_tree()


def m_tree():
    mstate["expanded"] = media.expand_to(mstate["folder"], mstate["expanded"])
    mstate["tree"] = media.tree_rows(mstate["counts"], mstate["expanded"])


def m_list():
    """The footage list for the picked folder: its own assets (recursive:
    everything below), and when the folder is a shot / sequence (or the
    project), the footage linked to it from elsewhere."""
    folder, pid = mstate["folder"], mstate["pid"]
    ans = media.under(mstate["all"] or {}, folder, mstate["recursive"])
    inst = media.instance_at(folder, mstate["instances"])
    if pid is not None and (inst is not None or folder == ""):
        q = f"?instance={inst}" if inst is not None else ""
        try:
            ans["linked"] = shots.api_get(f"/v1/projects/{pid}/assets{q}").get("linked", [])
        except RuntimeError:
            pass
    mstate["answer"] = ans
    m_rows()


def m_fetch_project(pid, folder="", index=False):
    mstate["pid"], mstate["folder"], mstate["pick"] = pid, folder, None
    mstate["expanded"] = {""}
    p = shots.api_get(f"/v1/projects/{pid}") if pid is not None else {}
    mstate["instances"] = p.get("instances", [])
    mstate["root"] = p.get("root", "")
    if index:
        m_register(None)
    elif folder:
        m_register(folder)
    m_fetch_all()
    m_list()


def m_show():
    names = [p["name"] for p in mstate["projects"]]
    cur = next((p["name"] for p in mstate["projects"] if p["id"] == mstate["pid"]), "")
    here = next((lbl for lbl, d in mstate["tree"] if d == mstate["folder"]), "")
    node = mstate["node"]
    addon.set(
        M,
        target=f"fills Load {node}" if node is not None else "makes a new Load",
        project={"options": names, "value": cur},
        tree={"options": [lbl for lbl, _ in mstate["tree"]], "value": here},
        recursive=mstate["recursive"],
        footage={"options": [r["label"] for r in mstate["rows"]], "value": (mstate["pick"] or {}).get("label", "")},
        info=media.info(mstate["pick"], mstate["answer"]),
        load="load" if node is not None else "add Load",
    )


def m_start(w, node):
    """Open the browser for `node` (None = a new Load), on the footage the
    Load holds now, else on the shot the Projects panel has picked."""
    mstate["node"] = node
    mstate["projects"] = [p for p in shots.api_get("/v1/projects") if p.get("online", True)]
    pid, folder = None, ""
    asset = (w.node_get(id=node).get("asset") or {}) if node is not None else {}
    if asset.get("id"):
        try:
            a = shots.api_get(f"/v1/assets/{asset['id']}")
            pid = media.locate(a)[0]
            root = next((p.get("root", "") for p in mstate["projects"] if p["id"] == pid), "")
            folder = media.folder_of(a.get("path", ""), root)
        except RuntimeError:
            pass
    if pid is None and state["project"] is not None:
        root = os.path.realpath(state["project"].root)
        pid = next((p["id"] for p in mstate["projects"] if os.path.realpath(p.get("root", "")) == root), None)
        if pid is not None and state["shot"] is not None:
            folder = state["shot"].rel.replace("\\", "/").strip("/")
    if pid is None and mstate["projects"]:
        pid = mstate["projects"][0]["id"]
    m_fetch_project(pid, folder)
    m_show()
    addon.set(M, open=True)


def m_load(w, new=False):
    row = mstate["pick"]
    if row is None:
        raise RuntimeError("pick footage first")
    what = {"asset": {"id": row["id"]}} if row["id"] is not None else {"path": row["path"]}
    node = mstate["node"]
    if new or node is None:
        name = row["rel"].rsplit("/", 1)[-1].split(".")[0] or "footage"
        r = w.node_create(type="Load", name=name, **what)
        node = r.get("id")
        if not new:
            mstate["node"] = node
        w.view_set(id=node)
    else:
        r = w.load_set(id=node, **what)
    err = r.get("load_error")
    addon.status(f"couldn't load {row['rel']}: {err}" if err else f"loaded {row['rel']}")
    if not err and not new:
        addon.set(M, open=False)


@addon.pick("media")
def pick_media(w, node):
    m_start(w, node)


@addon.row(M, "project")
def m_pick_project(w, v):
    pid = next((p["id"] for p in mstate["projects"] if p["name"] == v), None)
    m_fetch_project(pid)
    m_show()


@addon.row(M, "tree")
def m_pick_folder(w, v):
    """Picking a folder opens it (and lists it); picking the open one again
    folds it."""
    folder = next((d for lbl, d in mstate["tree"] if lbl == v), None)
    if folder is None:
        return
    if folder == mstate["folder"] and folder in mstate["expanded"] and folder:
        mstate["expanded"].discard(folder)
        mstate["tree"] = media.tree_rows(mstate["counts"], mstate["expanded"])
    else:
        mstate["folder"], mstate["pick"] = folder, None
        mstate["expanded"].add(folder)
        if folder:
            m_register(folder)
            m_fetch_all()
        else:
            m_tree()
        m_list()
    m_show()


@addon.row(M, "recursive")
def m_recursive(w, v):
    mstate["recursive"] = bool(v)
    m_list()
    m_show()


@addon.row(M, "filter")
def m_filter(w, v):
    mstate["filter"] = str(v or "")
    m_rows()
    m_show()


@addon.row(M, "footage")
def m_pick(w, v):
    mstate["pick"] = next((r for r in mstate["rows"] if r["label"] == v), None)
    if addon.event.get("activate"):
        return m_load(w)
    m_show()


@addon.row(M, "load")
def m_load_row(w, _v=None):
    m_load(w)


@addon.row(M, "new")
def m_new(w, _v=None):
    m_load(w, new=True)


@addon.row(M, "refresh")
def m_refresh(w, _v=None):
    mstate["projects"] = [p for p in shots.api_get("/v1/projects") if p.get("online", True)]
    keep = mstate["expanded"]
    m_fetch_project(mstate["pid"], mstate["folder"], index=True)
    mstate["expanded"] |= {d for d in keep if d in mstate["counts"]}
    m_tree()
    m_show()


addon.command("browse")(lambda w: m_start(w, None))


# ---- start -----------------------------------------------------------------

state["root"] = shots.load_prefs().get("root") or shots.default_root()
scan()
state["last_setup"] = addon.w.app_info().get("setup")
follow_open_comp(addon.w)
show()
addon.status(f"{len(state['projects'])} project(s) under {state['root'] or '(no root)'}")
addon.run()
