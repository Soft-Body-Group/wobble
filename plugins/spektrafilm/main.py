"""The spektrafilm wobbin: see addon.json, and core.py for the logic.

A film node is a Shader node whose source the baker wrote; its recipe rides
in that source, so a saved comp keeps it and 'edit selected node' reads it
back. Exposure and mix are the node's own (uniform) params and never re-bake.
"""

import core
import wobble

addon = wobble.Addon()
w = addon.w

st = {
    "node": None,        # the film node the panel edits
    "recipe": core.default_recipe(),
    "films": [],
    "papers": [],
    "live": True,
}

# Panel slider -> (RuntimeParams path, default).
SLIDERS = {
    "print_exposure": ("enlarger.print_exposure", 1.0),
    "m_shift": ("enlarger.m_filter_shift", 0.0),
    "y_shift": ("enlarger.y_filter_shift", 0.0),
    "film_gamma": ("film_render.chemistry.gamma_factor", 1.0),
    "print_gamma": ("print_render.chemistry.gamma_factor", 1.0),
}


def film_label(f):
    return f"{f['label']}  ({f['id']})"


def paper_label(p):
    return f"{p['label']}  ({p['id']})"


def load_profiles():
    if not core.baker_ready():
        addon.set("film", info="the baker is not built yet: press 'build the baker' (needs cargo and ~/src/devl/spektrafilm-rs)")
        return False
    p = core.profiles()
    st["films"], st["papers"] = p["films"], p["papers"]
    return True


def show():
    """Push the current recipe into the panel."""
    r = st["recipe"]
    film = next((f for f in st["films"] if f["id"] == r["film"]), None)
    paper_opts = [core.AUTO_PRINT, core.SCAN] + [paper_label(p) for p in st["papers"]]
    if r.get("scan_film"):
        paper = core.SCAN
    elif r.get("print"):
        paper = next((paper_label(p) for p in st["papers"] if p["id"] == r["print"]), core.AUTO_PRINT)
    else:
        paper = core.AUTO_PRINT
    vals = {
        "film": {"options": [film_label(f) for f in st["films"]], "value": film_label(film) if film else ""},
        "print": {"options": paper_opts, "value": paper},
        "size": {"options": ["33", "25", "17"], "value": str(r.get("size", 33))},
        "couplers": bool(core.get_param(r, "film_render.dir_couplers.active", True)),
    }
    for row, (path, d) in SLIDERS.items():
        vals[row] = float(core.get_param(r, path, d))
    node = st["node"]
    vals["info"] = (f"editing {node_name(node)}" if node is not None
                    else "no film node yet: 'add film node' puts one after the viewed node")
    addon.set("film", **vals)


def node_name(nid):
    try:
        return f"'{w.node_get(id=nid).get('name') or nid}' (node {nid})"
    except wobble.WobbleError:
        return f"node {nid}"


def film_nodes():
    """Ids of every film node in the comp, by its recipe line."""
    out = []
    for n in w.graph_get(include_kind=True)["nodes"]:
        k = n.get("kind") or {}
        if "Shader" in k and core.recipe_of(k["Shader"].get("source")):
            out.append(n["id"])
    return out


def bake_into(nid, fresh=False):
    r = st["recipe"]
    r["space"] = core.space_of(w.app_info().get("working_space"))
    name = core.label_of(r, st["films"], st["papers"])
    r["label"] = name
    addon.status(f"baking {name}…")
    src = core.bake(r)
    shader = {"source": src}
    if fresh:
        shader["values"] = {"exposure": [0.0, 0, 0, 0], "mix": [1.0, 0, 0, 0]}
    w.node_patch(id=nid, kind={"Shader": shader}, name=name)
    addon.status(f"{name} baked")


def pick(_w=None, _v=None):
    """Edit the selected film node (or the only one in the comp)."""
    sel = w.app_info().get("selection") or []
    films = film_nodes()
    nid = next((s for s in sel if s in films), None)
    if nid is None and len(films) == 1:
        nid = films[0]
    if nid is None:
        addon.status("select a film node first" if films else "no film node in the comp")
        return
    src = w.node_get(id=nid)["kind"]["Shader"]["source"]
    st["node"], st["recipe"] = nid, core.recipe_of(src)
    show()


def edited():
    if st["live"] and st["node"] is not None:
        bake_into(st["node"])


@addon.command("panel")
def panel(_w):
    if load_profiles():
        if st["node"] is None:
            try:
                pick()
            except wobble.WobbleError:
                pass
        show()
    addon.set("film", open=True)


@addon.command("add")
def add(_w, _v=None):
    if not st["films"] and not load_profiles():
        return
    view = w.app_info().get("view")
    if view is None:
        addon.status("view a node first: the film node goes after it")
        return
    nid = w.node_create(type="Shader", inputs=[view])["id"]
    bake_into(nid, fresh=True)
    w.view_set(id=nid)
    w.selection_set(ids=[nid])
    st["node"] = nid
    show()


addon.row("film", "add")(add)
addon.row("film", "pick")(pick)


@addon.command("rebake")
def rebake(_w, _v=None):
    if st["node"] is None:
        pick()
    if st["node"] is not None:
        bake_into(st["node"])


addon.row("film", "rebake")(rebake)


@addon.row("film", "build")
def build(_w, _v):
    addon.status("building the baker (cargo build --release, minutes the first time)…")
    core.build_baker()
    load_profiles()
    show()
    addon.status("baker built")


@addon.row("film", "film")
def film(_w, v):
    fid = v.rsplit("(", 1)[-1].rstrip(")") if v else ""
    if not fid or fid == st["recipe"]["film"]:
        return
    st["recipe"]["film"] = fid
    # A new stock brings its own paper (and scans a slide).
    st["recipe"]["print"] = ""
    st["recipe"].pop("scan_film", None)
    show()
    edited()


@addon.row("film", "print")
def paper(_w, v):
    r = st["recipe"]
    if v == core.SCAN:
        r["scan_film"], r["print"] = True, ""
    elif v == core.AUTO_PRINT:
        r.pop("scan_film", None)
        r["print"] = ""
    else:
        r["scan_film"], r["print"] = False, v.rsplit("(", 1)[-1].rstrip(")")
    edited()


def slider(row):
    path, _d = SLIDERS[row]

    def handler(_w, v):
        core.set_param(st["recipe"], path, float(v))
        edited()

    addon.row("film", row)(handler)


for _row in SLIDERS:
    slider(_row)


@addon.row("film", "couplers")
def couplers(_w, v):
    core.set_param(st["recipe"], "film_render.dir_couplers.active", bool(v))
    edited()


@addon.row("film", "size")
def size(_w, v):
    st["recipe"]["size"] = int(v)
    edited()


@addon.row("film", "live")
def live(_w, v):
    st["live"] = bool(v)


if __name__ == "__main__":
    try:
        if load_profiles():
            addon.status("ready")
    except Exception as e:  # keep the addon up so the panel can say why
        addon.status(f"error: {e}")
    addon.run()
