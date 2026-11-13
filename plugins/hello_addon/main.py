"""The example addon: see addon.json for what it declares."""

import wobble

addon = wobble.Addon()
state = {"blur": None, "size": 12.0, "live": True}


def add_blur(w):
    view = w.app_info().get("view")
    if view is None:
        addon.status("view a node first")
        return
    blur = w.node_create(type="Blur", inputs=[view])["id"]
    w.node_set(id=blur, values={"BlurSize": state["size"]})
    w.view_set(id=blur)
    state["blur"] = blur
    addon.set("main", open=True, info=f"driving Blur {blur}")
    addon.status(f"added Blur {blur}")


addon.command("add_blur")(add_blur)
addon.row("main", "add")(lambda w, _v: add_blur(w))


@addon.command("count")
def count(w):
    addon.status(f"{len(w.graph_get()['nodes'])} nodes in the comp")


@addon.row("main", "live")
def live(w, v):
    state["live"] = bool(v)


@addon.row("main", "size")
def size(w, v):
    state["size"] = float(v)
    if state["live"] and state["blur"] is not None:
        w.node_set(id=state["blur"], values={"BlurSize": state["size"]})


addon.status("ready")
addon.run()
