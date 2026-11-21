"""Thin client for wobble's scripting protocol.

The protocol is the API: newline-delimited JSON-RPC 2.0 on 127.0.0.1, served
by `wobble --api`. This module knows no method names — it asks the app for
its schema (`api_schema`) and exposes each method as a Python method, so it
cannot drift from the app and needs no release when a method is added.

    import wobble
    w = wobble.connect()
    blur = w.node_create(type="Blur", inputs=[1])["id"]
    w.node_set(id=blur, values={"BlurSize": 12})
    w.view_set(id=blur)
    open("look.png", "wb").write(w.capture())

Standard library only, Python 3.9+ (every VFX Reference Platform year that
matters).
"""

import base64
import json
import os
import socket
import sys
import traceback

__all__ = ["connect", "Client", "Batch", "Addon", "WobbleError", "DEFAULT_PORT"]

DEFAULT_PORT = 7979


class WobbleError(Exception):
    """The app refused or failed a call. `code` follows JSON-RPC."""

    def __init__(self, code, message):
        super().__init__(f"{message} ({code})")
        self.code = code
        self.message = message


class Client:
    def __init__(self, port=None, host="127.0.0.1", timeout=90.0):
        port = port or int(os.environ.get("WOBBLE_API_PORT", DEFAULT_PORT))
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._file = self._sock.makefile("rwb")
        self._next = 1
        self.schema = self.call("api_schema")
        self._methods = {m["name"]: m for m in self.schema["methods"]}

    def call(self, method, **params):
        """Call any method by name. Returns its result; raises WobbleError."""
        req = {"jsonrpc": "2.0", "id": self._next, "method": method, "params": params}
        self._next += 1
        self._file.write(json.dumps(req).encode() + b"\n")
        self._file.flush()
        line = self._file.readline()
        if not line:
            raise WobbleError(-32099, "connection closed")
        resp = json.loads(line)
        if "error" in resp:
            raise WobbleError(resp["error"].get("code"), resp["error"].get("message"))
        return resp.get("result")

    def __getattr__(self, name):
        m = self.__dict__.get("_methods", {}).get(name)
        if m is None:
            raise AttributeError(f"wobble has no method {name!r} (see .schema)")

        def method(**params):
            return self.call(name, **params)

        method.__name__ = name
        method.__doc__ = m["summary"]
        return method

    def __dir__(self):
        return list(super().__dir__()) + list(self._methods)

    def capture(self, max_size=768):
        """The viewer as PNG bytes, once the playhead's frame has rendered."""
        r = self.call("viewer_capture", max_size=max_size)
        return base64.b64decode(r["png_base64"])

    def sample(self, x=None, y=None, w=None, h=None):
        """Scene-linear mean/min/max RGBA of the viewed output over a
        rectangle (default the whole frame; w=h=1 for one pixel)."""
        rect = {k: v for k, v in dict(x=x, y=y, w=w, h=h).items() if v is not None}
        return self.call("pixel_sample", **rect)

    def run(self, id, action=None, wait=True, timeout=3600.0, poll=0.25, progress=None):
        """Press a node's run button (a Write's render, a bake, a track, a
        key) and, by default, wait for it. Raises WobbleError if the node
        ends with an error. `progress(jobs)` is called on every poll."""
        params = {"id": id}
        if action:
            params["action"] = action
        r = self.call("node_run", **params)
        return self.wait(id=id, timeout=timeout, poll=poll, progress=progress) if wait else r

    def wait(self, id=None, timeout=3600.0, poll=0.25, progress=None):
        """Block until nothing is running. With `id`, raise that node's
        error if it has one when the work stops."""
        import time

        end = time.monotonic() + timeout
        while True:
            j = self.call("jobs_get")
            if progress:
                progress(j)
            if not j["busy"]:
                err = j["errors"].get(str(id)) if id is not None else None
                key = j.get("key") or {}
                if id is not None and key.get("node") == id and key.get("failed"):
                    err = key["failed"]
                if err:
                    raise WobbleError(-32000, err)
                return j
            if time.monotonic() > end:
                raise WobbleError(-32000, "timed out waiting for wobble to finish")
            time.sleep(poll)

    def batch(self):
        """Several calls as ONE undo step, all or nothing:

            with w.batch() as b:
                g = b.node_create(type="Grade", inputs=[read])
                b.node_create(type="Blur", inputs=[g])   # g is that node
            blur_id = b.results[1]["id"]
        """
        return Batch(self)

    def close(self):
        self._file.close()
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Batch:
    """Calls collected under `with w.batch() as b:` and sent as one `batch`
    when the block ends. A call returns a reference to the node it will
    make, usable as an id in later calls of the same batch."""

    def __init__(self, client):
        self._client = client
        self._calls = []
        self.results = None

    def call(self, method, **params):
        self._calls.append({"method": method, "params": params})
        return f"${len(self._calls) - 1}"

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda **params: self.call(name, **params)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        if exc_type is None and self._calls:
            self.results = self._client.call("batch", calls=self._calls)["results"]


def connect(port=None, host="127.0.0.1"):
    """Connect to a running wobble (`wobble --api`)."""
    return Client(port=port, host=host)


class Addon:
    """The process side of an addon (`addon.json` with a `run` entry).

    The app starts the process with WOBBLE_API_PORT and WOBBLE_ADDON set;
    register handlers, then `run()` waits for what the user does:

        addon = wobble.Addon()

        @addon.command("add_blur")
        def add_blur(w):
            w.node_create(type="Blur", inputs=[w.graph_get()["view"]])

        @addon.row("main", "size")
        def size(w, value):
            ...

        @addon.pick("media")          # a Load picker ("pickers" in addon.json)
        def media(w, node):
            ...                       # answer with w.load_set(id=node, ...)

        addon.run()

    ``addon.event`` is the event being handled (a list row's double-click
    has ``"activate": True``). A handler that raises is printed (the addons window shows the output)
    and the loop carries on.
    """

    def __init__(self, id=None, client=None):
        self.id = id or os.environ.get("WOBBLE_ADDON")
        if not self.id:
            raise WobbleError(-32602, "no addon id: run me from wobble, or pass id=")
        self.w = client or connect()
        self._commands = {}
        self._rows = {}
        self._picks = {}
        self.event = {}

    def command(self, name):
        def deco(fn):
            self._commands[name] = fn
            return fn
        return deco

    def row(self, panel, row):
        def deco(fn):
            self._rows[(panel, row)] = fn
            return fn
        return deco

    def pick(self, picker):
        """A handler for a picker's button on a node's panel: fn(w, node)."""
        def deco(fn):
            self._picks[picker] = fn
            return fn
        return deco

    def set(self, panel, open=None, **values):
        """Set rows of a panel (and show or hide it)."""
        params = {"addon": self.id, "panel": panel, "values": values}
        if open is not None:
            params["open"] = open
        return self.w.call("addon_panel_set", **params)

    def status(self, text):
        return self.w.call("addon_status", addon=self.id, text=text)

    def events(self):
        """Yield events forever; returns when the app goes away."""
        while True:
            try:
                r = self.w.call("addon_events", addon=self.id, timeout_ms=20000)
            except (OSError, WobbleError) as e:
                if isinstance(e, WobbleError) and e.code != -32099:
                    raise
                return
            for ev in r.get("events", []):
                yield ev

    def run(self):
        for ev in self.events():
            self.event = ev
            if ev.get("type") == "command":
                fn, args = self._commands.get(ev.get("command")), ()
            elif ev.get("type") == "pick":
                fn, args = self._picks.get(ev.get("picker")), (ev.get("node"),)
            else:
                fn, args = self._rows.get((ev.get("panel"), ev.get("row"))), (ev.get("value"),)
            if fn is None:
                continue
            try:
                fn(self.w, *args)
            except Exception as e:  # keep the addon alive; say what broke
                traceback.print_exc()
                sys.stderr.flush()
                try:
                    self.status(f"error: {e}")
                except Exception:
                    pass
