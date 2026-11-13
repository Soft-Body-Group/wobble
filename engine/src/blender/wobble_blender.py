# SPDX-License-Identifier: GPL-3.0-or-later
# Part of wobble; see LICENSING.md.
"""wobble <-> Blender bridge.

One file, two ways to run it:

  launched   wobble writes this file into its cache directory on every
             launch and starts Blender with `--python`; the copy on disk is
             never the source of truth, so do not edit it there.
             $WOBBLE_PORT names the port to listen on, $WOBBLE_SHM_DIR the
             directory the pixel ring lives in.

  addon      installed in Blender (Preferences > Add-ons), it adds a
             "wobble" tab to the 3D viewport's sidebar with a port field
             and connect / disconnect: a Blender the user already has open
             joins a wobble node the same way a launched one does.

Blender is the SERVER (it binds the port wobble chose), wobble the client.
One flat JSON object per line each way; pixels never travel on the socket.

  wobble -> blender   {"cmd": "frame", "frame": 1001}
                      {"cmd": "surfaces", "ok": true}   wobble imported the shared surfaces (or not)
                      {"cmd": "sack", "slot": s, "seq": q}  done with surface slot s
                      {"cmd": "spec", "w": W, "h": H, "scene": "", "camera": "",
                       "engine": "file|eevee|cycles", "quality": "viewport|final",
                       "shading": "|MATERIAL|RENDERED", "samples": 0,
                       "transparent": true, "inputs": 0}
                      {"cmd": "shm", "name": n, "bytes": b}   the ring wobble's textures
                          come in (POSIX shared memory; or {"path": p}, a file)
                      {"cmd": "tex", "port": 0, "slot": s, "seq": q}
                      {"cmd": "view", "lx": .., "ly": .., "lz": .., "rw": .., "rx": .., "ry": .., "rz": ..,
                       "dist": d, "lens": mm, "ortho": false, "moving": true}
                          the eye (pivot, view->world quaternion, distance): the 3D
                          view in free mode, the camera in camera mode; `moving`
                          = a drag in progress, draw at one sample
                      {"cmd": "scene3d", "path": p}   write the cameras, planes and empties
                          over the frame range to p (wobble's 3D cards JSON)
                      {"cmd": "ping"}
                      {"cmd": "quit"}

  blender -> wobble   {"ev": "hello", "version": ..., "w": W, "h": H, "fps": F,
                       "blend": path, "frame": f, "start": a, "end": b}
                      {"ev": "shm", "name": n, "bytes": b}   the ring frames go back in
                          (or {"path": p} where only a file mapping exists)
                      {"ev": "surfaces", "ids": "1,2,3", "w": W, "h": H}   shared GPU surfaces (native module)
                      {"ev": "frame", "slot": s, "seq": q}                 a frame in the ring
                      {"ev": "frame", "slot": s, "seq": q, "frame": f, "w": .., "h": .., "pass": p,
                       "flags": .., "scene": .., "ms": .., "samples": ..}   a frame in surface s
                      {"ev": "frame_changed", "frame": f}
                      {"ev": "view", ...same fields...}   where the eye is now
                      {"ev": "scene", "scene": n}   the scene changed (an edit)
                      {"ev": "saved", "path": path}
                      {"ev": "scene3d", "path": p}   the scene3d file is written
                      {"ev": "error", "msg": text}
                      {"ev": "pong"}

The pixel ring is a file both processes map (layout in
`engine/src/blender/shm.rs`, mirrored by `RingWriter` below). A frame is
written into a free slot under a seqlock and then ANNOUNCED on the socket:
the announcement leaves after the write completes, through the kernel, so
the reader always sees the finished pixels. wobble ACKS a slot when it is
done with it ({"cmd": "ack", "slot": s, "seq": q}) — on its GPU path the
ring's pages are read by the GPU, so a slot stays held until that blit has
run; a slot never acked is never reused, and with every slot held a frame
is skipped rather than torn. Pixels are float32 rows padded to 256 bytes,
bottom-up as the GPU reads them back: no conversion, no flip, one memcpy.

Everything that touches `bpy` runs on Blender's main thread: the socket
threads only move lines in and out of a queue, and a `bpy.app.timers`
callback drains it. Every scene assignment is guarded, because property
names and engine ids have moved between Blender releases and one missing
attribute must not take the whole session down.
"""

bl_info = {
    "name": "wobble bridge",
    "author": "wobble",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > wobble",
    "description": "Live link to the wobble compositor: it drives the playhead, the scene camera's picture goes back, its plates come in as images",
    "category": "System",
}

import json
import math
import mmap
import os
import queue
import socket
import struct
import sys
import threading
import time

try:
    from multiprocessing import shared_memory as _shm
except ImportError:  # pragma: no cover
    _shm = None
# POSIX shared memory where it exists: pages with no file behind them, so
# nothing is ever written back to a disk. Windows keeps a temp file
# mapping (page-cache backed, written only under memory pressure).
USE_POSIX_SHM = _shm is not None and sys.platform != "win32"

# The compiled half (`wobble_native`, built by wobble and written beside
# this file): hands the offscreen's texture to wobble as a shared surface,
# no readback. Optional — without it the shared-memory ring serves.
NATIVE = None
try:
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    import wobble_native as _native

    if _native.available():
        NATIVE = _native
except Exception as _e:  # noqa: BLE001
    print("wobble | native module not loaded: %s" % _e, flush=True)

import bpy
from mathutils import Matrix, Quaternion, Vector

try:
    import numpy as np
except ImportError:  # Blender ships numpy; a stripped build is a loud error
    np = None

PORT = int(os.environ.get("WOBBLE_PORT", "0") or 0)
SHM_DIR = os.environ.get("WOBBLE_SHM_DIR", "")

# Blender's scene-linear space, by the names successive releases gave it.
LINEAR_NAMES = ("Linear Rec.709", "Linear", "Linear CIE-XYZ I-E")

# A scene edit redraws the current frame once the hand pauses for this
# long — a camera drag fires dozens of updates a second.
DIRTY_DEBOUNCE = 0.04
# ...but never wait longer than this during a long drag, so wobble keeps
# updating while the camera is moving.
DIRTY_MAX_WAIT = 0.25
# Viewport draws are published at most this often while a frame settles.
DRAW_PERIOD = 1.0 / 40.0
# Cards sit this far in front of the camera, one unit further per port.
CARD_DIST = 10.0
MAX_CARDS = 8


def log(msg):
    print("wobble | " + str(msg), flush=True)


def guarded(label, fn):
    """Run a scene assignment; log and carry on if this Blender lacks it."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        log("%s: %s" % (label, e))
        return None


# --------------------------------------------------------------------------
# The pixel ring (mirror of engine/src/blender/shm.rs)

MAGIC = b"WOBSHM02"
VERSION = 2
PAGE = 16384
HEADER_BYTES = PAGE
DESC_OFFSET = 64
DESC_BYTES = 64
ROW_ALIGN = 256
FMT_RGBA16F = 1
FMT_RGBA32F = 2
BPP = {FMT_RGBA16F: 8, FMT_RGBA32F: 16}
FLAG_FINAL = 1
FLAG_SETTLED = 2
DESC = struct.Struct("<QqIIIIQfIII")  # seq frame w h fmt flags scene ms samples stride pass
RING_SLOTS = 8


def stride_for(w, fmt):
    return -(-(w * BPP[fmt]) // ROW_ALIGN) * ROW_ALIGN


class RingWriter:
    """The ring frames go BACK in. Blender owns it: created here, recreated
    when the frame grows, announced to wobble as `shm`. Slots are whole
    pages (a GPU can map them), rows are 256-byte aligned (a GPU can blit
    them), and a slot wobble has not acked is not written again."""

    def __init__(self, name, path, slots, slot_bytes):
        self.slots = slots
        self.slot_bytes = -(-slot_bytes // PAGE) * PAGE
        size = HEADER_BYTES + slots * self.slot_bytes
        self.size = size
        self.f = None
        self.shm = None
        if USE_POSIX_SHM:
            # A stale object under this name (a crash last time) would
            # keep its old size: clear it first.
            try:
                old = _shm.SharedMemory(name=name, create=False)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
            self.shm = _shm.SharedMemory(name=name, create=True, size=size)
            self.mm = self.shm.buf
            self.announce = {"name": self.shm.name.lstrip("/"), "bytes": size}
        else:
            with open(path, "wb") as f:
                f.truncate(size)
            self.f = open(path, "r+b")
            self.mm = mmap.mmap(self.f.fileno(), size)
            self.announce = {"path": path}
        self.mm[0:8] = MAGIC
        struct.pack_into("<IIQ", self.mm, 8, VERSION, slots, self.slot_bytes)
        self.next = 0
        self.seq = 0
        # slot -> seq wobble has not acked yet
        self.held = {}

    def fits(self, nbytes):
        return nbytes <= self.slot_bytes

    def ack(self, slot, seq):
        if self.held.get(slot) == seq:
            del self.held[slot]

    def free_slot(self):
        for _ in range(self.slots):
            slot = self.next
            self.next = (self.next + 1) % self.slots
            if slot not in self.held:
                return slot
        return None

    def begin(self, w, h, fmt, frame, flags, scene, ms, samples, pass_id=0):
        """Claim a slot and return (slot, seq, ndarray view of its rows,
        padded) for the caller to write into; `None` with every slot held.
        `end` publishes it."""
        stride = stride_for(w, fmt)
        if stride * h > self.slot_bytes:
            return None
        slot = self.free_slot()
        if slot is None:
            return None
        self.seq += 2
        seq = self.seq
        d = DESC_OFFSET + slot * DESC_BYTES
        struct.pack_into("<Q", self.mm, d, seq - 1)  # odd: writing
        DESC.pack_into(self.mm, d, seq - 1, frame, w, h, fmt, flags, scene, ms, samples, stride, pass_id)
        at = HEADER_BYTES + slot * self.slot_bytes
        dtype = np.float16 if fmt == FMT_RGBA16F else np.float32
        view = np.ndarray((h, stride // np.dtype(dtype).itemsize), dtype=dtype, buffer=self.mm, offset=at)
        return slot, seq, view

    def end(self, slot, seq):
        d = DESC_OFFSET + slot * DESC_BYTES
        struct.pack_into("<Q", self.mm, d, seq)
        self.held[slot] = seq
        return slot, seq

    def close(self):
        try:
            if self.shm is not None:
                self.mm.release() if hasattr(self.mm, "release") else None
                self.shm.close()
                self.shm.unlink()
            else:
                self.mm.close()
                self.f.close()
                os.remove(self.announce["path"])
        except Exception:  # noqa: BLE001
            pass


class RingReader:
    """The ring wobble's textures come IN on. wobble owns it."""

    def __init__(self, msg):
        self.f = None
        self.shm = None
        name = msg.get("name") or ""
        if name:
            if _shm is None:
                raise ValueError("no shared_memory module in this Python")
            self.shm = _shm.SharedMemory(name=name, create=False)
            self.mm = self.shm.buf
        else:
            path = msg.get("path") or ""
            self.f = open(path, "rb")
            self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        if bytes(self.mm[0:8]) != MAGIC:
            raise ValueError("%r is not a wobble pixel ring" % (msg,))
        version, self.slots, self.slot_bytes = struct.unpack_from("<IIQ", self.mm, 8)
        if version != VERSION:
            raise ValueError("pixel ring version %d, this bridge speaks %d" % (version, VERSION))

    def read(self, slot, seq):
        """(w, h, float32 array (h, w*4)) or None when lapped."""
        if slot >= self.slots:
            return None
        d = DESC_OFFSET + slot * DESC_BYTES
        got = DESC.unpack_from(self.mm, d)
        if got[0] != seq or seq & 1:
            return None
        w, h, fmt, stride = got[2], got[3], got[4], got[9]
        if fmt not in BPP or w == 0 or h == 0 or stride * h > self.slot_bytes:
            return None
        at = HEADER_BYTES + slot * self.slot_bytes
        dtype = np.float16 if fmt == FMT_RGBA16F else np.float32
        view = np.ndarray((h, stride // np.dtype(dtype).itemsize), dtype=dtype, buffer=self.mm, offset=at)
        px = view[:, : w * 4].astype(np.float32)
        if struct.unpack_from("<Q", self.mm, d)[0] != seq:
            return None
        return w, h, px

    def close(self):
        try:
            if self.shm is not None:
                self.mm.release() if hasattr(self.mm, "release") else None
                self.shm.close()
            else:
                self.mm.close()
                self.f.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# Socket


class Bridge:
    """One client at a time. Socket work stays off the main thread; `q`
    carries decoded commands to the timer, `send` is safe from anywhere."""

    def __init__(self, port):
        self.q = queue.Queue()
        self.conn = None
        self.lock = threading.Lock()
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", port))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()
        log("listening on 127.0.0.1:%d" % self.port)

    def _accept(self):
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.lock:
                old, self.conn = self.conn, conn
            if old is not None:
                try:
                    old.close()
                except OSError:
                    pass
            self.q.put({"cmd": "_connected"})
            threading.Thread(target=self._read, args=(conn,), daemon=True).start()

    def _read(self, conn):
        buf = b""
        while True:
            try:
                chunk = conn.recv(65536)
            except OSError:
                chunk = b""
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    self.q.put(json.loads(line.decode("utf-8")))
                except ValueError:
                    log("bad line from wobble: %r" % line[:80])
        with self.lock:
            if self.conn is conn:
                self.conn = None
        self.q.put({"cmd": "_disconnected"})

    def send(self, obj):
        data = (json.dumps(obj) + "\n").encode("utf-8")
        with self.lock:
            conn = self.conn
        if conn is None:
            return
        try:
            conn.sendall(data)
        except OSError as e:
            log("send failed: %s" % e)

    def close(self):
        with self.lock:
            conn, self.conn = self.conn, None
        for s in (conn, self.listener):
            try:
                if s is not None:
                    s.close()
            except OSError:
                pass


# --------------------------------------------------------------------------
# Scene


def set_engine(sc, engine):
    """Switch the render engine, by whichever id this Blender has."""
    want = {
        "eevee": ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"),
        "cycles": ("CYCLES",),
    }.get(engine)
    if not want:
        return
    for ident in want:
        try:
            sc.render.engine = ident
            return
        except TypeError:
            continue
    log("no engine among %r" % (want,))


def linear_colorspace(img):
    for name in LINEAR_NAMES:
        try:
            img.colorspace_settings.name = name
            return
        except TypeError:
            continue


PIXEL_TYPES = tuple(
    t
    for t in (
        getattr(bpy.types, n, None)
        for n in ("Camera", "Light", "Material", "World", "NodeTree", "Object", "Mesh")
    )
    if t is not None
)


def _changes_pixels(u):
    """Does one depsgraph update move pixels? Transforms, geometry and
    shading do; so does anything to a camera, light, material or world,
    whose flags do not always say so. A selection change does not — and
    neither does an Image datablock: our own texture updates and the render
    result are Images, and redrawing on those would never stop."""
    if isinstance(u.id, getattr(bpy.types, "Image", ())):
        return False
    if u.is_updated_transform or u.is_updated_geometry or u.is_updated_shading:
        return True
    return isinstance(u.id, PIXEL_TYPES)


def view_matrix(loc, rot, dist, ortho):
    """Blender's own: eye `dist` back along the view axis from the pivot —
    except an ortho view, whose matrix sits AT the pivot (verified against
    `RegionView3D.view_matrix` on 5.2.1)."""
    r = Quaternion(rot).to_matrix().to_4x4()
    m = r.inverted() @ Matrix.Translation(-Vector(loc))
    if not ortho:
        m = Matrix.Translation((0.0, 0.0, -dist)) @ m
    return m


def projection(w, h, lens, ortho, dist, clip_start, clip_end):
    """The 3D view's projection for a `w` x `h` draw: a 36mm sensor fitted
    to the larger side, doubled (the viewport's perspective zoom) — exactly
    `RegionView3D.window_matrix` (verified on 5.2.1, PERSP and ORTHO)."""
    vf = float(max(w, h, 1))
    if ortho:
        xmax = (w / vf) * dist * 36.0 / max(lens, 1.0)
        ymax = (h / vf) * dist * 36.0 / max(lens, 1.0)
        f = clip_end
        m = Matrix.Identity(4)
        m[0][0] = 1.0 / xmax
        m[1][1] = 1.0 / ymax
        m[2][2] = -2.0 / (f + f)
        m[2][3] = 0.0
        return m
    tx = (w / vf) * 36.0 / max(lens, 1.0)
    ty = (h / vf) * 36.0 / max(lens, 1.0)
    n, f = max(clip_start, 1e-4), max(clip_end, clip_start + 1e-3)
    m = Matrix.Identity(4)
    m[0][0] = 1.0 / tx
    m[1][1] = 1.0 / ty
    m[2][2] = -(f + n) / (f - n)
    m[2][3] = -2.0 * f * n / (f - n)
    m[3][2] = -1.0
    m[3][3] = 0.0
    return m


def card_material(name, img):
    """Emission through the plate's own alpha: a card shows the comp's
    pixels as they are, unlit, and is transparent where the plate is."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    tex = next((n for n in nt.nodes if n.type == "TEX_IMAGE" and n.name == "wobble_tex"), None)
    if tex is None:
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        out.location = (600, 0)
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.location = (400, 0)
        transp = nt.nodes.new("ShaderNodeBsdfTransparent")
        transp.location = (200, 120)
        emit = nt.nodes.new("ShaderNodeEmission")
        emit.location = (200, -80)
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.name = "wobble_tex"
        tex.location = (-100, 0)
        nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
        nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
        nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
        nt.links.new(emit.outputs["Emission"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    tex.image = img
    # EEVEE Next (4.2+) names it surface_render_method; older builds
    # blend_method. Either way: alpha blended, not clipped.
    guarded("surface_render_method", lambda: setattr(mat, "surface_render_method", "BLENDED"))
    guarded("blend_method", lambda: setattr(mat, "blend_method", "BLEND"))
    return mat


def frustum_width(cam_obj, dist):
    cam = cam_obj.data
    if getattr(cam, "type", "PERSP") == "ORTHO":
        return float(cam.ortho_scale)
    angle = float(getattr(cam, "angle", 0.8726))
    return 2.0 * dist * math.tan(angle / 2.0)


def is_fork():
    """A Blender built from the wobble fork: its Cycles can render the
    viewport straight into wobble. Stock Blender has no such hook."""
    try:
        import _cycles
    except ImportError:
        return False
    return hasattr(_cycles, "wobble_target")


def find_view3d():
    """A window, a 3D viewport area, its window region and its space: what
    an offscreen viewport draw needs for context."""
    wm = bpy.context.window_manager
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region is None:
                continue
            return win, area, region, area.spaces.active
    return None


class Live:
    def __init__(self, port):
        self.bridge = Bridge(port)
        self.spec = {}
        self.scene_name = ""
        self.size = (0, 0)
        # The frame wobble last asked for (pending = not yet applied).
        self.pending = None
        self.current = None
        # A monotonic scene version: bumped by every edit that moves
        # pixels, stamped onto every frame, so wobble can tell a stale
        # frame from a fresh one.
        self.scene_version = 1
        self.dirty_at = None
        self.dirty_since = None
        # Whether the frame on the ring is the current one, fully drawn;
        # `samples` is what the last draw used, for the panel.
        self.samples = 0
        self.settled = False
        self.last_draw = 0.0
        self.offscreen = None
        self.ring = None
        self.ring_n = 0
        self.tex_ring = None
        # A final render in flight (INVOKE_DEFAULT keeps the UI alive).
        self.rendering = None
        self.busy = False
        self.connected = False
        # The eye as last reported (free mode: the 3D view's fields; camera
        # mode: the camera), so a change in Blender is noticed and sent.
        self.last_view = None
        self.cam_dist = CARD_DIST
        # A wobble drag is in progress: one-sample draws until it ends.
        self.moving = False
        self.saved_samples = None
        self.pending_view = None
        # (blender pass identifier, wobble pass id) per extra output.
        self.passes = []
        # Shared GPU surfaces (native module): the ring, its held slots
        # awaiting wobble's `sack`, wobble's verdict on importing them.
        self.surf = None
        self.surf_held = {}
        self.surf_seq = 0
        self.surf_next = 0
        self.surf_ok = None
        guarded(
            "render_display_type",
            lambda: setattr(bpy.context.preferences.view, "render_display_type", "NONE"),
        )
        bpy.app.handlers.depsgraph_update_post.append(self._on_update)
        bpy.app.handlers.frame_change_post.append(self._on_frame)
        bpy.app.handlers.save_post.append(self._on_save)
        bpy.app.handlers.render_complete.append(self._on_render_done)
        bpy.app.handlers.render_cancel.append(self._on_render_cancel)
        bpy.app.timers.register(self.tick, first_interval=0.1, persistent=True)

    # -- scene ------------------------------------------------------------

    def scene(self):
        sc = bpy.data.scenes.get(self.scene_name) if self.scene_name else None
        return sc or bpy.context.scene

    def camera(self, sc):
        name = self.spec.get("camera") or ""
        if name:
            obj = bpy.data.objects.get(name)
            if obj is not None and obj.type == "CAMERA":
                return obj
        return sc.camera

    def apply_spec(self, spec):
        """Everything the comp needs from the scene, applied — never
        trusted to the file. Idempotent: wobble sends it whenever any of it
        changes."""
        self.spec = spec
        self.busy = True
        try:
            name = spec.get("scene") or ""
            self.scene_name = name if name and name in bpy.data.scenes else ""
            sc = self.scene()
            w, h = int(spec.get("w") or 0), int(spec.get("h") or 0)
            if w > 0 and h > 0:
                self.size = (w, h)
                guarded("resolution", lambda: setattr(sc.render, "resolution_x", w))
                guarded("resolution", lambda: setattr(sc.render, "resolution_y", h))
                guarded("resolution", lambda: setattr(sc.render, "resolution_percentage", 100))
            set_engine(sc, spec.get("engine") or "file")
            guarded("film_transparent", lambda: setattr(sc.render, "film_transparent", bool(spec.get("transparent", True))))
            # A final render per frame: keep Cycles' BVH and shaders between
            # frames instead of rebuilding them each time.
            if spec.get("quality") == "final":
                guarded("persistent_data", lambda: setattr(sc.render, "use_persistent_data", True))
            samples = int(spec.get("samples") or 0)
            if samples > 0:
                if spec.get("quality") == "final":
                    guarded("samples", lambda: setattr(sc.cycles, "samples", samples))
                    guarded("samples", lambda: setattr(sc.eevee, "taa_render_samples", samples))
                else:
                    # An offscreen draw renders every viewport sample at
                    # once (measured: converged from the first call), so
                    # the knob is the viewport's sample count.
                    guarded("samples", lambda: setattr(sc.eevee, "taa_samples", samples))
            self.apply_shading(sc)
            if spec.get("quality") == "final":
                self.ensure_viewer(sc)
            n_in = int(spec.get("inputs") or 0)
            for i in range(n_in):
                self.ensure_input_image(i, 4, 4)
            self.ensure_cards(sc, n_in if spec.get("cards") else 0)
            self.passes = []
            for item in (spec.get("passes") or "").split(","):
                if "=" in item:
                    name, pid = item.split("=", 1)
                    try:
                        self.passes.append((name.strip(), int(pid)))
                    except ValueError:
                        pass
            self.enable_passes(bpy.context.view_layer)
        finally:
            self.busy = False
        # A spec change is a scene change as far as the pixels go.
        self.invalidate()

    def apply_shading(self, sc):
        """The 3D view's shading for the engine the scene is on NOW. Run
        on a spec and again whenever the engine changes under us (the user
        picking another engine in Blender), so the picture follows it."""
        shading = self.spec.get("shading") or ""
        live = self.live_cycles(sc)
        self.engine_seen = (sc.render.engine, live)
        # Cycles has no offscreen viewport render: a draw with RENDERED
        # shading comes back black. Material preview (Eevee) is what
        # Blender shows Cycles users while they work, so it is what the
        # live picture is; `final` gives the Cycles pixels.
        if live:
            # A wobble-fork Blender with wobble's buffer bound: the
            # viewport's OWN Cycles session renders into wobble.
            self.setup_live_cycles(sc)
            return
        self.leave_live_cycles(sc)
        if (
            sc.render.engine == "CYCLES"
            and self.spec.get("quality") != "final"
            and not self.cycles_stock_sent
            and not is_fork()
        ):
            # Once per session: wobble explains it in a popup.
            self.cycles_stock_sent = True
            self.bridge.send({"ev": "cycles_stock"})
        if shading == "RENDERED" and sc.render.engine == "CYCLES":
            shading = "MATERIAL"
            if not self.cycles_noted:
                self.cycles_noted = True
                log("cycles: the viewport picture is the material preview; pick final for Cycles pixels")
        if shading:
            v = find_view3d()
            if v is not None:
                guarded("shading", lambda: setattr(v[3].shading, "type", shading))

    def watch_engine(self):
        """An engine switch in Blender is not a depsgraph update that moves
        pixels (it is the Scene, not what is in it), so nothing noticed
        one: leaving live Cycles for EEVEE left the picture settled on the
        last Cycles sample, wobble frozen on it while Blender's own
        viewport came back. Follow it here, both ways: re-shade for the
        new engine and redraw."""
        sc = self.scene()
        now = (sc.render.engine, self.live_cycles(sc))
        if now == self.engine_seen:
            return
        was = self.engine_seen
        self.busy = True
        try:
            self.apply_shading(sc)
        finally:
            self.busy = False
        log("engine %s -> %s%s" % (was[0] if was else "?", now[0], " (live cycles)" if now[1] else ""))
        self.invalidate()

    # Which view-layer flag a viewport pass needs, so it is not black.
    PASS_FLAGS = {
        "MIST": ("use_pass_mist",),
        "NORMAL": ("use_pass_normal",),
        "POSITION": ("use_pass_position",),
        "EMISSION": ("use_pass_emit",),
        "ENVIRONMENT": ("use_pass_environment",),
        "AO": ("use_pass_ambient_occlusion",),
        "SHADOW": ("use_pass_shadow",),
        "DIFFUSE_LIGHT": ("use_pass_diffuse_direct", "use_pass_diffuse_indirect"),
        "DIFFUSE_COLOR": ("use_pass_diffuse_color",),
        "SPECULAR_LIGHT": ("use_pass_glossy_direct", "use_pass_glossy_indirect"),
        "SPECULAR_COLOR": ("use_pass_glossy_color",),
    }

    def enable_passes(self, vl):
        for name, _ in self.passes:
            for flag in self.PASS_FLAGS.get(name, ()):
                guarded(flag, lambda f=flag: setattr(vl, f, True))
            if name == "VOLUME_LIGHT" and hasattr(vl, "eevee"):
                guarded("volume pass", lambda: setattr(vl.eevee, "use_pass_volume_direct", True))
            if name == "TRANSPARENT" and hasattr(vl, "eevee"):
                guarded("transparent pass", lambda: setattr(vl.eevee, "use_pass_transparent", True))

    def ensure_cards(self, sc, n):
        """A card per input port in front of the camera, fitted to the
        frame at CARD_DIST (+1 per port, so a stack of ports reads as a
        stack of layers), parented so it follows. `n` = 0 removes them."""
        cam = self.camera(sc)
        w, h = self.size if self.size != (0, 0) else (sc.render.resolution_x, sc.render.resolution_y)
        aspect = (float(w) / float(h)) if (w > 0 and h > 0) else 1.0
        for i in range(MAX_CARDS):
            name = "wobble_card_%d" % i
            obj = bpy.data.objects.get(name)
            if i >= n or cam is None:
                if obj is not None:
                    bpy.data.objects.remove(obj, do_unlink=True)
                continue
            img = self.ensure_input_image(i, 4, 4)
            mat = card_material(name, img)
            if obj is None:
                mesh = bpy.data.meshes.new(name)
                verts = [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)]
                mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
                uv = mesh.uv_layers.new(name="UVMap")
                for loop, coord in zip(mesh.loops, [(0, 0), (1, 0), (1, 1), (0, 1)]):
                    uv.data[loop.index].uv = coord
                mesh.update()
                obj = bpy.data.objects.new(name, mesh)
                sc.collection.objects.link(obj)
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)
            dist = CARD_DIST + float(i)
            width = frustum_width(cam, dist)
            # Landscape frames fit the width; portrait ones fit the height,
            # as Blender's AUTO sensor fit does.
            if h > w:
                width = width * aspect
            obj.parent = cam
            obj.matrix_parent_inverse = Matrix.Identity(4)
            obj.location = (0.0, 0.0, -dist)
            obj.rotation_euler = (0.0, 0.0, 0.0)
            obj.scale = (width, width / aspect, 1.0)
            guarded("card shadow", lambda: setattr(obj, "visible_shadow", False))

    def ensure_viewer(self, sc):
        """A Viewer node on the comp's output, so a final render can be read
        back (Render Result cannot). The user's compositor is respected: the
        viewer taps whatever feeds their output node, or the render layers
        if nothing does."""
        guarded("use_compositing", lambda: setattr(sc.render, "use_compositing", True))
        tree = None
        if hasattr(sc, "compositing_node_group"):
            tree = sc.compositing_node_group
            if tree is None:
                tree = bpy.data.node_groups.new("wobble comp", "CompositorNodeTree")
                try:
                    tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
                except Exception as e:  # noqa: BLE001
                    log("comp output socket: %s" % e)
                out = tree.nodes.new("NodeGroupOutput")
                rl = tree.nodes.new("CompositorNodeRLayers")
                tree.links.new(rl.outputs[0], out.inputs[0])
                sc.compositing_node_group = tree
        else:
            guarded("use_nodes", lambda: setattr(sc, "use_nodes", True))
            tree = sc.node_tree
        if tree is None:
            return
        viewer = next((n for n in tree.nodes if n.type == "VIEWER" and n.name == "wobble"), None)
        if viewer is None:
            viewer = tree.nodes.new("CompositorNodeViewer")
            viewer.name = "wobble"
            viewer.label = "wobble"
        # What feeds the output: the group output's / composite node's
        # first linked input, else a render layers node.
        src = None
        for n in tree.nodes:
            if n.type in ("GROUP_OUTPUT", "COMPOSITE") and n.inputs and n.inputs[0].is_linked:
                src = n.inputs[0].links[0].from_socket
                break
        if src is None:
            rl = next((n for n in tree.nodes if n.type == "R_LAYERS"), None)
            if rl is None:
                rl = tree.nodes.new("CompositorNodeRLayers")
            src = rl.outputs[0]
        if not viewer.inputs[0].is_linked or viewer.inputs[0].links[0].from_socket != src:
            for l in list(viewer.inputs[0].links):
                tree.links.remove(l)
            tree.links.new(src, viewer.inputs[0])

    def ensure_input_image(self, port, w, h):
        name = "wobble_in_%d" % port
        img = bpy.data.images.get(name)
        if img is None or tuple(img.size) != (w, h) or not img.is_float:
            if img is not None:
                img.scale(w, h)
            else:
                img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=True)
                img.use_fake_user = True
                linear_colorspace(img)
        return img

    # -- handlers ---------------------------------------------------------

    def _on_update(self, scene, depsgraph=None):
        if self.busy:
            return
        if depsgraph is not None and not any(_changes_pixels(u) for u in depsgraph.updates):
            return
        now = time.time()
        if self.dirty_since is None:
            self.dirty_since = now
        self.dirty_at = now

    def _on_frame(self, scene, depsgraph=None):
        # Our own frame_set fires this too; only the user's moves go out.
        if self.busy:
            return
        self.bridge.send({"ev": "frame_changed", "frame": int(scene.frame_current)})

    def _on_save(self, *_):
        self.bridge.send({"ev": "saved", "path": bpy.data.filepath})

    def _on_render_done(self, *_):
        if self.rendering is None:
            return
        frame, t0 = self.rendering
        self.rendering = None
        img = bpy.data.images.get("Viewer Node")
        if img is None or img.size[0] == 0:
            self.bridge.send({"ev": "error", "msg": "final render: no viewer node picture"})
            return
        w, h = img.size
        a = np.empty(w * h * 4, np.float32)
        try:
            img.pixels.foreach_get(a)
        except Exception as e:  # noqa: BLE001
            self.bridge.send({"ev": "error", "msg": "final render: %s" % e})
            return
        ms = (time.time() - t0) * 1000.0
        self.publish(a.reshape(h, w * 4), frame, w, h, FLAG_FINAL | FLAG_SETTLED, ms, 1)
        self.settled = True

    def _on_render_cancel(self, *_):
        self.rendering = None

    # -- outbound ---------------------------------------------------------

    def hello(self):
        sc = self.scene()
        w, h = self.size if self.size != (0, 0) else (sc.render.resolution_x, sc.render.resolution_y)
        self.bridge.send(
            {
                "ev": "hello",
                "version": bpy.app.version_string,
                "w": int(w),
                "h": int(h),
                "fps": sc.render.fps / (sc.render.fps_base or 1.0),
                "blend": bpy.data.filepath,
                "frame": int(sc.frame_current),
                "start": int(sc.frame_start),
                "end": int(sc.frame_end),
            }
        )
        self.send_view(force=True)
        self.announce_cycles_fork()

    # -- live Cycles (wobble-fork builds only) ----------------------------

    def announce_cycles_fork(self):
        """A Blender built from the wobble fork can render Cycles straight
        into a buffer wobble owns: say so, with the pid wobble duplicates
        the buffer's handle into."""
        try:
            import _cycles
        except ImportError:
            return
        if hasattr(_cycles, "wobble_target"):
            self.bridge.send({"ev": "cycles_fork", "pid": os.getpid()})

    cycles_bound = False

    def live_cycles(self, sc):
        """Live Cycles: a fork with wobble's buffer bound, Cycles as the
        engine and the viewport (not final) picture asked for."""
        return (
            self.cycles_bound
            and sc.render.engine == "CYCLES"
            and self.spec.get("quality") != "final"
        )

    def setup_live_cycles(self, sc):
        """The 3D view renders the scene camera's frame with Cycles: camera
        view, the render border on the whole frame (so Cycles renders the
        frame and nothing around it), RENDERED shading, and the frame
        fitted to the region. Its display updates land in wobble."""
        v = find_view3d()
        if v is None:
            self.bridge.send({"ev": "error", "msg": "live cycles needs a 3D viewport"})
            return
        win, area, region, space = v
        if self.border_saved is None:
            r = sc.render
            self.border_saved = (r.use_border, r.border_min_x, r.border_min_y, r.border_max_x, r.border_max_y)
            self.lock_saved = space.lock_camera
        # The live picture IS this view, so in camera mode it must stay the
        # camera: a tumble in Blender moves the camera (Lock Camera to View)
        # rather than leaving camera view, which used to strand wobble's
        # orbit on a camera nobody was looking through.
        if not self.free_mode():
            guarded("live_cycles", lambda: setattr(space, "lock_camera", True))
        guarded("live_cycles", lambda: setattr(sc.render, "use_border", True))
        guarded("live_cycles", lambda: setattr(sc.render, "border_min_x", 0.0))
        guarded("live_cycles", lambda: setattr(sc.render, "border_min_y", 0.0))
        guarded("live_cycles", lambda: setattr(sc.render, "border_max_x", 1.0))
        guarded("live_cycles", lambda: setattr(sc.render, "border_max_y", 1.0))
        guarded("live_cycles", lambda: setattr(space.shading, "type", "RENDERED"))
        rv = region.data
        if rv is not None and rv.view_perspective != "CAMERA":
            guarded("live_cycles", lambda: setattr(rv, "view_perspective", "CAMERA"))
        try:
            with bpy.context.temp_override(window=win, area=area, region=region):
                bpy.ops.view3d.view_center_camera()
        except Exception as e:  # noqa: BLE001
            log("live cycles: could not fit the camera: %s" % e)
        area.tag_redraw()
        log("live cycles: the viewport renders into wobble")

    border_saved = None
    lock_saved = False
    engine_seen = None

    def leave_live_cycles(self, sc):
        """Out of live Cycles: give the scene its own render border back
        (the offscreen draw and an F12 must not inherit ours)."""
        if self.border_saved is None:
            return
        b, self.border_saved = self.border_saved, None
        for k, val in zip(("use_border", "border_min_x", "border_min_y", "border_max_x", "border_max_y"), b):
            guarded("live_cycles", lambda k=k, val=val: setattr(sc.render, k, val))
        v = find_view3d()
        if v is not None:
            guarded("live_cycles", lambda: setattr(v[3], "lock_camera", self.lock_saved))

    def bind_cycles(self, msg):
        """Point Cycles at wobble's buffer (or detach it: handle 0), and put
        Cycles on the GPU backend that writes it: CUDA on Windows (it
        imports the Vulkan buffer), Metal on macOS (it wraps the shared
        pages)."""
        try:
            import _cycles

            handle = int(msg.get("handle", 0))
            if handle:
                backend = "METAL" if sys.platform == "darwin" else "CUDA"
                prefs = bpy.context.preferences.addons["cycles"].preferences
                if prefs.compute_device_type != backend:
                    prefs.compute_device_type = backend
                prefs.get_devices()
                for d in prefs.devices:
                    d.use = d.type == backend
                self.scene().cycles.device = "GPU"
            _cycles.wobble_target(
                handle,
                int(msg.get("size", 0)),
                bytes.fromhex(msg.get("uuid", "") or ""),
                str(msg.get("stamp", "")),
            )
            self.cycles_bound = bool(handle)
            self.apply_shading(self.scene())
            self.bridge.send({"ev": "cycles_bound", "ok": True, "msg": ""})
            log("cycles renders into wobble's buffer" if handle else "cycles detached from wobble")
        except Exception as e:  # noqa: BLE001
            self.bridge.send({"ev": "cycles_bound", "ok": False, "msg": str(e)})

    # -- the eye ----------------------------------------------------------

    def free_mode(self):
        return self.spec.get("view") == "free"

    def current_view(self):
        """(loc, rot, dist, lens, ortho) of whoever's eye the spec names."""
        sc = self.scene()
        if self.free_mode():
            v = find_view3d()
            if v is None:
                return None
            rv, space = v[2].data, v[3]
            if rv is None:
                return None
            return (
                tuple(rv.view_location),
                tuple(rv.view_rotation),
                float(rv.view_distance),
                float(space.lens),
                rv.view_perspective == "ORTHO",
            )
        cam = self.camera(sc)
        if cam is None:
            return None
        rot = cam.matrix_world.to_quaternion()
        eye = cam.matrix_world.translation
        loc = eye - (rot @ Vector((0.0, 0.0, self.cam_dist)))
        return (
            tuple(loc),
            tuple(rot),
            float(self.cam_dist),
            float(getattr(cam.data, "lens", 50.0)),
            getattr(cam.data, "type", "PERSP") == "ORTHO",
        )

    @staticmethod
    def _rounded(v):
        return None if v is None else tuple(
            tuple(round(x, 5) for x in part) if isinstance(part, tuple) else (round(part, 5) if isinstance(part, float) else part)
            for part in v
        )

    def send_view(self, force=False):
        v = self.current_view()
        r = self._rounded(v)
        if v is None or (not force and r == self.last_view):
            return False
        self.last_view = r
        loc, rot, dist, lens, ortho = v
        self.bridge.send(
            {
                "ev": "view",
                "lx": loc[0], "ly": loc[1], "lz": loc[2],
                "rw": rot[0], "rx": rot[1], "ry": rot[2], "rz": rot[3],
                "dist": dist, "lens": lens, "ortho": bool(ortho),
            }
        )
        return True

    def set_moving(self, moving):
        """One sample while wobble's hand is on the view, the scene's own
        count when it stops — what Blender's viewport does for itself."""
        if moving == self.moving:
            return
        sc = self.scene()
        self.moving = moving
        if moving:
            self.saved_samples = getattr(sc.eevee, "taa_samples", None)
            guarded("taa_samples", lambda: setattr(sc.eevee, "taa_samples", 1))
        elif self.saved_samples is not None:
            want = int(self.spec.get("samples") or 0) or self.saved_samples
            guarded("taa_samples", lambda: setattr(sc.eevee, "taa_samples", want))

    def apply_view(self, msg):
        """wobble moved the eye: the 3D view's fields in free mode, the
        camera's transform in camera mode (keyed too when Blender's
        autokey is on and the drag has ended — an orbit becomes a camera
        move)."""
        try:
            loc = (float(msg["lx"]), float(msg["ly"]), float(msg["lz"]))
            rot = (float(msg["rw"]), float(msg["rx"]), float(msg["ry"]), float(msg["rz"]))
            dist = float(msg["dist"])
            lens = float(msg.get("lens") or 50.0)
            ortho = bool(msg.get("ortho", False))
            moving = bool(msg.get("moving", False))
        except (KeyError, TypeError, ValueError) as e:
            log("bad view: %s" % e)
            return
        sc = self.scene()
        self.busy = True
        try:
            if self.free_mode():
                v = find_view3d()
                if v is None:
                    return
                rv, space = v[2].data, v[3]
                rv.view_perspective = "ORTHO" if ortho else "PERSP"
                rv.view_location = Vector(loc)
                rv.view_rotation = Quaternion(rot)
                rv.view_distance = dist
                if abs(space.lens - lens) > 1e-3:
                    space.lens = lens
                v[2].tag_redraw()
            else:
                cam = self.camera(sc)
                if cam is None:
                    return
                if self.live_cycles(sc):
                    # Live Cycles renders the 3D view: if it left camera
                    # view (a numpad key, a view menu), wobble's orbit of
                    # the camera would move nothing on screen. Back in.
                    v = find_view3d()
                    if v is not None and v[2].data is not None and v[2].data.view_perspective != "CAMERA":
                        v[2].data.view_perspective = "CAMERA"
                        v[2].tag_redraw()
                self.cam_dist = dist
                m = Matrix.Translation(Vector(loc)) @ Quaternion(rot).to_matrix().to_4x4() @ Matrix.Translation((0.0, 0.0, dist))
                cam.matrix_world = m
                if not moving and getattr(sc.tool_settings, "use_keyframe_insert_auto", False):
                    guarded("camera key", lambda: cam.keyframe_insert("location"))
                    guarded("camera key", lambda: cam.keyframe_insert("rotation_euler"))
        finally:
            self.busy = False
        # Our own move: remember it so the poll does not echo it back.
        self.last_view = self._rounded(self.current_view())
        self.set_moving(moving)
        self.invalidate()

    def export_scene3d(self, path):
        """wobble's 3D cards read Blender's cameras, planes and empties:
        every camera (the scene camera first), every empty and every mesh
        of exactly four vertices, as world matrices per frame over the
        scene range, in `scene3d::import`'s JSON (Z up). A plane's own
        width is folded into its scale, so a card one unit wide fills it.
        Sampled frame by frame only when something could move."""
        if not path:
            return
        sc = self.scene()
        objs = [o for o in sc.objects if o.type in ("CAMERA", "EMPTY")
                or (o.type == "MESH" and len(o.data.vertices) == 4)]
        cam = self.camera(sc)
        objs.sort(key=lambda o: (o is not cam, o.type != "CAMERA", o.name))
        moving = any(o.animation_data is not None or o.parent is not None or len(o.constraints) > 0
                     or (o.type == "CAMERA" and o.data.animation_data is not None) for o in objs)
        frames = range(sc.frame_start, sc.frame_end + 1) if moving else [sc.frame_current]
        width = {}
        for o in objs:
            if o.type == "MESH":
                xs = [v.co.x for v in o.data.vertices]
                ys = [v.co.y for v in o.data.vertices]
                cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
                width[o.name] = (max(max(xs) - min(xs), 1e-6), cx, cy)
        out = {o.name: [] for o in objs}
        aspect = sc.render.resolution_x * sc.render.pixel_aspect_x / max(
            1e-6, sc.render.resolution_y * sc.render.pixel_aspect_y)
        here = sc.frame_current
        self.busy = True
        try:
            for f in frames:
                if len(frames) > 1:
                    sc.frame_set(f)
                for o in objs:
                    m = o.matrix_world.copy()
                    if o.name in width:
                        w, cx, cy = width[o.name]
                        m = m @ Matrix.Translation((cx, cy, 0.0)) @ Matrix.Scale(w, 4)
                    rec = {"frame": f, "matrix": [m[r][c] for r in range(4) for c in range(4)]}
                    if o.type == "CAMERA":
                        d = o.data
                        fit = d.sensor_fit
                        if fit == "VERTICAL" or (fit == "AUTO" and aspect < 1.0):
                            ap_w = d.sensor_height if fit == "VERTICAL" else d.sensor_width
                            ap_w = ap_w * aspect
                        else:
                            ap_w = d.sensor_width
                        rec.update({"focal": d.lens, "aperture": [ap_w, ap_w / aspect],
                                    "near": d.clip_start, "far": d.clip_end})
                    out[o.name].append(rec)
        finally:
            if len(frames) > 1:
                sc.frame_set(here)
            self.busy = False
        doc = {"fps": sc.render.fps / max(sc.render.fps_base, 1e-6), "up": "z", "objects": [
            {"name": o.name, "type": "camera" if o.type == "CAMERA" else "xform", "frames": out[o.name]}
            for o in objs]}
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(doc, fh)
        os.replace(tmp, path)
        self.bridge.send({"ev": "scene3d", "path": path})

    def invalidate(self):
        """The picture changed for reasons other than the frame: bump the
        version wobble stamps its cache with, and redraw."""
        self.scene_version += 1
        self.samples = 0
        self.settled = False
        self.dirty_at = None
        self.dirty_since = None
        self.bridge.send({"ev": "scene", "scene": self.scene_version})
        # An edit may have moved the camera: say where the eye is now.
        self.send_view()

    def ensure_ring(self, nbytes):
        if self.ring is not None and self.ring.fits(nbytes):
            return True
        if not SHM_DIR and not self.shm_dir and not USE_POSIX_SHM:
            return False
        d = self.shm_dir or SHM_DIR
        os.makedirs(d, exist_ok=True)
        old = self.ring
        self.ring_n += 1
        name = "wob_%d_%d" % (self.bridge.port, self.ring_n)
        path = os.path.join(d, name + ".shm")
        try:
            self.ring = RingWriter(name, path, RING_SLOTS, nbytes)
        except (OSError, ValueError) as e:
            self.bridge.send({"ev": "error", "msg": "pixel ring: %s" % e})
            return False
        self.bridge.send(dict({"ev": "shm"}, **self.ring.announce))
        if old is not None:
            old.close()
        return True

    shm_dir = ""
    cycles_noted = False
    cycles_stock_sent = False

    def publish(self, rows, frame, w, h, flags, ms, samples, pass_id=0, scale=None):
        """`rows`: an (h, w*4) float32 array, bottom-up. One memcpy into a
        free slot of the ring (a multiply on the way when `scale` is
        given), then the announcement."""
        need = stride_for(w, FMT_RGBA32F) * h
        if not self.ensure_ring(need):
            return
        got = self.ring.begin(w, h, FMT_RGBA32F, frame, flags, self.scene_version, ms, samples, pass_id)
        if got is None:
            # Every slot is still in wobble's hands: this frame is skipped,
            # the next one lands. Never torn.
            self.skipped += 1
            return
        slot, seq, view = got
        if scale is None:
            view[:, : w * 4] = rows
        else:
            np.multiply(rows, scale, out=view[:, : w * 4])
        self.ring.end(slot, seq)
        self.bridge.send({"ev": "frame", "slot": slot, "seq": seq})

    skipped = 0

    # -- drawing ----------------------------------------------------------

    def ensure_surfaces(self, w, h):
        """A ring of shared surfaces at the comp's size, announced to
        wobble; `False` when this Blender has no native module, wobble
        refused them, or a size change is pending (a new ring is made)."""
        if NATIVE is None or self.surf_ok is False:
            return False
        if self.surf is not None and self.surf.width() >= w and self.surf.height() >= h:
            return True
        try:
            tr = time.time()
            self.surf = NATIVE.SurfaceRing(RING_SLOTS, w, h)
            log("surface ring %dx%d made in %.0fms" % (w, h, (time.time() - tr) * 1000.0))
        except Exception as e:  # noqa: BLE001
            self.bridge.send({"ev": "error", "msg": "shared surfaces: %s" % e})
            self.surf = None
            self.surf_ok = False
            return False
        self.surf_held = {}
        self.surf_ok = None
        self.bridge.send({"ev": "surfaces", "ids": ",".join(str(i) for i in self.surf.ids()), "w": w, "h": h})
        return True

    def publish_surface(self, frame, w, h, flags, ms, samples, pass_id):
        """Blit the offscreen into a free surface and announce it with its
        stamp on the wire. `False` when every surface is still held."""
        for _ in range(RING_SLOTS):
            slot = self.surf_next
            self.surf_next = (self.surf_next + 1) % RING_SLOTS
            if slot not in self.surf_held:
                break
        else:
            self.skipped += 1
            return False
        try:
            tb = time.time()
            self.surf.blit(self.offscreen, slot, w, h)
            tb = (time.time() - tb) * 1000.0
            if tb > 50.0:
                log("surface blit took %.0fms (slot %d, %dx%d)" % (tb, slot, w, h))
        except Exception as e:  # noqa: BLE001
            self.bridge.send({"ev": "error", "msg": "surface blit: %s" % e})
            self.surf = None
            self.surf_ok = False
            return False
        self.surf_seq += 2
        seq = self.surf_seq
        self.surf_held[slot] = seq
        self.bridge.send({
            "ev": "frame", "slot": slot, "seq": seq, "frame": frame, "w": w, "h": h,
            "fmt": FMT_RGBA32F, "flags": flags, "scene": self.scene_version, "ms": ms,
            "samples": samples, "pass": pass_id,
        })
        return True

    def draw_viewport(self, sc, frame):
        """One offscreen draw of the viewport through the scene camera, at
        the comp's size, colour management OFF — scene-linear pixels in
        Blender's primaries, exactly as a float render writes them."""
        import gpu

        v = find_view3d()
        if v is None:
            self.bridge.send({"ev": "error", "msg": "no 3D viewport to draw from"})
            self.settled = True
            return
        win, area, region, space = v
        cam = self.camera(sc)
        if cam is None and not self.free_mode():
            self.bridge.send({"ev": "error", "msg": "the scene has no camera"})
            self.settled = True
            return
        w, h = self.size if self.size != (0, 0) else (sc.render.resolution_x, sc.render.resolution_y)
        w, h = max(1, int(w)), max(1, int(h))
        if self.moving:
            # A quarter of the pixels through the draw, the readback, the
            # ring and the upload while the hand is on the view; wobble
            # scales the smaller layer onto the canvas.
            w, h = max(1, w // 2), max(1, h // 2)
        if self.offscreen is None or (self.offscreen.width, self.offscreen.height) != (w, h):
            if self.offscreen is not None:
                self.offscreen.free()
            # Float32: a depth pass keeps its precision through the readback.
            self.offscreen = gpu.types.GPUOffScreen(w, h, format="RGBA32F")
        t0 = time.time()
        dg = bpy.context.evaluated_depsgraph_get()
        rv = region.data
        if self.free_mode() and rv is not None and rv.view_perspective != "CAMERA":
            # The 3D view's own eye, at the comp's size and aspect: the
            # same maths Blender uses for the region, for our size.
            ortho = rv.view_perspective == "ORTHO"
            vm = view_matrix(rv.view_location, rv.view_rotation, rv.view_distance, ortho)
            pm = projection(w, h, space.lens, ortho, rv.view_distance, space.clip_start, space.clip_end)
        else:
            vm = cam.matrix_world.inverted()
            pm = cam.calc_matrix_camera(dg, x=w, y=h)
        overlays = space.overlay.show_overlays
        # Film > Transparent is a render setting the viewport draw ignores:
        # the world comes back opaque. Leaving the background undrawn is
        # what makes the offscreen picture alpha 0 past the scene.
        background = not bool(self.spec.get("transparent", True))
        render_pass = getattr(space.shading, "render_pass", None)
        # One offscreen draw is the whole viewport render, every sample
        # (measured on 5.2.1: a second draw is byte-identical), so the
        # frame is settled the moment it exists.
        self.samples = int(getattr(sc.eevee, "taa_samples", 0) or 0)
        # In final mode a moving-hand draw is a stand-in: the render is
        # still owed when the hand stops, so the frame is not settled.
        final_owed = self.spec.get("quality") == "final"
        flags = 0 if final_owed else FLAG_SETTLED
        # The combined picture, then each pass the node asked for: the
        # viewport draws any one render pass on its own, so a pass is one
        # more draw of the same view.
        draws = [("COMBINED", 0)] + list(self.passes)
        # The full comp size is what the surfaces are made at; a half-res
        # moving draw lands in their top-left corner.
        full_w, full_h = self.size if self.size != (0, 0) else (w, h)
        use_surfaces = self.ensure_surfaces(max(1, int(full_w)), max(1, int(full_h)))
        # Z is the mist pass with the world's mist linear over the clip
        # range: mist * range is depth along the camera axis, in units.
        z_range = None
        if any(name == "MIST" for name, _ in self.passes):
            clip_end = float(space.clip_end if (self.free_mode() and rv is not None and rv.view_perspective != "CAMERA") else getattr(cam.data, "clip_end", 1000.0))
            if sc.world is None:
                sc.world = bpy.data.worlds.new("wobble")
            ms_ = sc.world.mist_settings
            guarded("mist", lambda: setattr(ms_, "falloff", "LINEAR"))
            guarded("mist", lambda: setattr(ms_, "start", 0.0))
            guarded("mist", lambda: setattr(ms_, "depth", clip_end))
            z_range = clip_end
        self.busy = True
        try:
            space.overlay.show_overlays = False
            for name, pid in draws:
                if render_pass is not None:
                    guarded("render_pass", lambda n=name: setattr(space.shading, "render_pass", n))
                t1 = time.time()
                with bpy.context.temp_override(window=win, area=area, region=region):
                    try:
                        self.offscreen.draw_view3d(
                            sc, bpy.context.view_layer, space, region, vm, pm,
                            do_color_management=False, draw_background=background,
                        )
                    except TypeError:  # an older draw_view3d without the flag
                        self.offscreen.draw_view3d(
                            sc, bpy.context.view_layer, space, region, vm, pm, do_color_management=False
                        )
                ms = (time.time() - (t0 if pid == 0 else t1)) * 1000.0
                # Z rides the ring (its scale is applied on the way in);
                # everything else goes as a shared texture when it can.
                if use_surfaces and name != "MIST" and self.surf is not None:
                    if self.publish_surface(frame, w, h, flags, ms, self.samples, pid):
                        continue
                buf = self.offscreen.texture_color.read()
                # As the GPU handed it back: float32, bottom-up. The one
                # copy is the memcpy into the ring; wobble's shader flips.
                # In MEMORY order: Blender 5.1's Buffer exports reversed
                # (column-major) strides over row-major data, and a
                # logical-order reshape scrambled every frame (measured on
                # Windows; `to_list()` disagreed with `asarray`). 'K'
                # follows the strides back to memory, a view either way.
                rows = np.asarray(buf, dtype=np.float32).ravel(order="K").reshape(h, w * 4)
                self.publish(rows, frame, w, h, flags, ms, self.samples, pid, z_range if name == "MIST" else None)
        finally:
            if render_pass is not None:
                guarded("render_pass", lambda: setattr(space.shading, "render_pass", render_pass))
            space.overlay.show_overlays = overlays
            self.busy = False
        self.settled = not final_owed

    def start_final(self, sc, frame):
        """Kick off an F12 render of `frame` without blocking the UI; the
        `render_complete` handler reads it back."""
        if self.rendering is not None:
            return
        self.ensure_viewer(sc)
        self.rendering = (frame, time.time())
        v = find_view3d()
        try:
            if v is not None:
                with bpy.context.temp_override(window=v[0]):
                    bpy.ops.render.render("INVOKE_DEFAULT", scene=sc.name)
            else:
                bpy.ops.render.render("INVOKE_DEFAULT", scene=sc.name)
        except Exception as e:  # noqa: BLE001
            self.rendering = None
            self.settled = True
            self.bridge.send({"ev": "error", "msg": "render %d: %s" % (frame, e)})

    # -- inbound ----------------------------------------------------------

    def set_frame(self, frame):
        sc = self.scene()
        self.current = frame
        self.samples = 0
        self.settled = False
        if sc.frame_current != frame:
            self.busy = True
            try:
                sc.frame_set(frame)
            finally:
                self.busy = False

    def take_texture(self, port, slot, seq):
        if self.tex_ring is None or np is None:
            return
        got = self.tex_ring.read(slot, seq)
        if got is None:
            log("texture for port %d lapped" % port)
            return
        w, h, px = got
        img = self.ensure_input_image(port, w, h)
        self.busy = True
        try:
            # wobble's rows are top-down; Blender's images are bottom-up.
            flat = np.ascontiguousarray(px[::-1]).ravel()
            img.pixels.foreach_set(flat)
            img.update()
        except Exception as e:  # noqa: BLE001
            self.bridge.send({"ev": "error", "msg": "texture %d: %s" % (port, e)})
        finally:
            self.busy = False

    def quit(self):
        try:
            wm = bpy.context.window_manager
            win = wm.windows[0] if wm and len(wm.windows) else None
            if win is not None:
                with bpy.context.temp_override(window=win):
                    bpy.ops.wm.quit_blender("INVOKE_DEFAULT")
                    return
        except Exception as e:  # noqa: BLE001
            log("quit (invoke): %s" % e)
        guarded("quit", bpy.ops.wm.quit_blender)

    def stop(self):
        """Tear the session down (the addon's disconnect)."""
        for hs, fn in (
            (bpy.app.handlers.depsgraph_update_post, self._on_update),
            (bpy.app.handlers.frame_change_post, self._on_frame),
            (bpy.app.handlers.save_post, self._on_save),
            (bpy.app.handlers.render_complete, self._on_render_done),
            (bpy.app.handlers.render_cancel, self._on_render_cancel),
        ):
            if fn in hs:
                hs.remove(fn)
        self.stopped = True
        self.bridge.close()
        if self.ring is not None:
            self.ring.close()
        if self.tex_ring is not None:
            self.tex_ring.close()
        if self.offscreen is not None:
            self.offscreen.free()
            self.offscreen = None
        self.surf = None

    stopped = False

    def tick(self):
        if self.stopped:
            return None
        while True:
            try:
                msg = self.bridge.q.get_nowait()
            except queue.Empty:
                break
            cmd = msg.get("cmd")
            if cmd == "frame":
                try:
                    self.pending = (int(msg.get("frame")), bool(msg.get("cached", False)))
                except (TypeError, ValueError):
                    pass
            elif cmd == "spec":
                self.apply_spec(msg)
            elif cmd == "view":
                # Coalesced: a burst of orbit steps applies the last one.
                self.pending_view = msg
            elif cmd == "ack":
                if self.ring is not None:
                    try:
                        self.ring.ack(int(msg["slot"]), int(msg["seq"]))
                    except (KeyError, TypeError, ValueError):
                        pass
            elif cmd == "sack":
                try:
                    slot, seq = int(msg["slot"]), int(msg["seq"])
                    if self.surf_held.get(slot) == seq:
                        del self.surf_held[slot]
                except (KeyError, TypeError, ValueError):
                    pass
            elif cmd == "surfaces":
                ok = bool(msg.get("ok", False))
                self.surf_ok = ok
                if not ok:
                    log("wobble could not import the shared surfaces; the ring serves")
                    self.surf = None
                    self.surf_held = {}
            elif cmd == "shm":
                try:
                    if self.tex_ring is not None:
                        self.tex_ring.close()
                    self.tex_ring = RingReader(msg)
                except (OSError, ValueError) as e:
                    self.bridge.send({"ev": "error", "msg": "texture ring: %s" % e})
            elif cmd == "tex":
                try:
                    self.take_texture(int(msg["port"]), int(msg["slot"]), int(msg["seq"]))
                except (KeyError, TypeError, ValueError) as e:
                    log("bad tex: %s" % e)
            elif cmd == "_connected":
                self.connected = True
                self.hello()
            elif cmd == "_disconnected":
                self.connected = False
                log("wobble disconnected")
            elif cmd == "cycles":
                self.bind_cycles(msg)
            elif cmd == "scene3d":
                guarded("scene3d", lambda: self.export_scene3d(str(msg.get("path", ""))))
            elif cmd == "ping":
                self.bridge.send({"ev": "pong"})
            elif cmd == "quit":
                self.quit()
        if not self.connected:
            return 0.05
        if self.spec:
            self.watch_engine()
        if self.pending_view is not None:
            msg, self.pending_view = self.pending_view, None
            self.apply_view(msg)
        # The user navigated in Blender (free mode) or moved the camera:
        # tell wobble where the eye is, and redraw for it.
        elif self.send_view():
            self.invalidate()
        # An edit redraws the frame wobble is looking at, once the hand
        # pauses — or after DIRTY_MAX_WAIT of continuous change, so a long
        # camera drag still reaches wobble as it goes.
        if self.dirty_at is not None:
            now = time.time()
            if now - self.dirty_at >= DIRTY_DEBOUNCE or now - self.dirty_since >= DIRTY_MAX_WAIT:
                self.invalidate()
        if self.pending is not None:
            (frame, cached), self.pending = self.pending, None
            self.set_frame(frame)
            # wobble has this one settled already: the playhead moved, the
            # draw is not owed.
            if cached:
                self.settled = True
        if self.current is not None and not self.settled and self.live_cycles(self.scene()):
            # Cycles renders continuously into wobble's buffer; a frame or
            # an edit restarts it on its own. Nothing to draw here.
            self.settled = True
            v = find_view3d()
            if v is not None:
                v[1].tag_redraw()
        if self.current is not None and not self.settled and np is not None:
            sc = self.scene()
            # Final quality renders the frame — but not under a moving hand:
            # while the eye is being dragged the viewport draw stands in
            # (Eevee, one sample, half res) and the render starts the moment
            # the hand stops. A Cycles orbit stays interactive that way.
            if self.spec.get("quality") == "final" and not self.moving:
                self.start_final(sc, self.current)
            elif time.time() - self.last_draw >= DRAW_PERIOD:
                self.last_draw = time.time()
                try:
                    self.draw_viewport(sc, self.current)
                except Exception as e:  # noqa: BLE001
                    self.settled = True
                    self.bridge.send({"ev": "error", "msg": "draw: %s" % e})
        # Poll fast: the timer's period is dead time on every round trip.
        return 0.008


LIVE = None


def start(port, shm_dir=""):
    global LIVE
    if LIVE is not None:
        LIVE.stop()
    if np is None:
        log("numpy is missing from this Blender; the bridge cannot move pixels")
        return None
    LIVE = Live(port)
    LIVE.shm_dir = shm_dir or SHM_DIR or bpy.app.tempdir
    log("live session ready on port %d" % LIVE.bridge.port)
    return LIVE


def stop():
    global LIVE
    if LIVE is not None:
        LIVE.stop()
        LIVE = None


# --------------------------------------------------------------------------
# Addon UI: a sidebar tab so a Blender the user already has open can join.


class WOBBLE_OT_connect(bpy.types.Operator):
    bl_idname = "wobble.connect"
    bl_label = "Connect to wobble"
    bl_description = "Listen for the wobble node on this port (the node's panel shows it)"

    def execute(self, context):
        port = context.window_manager.wobble_port
        if start(port) is None:
            self.report({"ERROR"}, "wobble bridge could not start (see the console)")
            return {"CANCELLED"}
        return {"FINISHED"}


class WOBBLE_OT_disconnect(bpy.types.Operator):
    bl_idname = "wobble.disconnect"
    bl_label = "Disconnect"

    def execute(self, context):
        stop()
        return {"FINISHED"}


class WOBBLE_PT_panel(bpy.types.Panel):
    bl_label = "wobble"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "wobble"

    def draw(self, context):
        col = self.layout.column()
        live = LIVE
        if live is None:
            col.prop(context.window_manager, "wobble_port", text="port")
            col.operator("wobble.connect")
        else:
            col.label(text="port %d — %s" % (live.bridge.port, "connected" if live.connected else "waiting"))
            if live.current is not None:
                col.label(text="frame %d%s" % (live.current, "" if live.settled else " (refining)"))
            col.operator("wobble.disconnect")


CLASSES = (WOBBLE_OT_connect, WOBBLE_OT_disconnect, WOBBLE_PT_panel)


def register():
    bpy.types.WindowManager.wobble_port = bpy.props.IntProperty(
        name="port", default=PORT or 47100, min=1, max=65535
    )
    for c in CLASSES:
        try:
            bpy.utils.register_class(c)
        except ValueError:
            pass


def unregister():
    stop()
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass


if __name__ == "__main__":
    # Launched by wobble: register the panel too, so the user sees the
    # state, and start on the port wobble chose.
    register()
    if PORT > 0 and not bpy.app.background:
        start(PORT, SHM_DIR)
    else:
        log("no WOBBLE_PORT in the environment (or background mode); panel only")
