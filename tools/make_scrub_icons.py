#!/usr/bin/env python3
# Scrub's app icon: wobble's tile (Apple's grid: an 824 square on a 1024
# canvas, ~185 corners), a dark screen inset, and on it a timeline: three
# tracks of clips and a white playhead with its head. Every size is drawn at
# 4x and filtered down; Pillow writes the .ico and the .icns too, so this
# runs on any OS. Commit the results; builds never render the icon.
#
#   python3 tools/make_scrub_icons.py scrub/app/icons
import sys
from PIL import Image, ImageDraw
OUT = sys.argv[1]
TILE = (40, 132, 140)      # teal: not wobble's magenta, never orange
SCREEN = (24, 24, 24)
CLIPS = [(122, 196, 204), (88, 160, 170), (150, 214, 220)]
def draw(px):
    ss = 4
    S = px * ss
    k = S / 1024
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    t0, t1 = 100 * k, 924 * k
    d.rounded_rectangle([t0, t0, t1, t1], radius=185 * k, fill=TILE)
    s0, s1 = 190 * k, 834 * k
    d.rounded_rectangle([s0, s0, s1, s1], radius=110 * k, fill=SCREEN)
    # three tracks of clips: (start, end) in screen units 0..1
    tracks = [[(0.06, 0.40), (0.46, 0.94)], [(0.06, 0.24), (0.30, 0.70), (0.76, 0.94)], [(0.06, 0.56), (0.62, 0.94)]]
    w = s1 - s0
    for i, (row, col) in enumerate(zip(tracks, CLIPS)):
        y0 = s0 + w * (0.26 + i * 0.22)
        y1 = y0 + w * 0.14
        for a, b in row:
            d.rounded_rectangle([s0 + w * a, y0, s0 + w * b, y1], radius=22 * k, fill=col)
    # playhead
    x = s0 + w * 0.58
    lw = max(28 * k, 1.5 * ss)
    d.rectangle([x - lw / 2, s0 + w * 0.17, x + lw / 2, s0 + w * 0.94], fill=(255, 255, 255))
    hw = 70 * k
    top = s0 + w * 0.08
    d.polygon([(x - hw, top), (x + hw, top), (x, top + hw * 1.25)], fill=(255, 255, 255))
    return im.resize((px, px), Image.LANCZOS)
import os
ico = []
for px in [16, 24, 32, 48, 64, 128, 256, 512, 1024]:
    im = draw(px)
    im.save(os.path.join(OUT, f"scrub-{px}.png"))
draw(1024).save(os.path.join(OUT, "scrub.ico"), sizes=[(s, s) for s in [16, 24, 32, 48, 64, 128, 256]])
draw(1024).save(os.path.join(OUT, "scrub.icns"))
