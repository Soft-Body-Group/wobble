# Blender's own AgX numbers for engine/tests/agx_blender.rs:
#   blender -b --python tools/agx_reference.py -- out.png out.json [--looks]
# Without --looks: look None (the BLENDER table). With it: every AgX look
# as well (the LOOKS table), keyed by Blender's look name.
import bpy, json, sys
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[-2:]
with_looks = "--looks" in args
args = [a for a in args if a != "--looks"]
out_png, out_json = args[-2], args[-1]
patches = [
  (0.0,0.0,0.0),(0.001,0.001,0.001),(0.01,0.01,0.01),(0.05,0.05,0.05),(0.18,0.18,0.18),
  (0.5,0.5,0.5),(1.0,1.0,1.0),(2.0,2.0,2.0),(4.0,4.0,4.0),(16.0,16.0,16.0),(100.0,100.0,100.0),
  (1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),(0.18,0.02,0.01),(0.02,0.1,0.3),(0.9,0.6,0.1),
  (4.0,0.5,0.1),(0.3,2.0,0.4),(0.05,0.2,5.0),(0.7,0.35,0.9),(0.123,0.456,0.789),(8.0,8.0,0.2),
]
n = len(patches)
img = bpy.data.images.new("p", width=n, height=1, float_buffer=True, alpha=False)
px = []
for r,g,b in patches: px += [r,g,b,1.0]
img.pixels[:] = px
sc = bpy.context.scene
sc.display_settings.display_device = 'sRGB'
sc.view_settings.view_transform = 'AgX'
sc.view_settings.exposure = 0.0
sc.view_settings.gamma = 1.0
sc.render.dither_intensity = 0.0
s = sc.render.image_settings
s.file_format = 'PNG'; s.color_depth = '16'; s.color_mode = 'RGB'

def render(look):
    sc.view_settings.look = look
    img.save_render(out_png, scene=sc)
    back = bpy.data.images.load(out_png)
    back.colorspace_settings.name = 'Non-Color'
    v = list(back.pixels[:])
    bpy.data.images.remove(back)
    return [[round(v[i*4+k], 5) for k in range(3)] for i in range(n)]

looks = ['None']
if with_looks:
    looks += ['AgX - ' + l for l in ('Punchy', 'Greyscale', 'Very High Contrast',
              'High Contrast', 'Medium High Contrast', 'Base Contrast',
              'Medium Low Contrast', 'Low Contrast', 'Very Low Contrast')]
res = {look: render(look) for look in looks}
json.dump({"patches": patches, "srgb_display": res['None'], "looks": res,
           "blender": bpy.app.version_string}, open(out_json, "w"))
print("OK", bpy.app.version_string, looks)
