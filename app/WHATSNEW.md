# What's new

Shown once, on the first launch of a new version (`app/src/whatsnew.rs`).
Newest first: a `## <version>` heading over `- ` bullets, a `**bold**` lead
drawn strong. Plain words for the people using wobble, not commit messages.

## 0.1.0-alpha.11
- **Collect** (new, top bar **collect…**): copies every file the comp reads (Loads, LUTs, fonts, 3D and .blend files) to a folder and saves the comp there, pointing at the copies. Or pick an organisicator shot: the footage is ingested into the project and the Loads name it by asset id, and the comp goes in the shot's comp folder.
- **LUT** (new): applies a .cube, .3dl, .spi1d or .spi3d file, 1D or 3D, with shaper and domain support.
- **Inverse View** (new): runs the viewer transform backwards, so display-referred plates (such as AI video) come in scene-linear. Dither and a soft limit for 8-bit sources.
- **Freeze** (new): holds the input on one frame. The frame can be keyed.
- **Frame Fix** (new): rebuilds bad frames from the good frames either side. **blend** is a live cross-fade; **optical flow** makes the frames with RIFE on a click.
- **Text** (new): words over the picture, in sans, mono or any font file. Position, size and colour can be keyed, and stay sharp at any size.
- **Burnin** (new): prints the frame number into a corner, on an optional box. The words around the number are yours ("frame {frame}").
- **Crop**: Soft Crop is now Crop, one node with three modes: **bounding box** (the old soft crop), **crop** (removes the pixels outside the box) and **reformat** (the box becomes the frame). A Crop has its own **bake** tab, and a bake after a crop stores only the box.
- **Transform** added under a crop starts with its pivot at the centre of the crop.
- **Layer**: a **layer outputs** switch gives the stack one extra output per layer. Each one is that layer on its own, with its start, speed and trim but no opacity, blend or other layers.
- **Layer** timeline: dragging an end trims (the footage stays put), alt-drag stretches, shift-drag on a bar slips the footage. Bars sit on their input's own frames and move left and right only. Each layer can run on past its clip with hold, loop or bounce.
- **Layer**: added with several nodes selected, it stacks them all in pick order. A **global** button sets its canvas to the script's format.
- **Load**: can keep a single channel (r, g, b, a or luma) and/or store 8-bit, which decodes and plays much faster. Noodles show it: a thin strand for one channel, dashes for 8-bit.
- **Render**: location, name, padding and an **enclose in subfolder** switch build the path, the full path is shown, and **make Load** adds a Load that reads the render. It can write one channel and pick the EXR compression, and **send to…** a paired wobble (the other side must switch on accepting).
- **Bake**: renders at its input's format. Mattes bake as one channel. Under proxy, a full bake plays at half size, otherwise a half-size bake is made beside it. EXR writes no longer hold up the next frame.
- **Graph**: alt+B bakes every selected Bake and Crop, upstream first; alt+C caches every selected Load.
- **Graph**: hold B and swipe a dragged node over another to blend them (the same swipe as M, without M's region popup).
- **Graph**: **boring** mode (the graph bar's toggle): plain pills and Load cards showing their frame, sockets on the edges, and straight noodles.
- **Graph**: dragged nodes snap so their centres line up with nearby nodes, or so a noodle runs exactly straight up or across. A faint line shows what they snapped to. Hold ctrl to drag freely.
- **Switch**: click a stud on the node to switch to that input, or the hub for the next one.
- **Number fields**: the scroll wheel steps the value under the pointer (one notch = four pixels of drag; Z/X/C/V still set the rate). Only a scroll that starts on a field changes it, so scrolling a panel past fields still scrolls the panel.
- **Curve editors** in properties (a LumaKey's bands, a Grade's zones, colour curves): Q frames the picked points, or all of them with none picked.
- **Time bar**: left and right arrows step one frame.
- **Time Range**: the same in/out bar as a Load, with **reset**, **timeline in/out** and an optional **start at**.
- **Layer** and **Time Range** timelines: navigate like the main time bar (wheel / ctrl+MMB zoom, MMB or alt-drag pan, R frames the in–out) with its ` pie, and no longer jump to re-fit when you let go of a drag.
- **2dini**: keys on nodes inside it (bone turns, any socket) play back and survive stepping in and out.
- Viewer handles follow a UV Warp downstream.
- **Playback**: a plate cut into sections by Crops decodes once and keeps only the boxes it needs; only the viewed slot decodes while playing; a bake decodes only what it bakes. Scrubbing big plates shows pictures where it used to stall.
- **Recorder**: right-click the record button to choose where clips and screenshots are saved.
- **Scripting API**: opens at launch in every build (prefs → **scripting API** turns it off).
- **Browser**: a bare drive letter (C:) goes to its root on Windows.

## 0.1.0-alpha.10
- **CameraTrack** (new): solves a 3D camera from the footage using SEA-RAFT feature tracks and a structure-from-motion solve.
- **CameraTrack**: S-click solved points to pick them. Then **ground** fits the floor, **origin** moves the centre, **set scale** sizes the scene from two points.
- **3D view**: every Camera in a Blend 3D is drawn.

## 0.1.0-alpha.9
- **3D view**: a Card or Blend 3D on the viewer renders through an orbiting camera with Blender's controls (MMB orbit, shift pan, ctrl or wheel dolly, numpad views, R or Home to frame), with a floor grid and axis gizmo. Cards show their outlines, can be clicked to pick, and have move, rotate and scale handles (G / R / S).
- **Blend 3D** (was Scene): one input bar like 2dini's Merge that grows a free input. 3D noodles are green.
- **Ratio locks**: a padlock between paired values (format width and height, scale on Transform, roto and cards, Grid spacing, filmback). Locked, an edit keeps the ratio.
- **Graph**: hold S and drag to scale the selection's spacing, hold R and drag to rotate it. Hovering an input socket says what the node does with it.
- **Right-angle noodles**: parallel runs spread into lanes, and tidy leaves room for them.
- **Convolve** (FFT) is 4.5x faster.
- **Godrays**: samples and dither controls.
- Modulate sockets and noodles are yellow.
- A node's frame range is its plate's.

## 0.1.0-alpha.8
- **3D cards** (new): Camera, Card, Scene and Render3D nodes. Cards are ray-cast in exact depth order, with a depth output and motion blur. Cameras and transforms load from .usda, .json and .chan files, or follow a Blender node's scene live.
- **Time Remap** (new): plays a source in..out over a target in..out, shown on two range bars with a line from each target frame to the source frame it reads. The curve-driven remap is now called Time Curve.
- **Time Range** (new): sets a noodle's frame range only.
- **Lens Distortion** (new): Brown-Conrady undistort and distort, anamorphic squeeze, overscan, chromatic aberration and an STMap output.
- **Godrays** (new): light shafts from a source point, with a source matte and a viewer handle.
- **EdgeExtend** (new): pixel spread and edge expander in one.
- **Kernels**: kernel generator nodes, an FFT Convolve, and custom kernels for Defocus.
- **Load**: in and out in the clip's own frames, and "in at" to place the clip in the comp. A range bar over the files: drag an end, or slide it to slip. A Load draws green when its in..out is cached and red when not, with cache and clear buttons.
- **Load**: pick footage from an organisicator project by name, and ingest a plain Load's footage into a shot.
- **Time bar**: taller, with frame ticks, labelled in/out marks you can drag, click to type a frame, and a backtick pie to zoom to the comp, node or in/out range. It pans and zooms with the same hands as the viewer and graph. The transport buttons moved around it.
- **2dini**: light (radiance cascades) and SDF effects, a fluid mode on the Physics node, a Particles node, onion skins, and Skeleton, Skin and Ragdoll nodes.
- **Layer**: a multi-input bar; slide an input along it to reorder. A Defocus or Blur on a layer no longer vanishes.
- **Network**: share a wobble on the local network behind a pairing code, send nodes to a paired wobble, and help render a shared setup.
- **Viewer**: Blender's AgX looks and ACES 1.3 gamut compression. A backtick pie to frame, go 1:1 or 1/2, and show a channel.
- **CornerPin / Transform**: import track… keys the pin from exported corner pin or transform text.
- **Sammie**: auto preview shows the segment as you click. A track covers the plate's range without handles.
- **Depth**: much less GPU memory, steps down in size instead of running out, and works on Windows.
- **Glint**: chromatic spread and hue shift.
- **Render** writes Data inputs as one float Z channel.
- Stills of another size in a sequence are fitted into its frame.
- A stale bake draws red like an unbaked one.
- A bypassed Blend passes the background, not the foreground.
- R in the curve editor frames the shown curves.
- I / O set in and out over the timeline or while scrubbing.

## 0.1.0-alpha.7
- **Stabilize** (new): solves the camera move with SEA-RAFT and locks it off, smooths it or reverses it.
- **Depth** (new): depth from Video Depth Anything, rendered on a click. The large models are non-commercial and tagged.
- **HighlightRebuild** (new): rebuilds clipped channels from the ones that are not clipped.
- **HDRCNN** (new): highlight reconstruction (Eilertsen et al.), rendered on a click.
- **LTX HDR** (new): LTX-2.5 SDR-to-HDR through an external tool, rendered on a click. It is under the LTX-2 Community Licence, so the node is tagged and a Render warns.
- **Tracker**: a stabilise option drives a CornerPin's source corners so the tracked region holds still.
- **2dini**: an Edit node and a direct-select tool for moving individual points.
- **Movies**: Load shows the file's frame rate, with a button to set the comp to match. Movie frames decode faster.
- Large node graphs (1000+ nodes) draw and edit much faster.
- Fixed a crash when loading a setup over one with different nodes.
- Keys 1-4 deselect the node they send to a viewer.
- A screenshot button next to record saves a PNG of the window.

## 0.1.0-alpha.6
- **Rifetime** (new): slow motion with RIFE in-between frames, set by a speed or by a keyed curve of source frames.
- **Warp** (new): warp with splines
- **Mirror** and **Flip/Flop** (new nodes).
- **Transform**: shear and taper.
- **GridWarp**: a box around several picked points moves, scales, skews and rotates them.
- **Curve editor**: extrapolation, typed key values, G/S/R, copy and paste, smooth, decimate, mirror, vector handles, stepped keys, easing presets, normalize and ghosts.
- **2dini**: a tool bar in the viewer (circle, rectangle, polygon, star, text, line, spline), a Brush node, gradient fills with a ramp editor, fill and stroke swatches, handles on drawn shapes, rulers and snapping, and blend mode and stacking order per noodle.
- Motion vectors have their own noodle type (yellow), from a Flow into VectorBlur and VectorWarp.
- Invert, Gamma, Mult, Grade and Curve accept a matte and output a matte.
- Blur, Median, Convolve, Dilate and VectorBlur process a matte as one channel (a 4K matte median: 187 ms to 76 ms).
- **Sammie**: preview the segment on the current frame before tracking. SAM 2.1 large is the default model.
- New nodes take the size of the nodes near them.
- A connected output keeps its disc, so another noodle can branch from it.
- Shift-connect fills a filter's main input before its mask and modulate inputs.
- Fixed wrong colours (magenta skin, lime green screens) in some H.264 movies.
- **Curve node**: smoothed ends no longer overshoot past the last key.
- **Scripting API**: channels are listed in panel order with labels; node_set with drag=true is one undo step.

## 0.1.0-alpha.5
- **Flow** (new): optical flow with SEA-RAFT (default) or RIFE, read by the new VectorBlur and VectorWarp nodes.
- **Upscale** (new): Real-ESRGAN and SPAN upscaling, rendered on a click.
- **SeedVR2** (new): diffusion restoration through an external tool, rendered on a click.
- **Sammie** (new): click-to-roto with SAM 2.
- New nodes: Diffuse, Convolve (kernels up to 7x7), Median, EdgeDetect, GridWarp.
- **Defocus**: a modulate input that scales the blur per pixel.
- **Paint**: pen pressure, brush presets, and blend modes (normal, erase, add, subtract, multiply, screen).
- **Render**: the frame range comes from the input, a set range, or the global range.
- **Properties**: click a panel's name to select it and resize it on its own.
- **Graph**: shift+R labels each noodle with its frame range. Output sockets take their noodle's colour.
- Q and W restore the layout exactly. The hotkeys window scrolls and jumps to a key.
- **Scrub** (edit mode): undo, a save prompt on close, blank clips, and a menubar button that opens it in its own process.
- Symbols that drew as empty boxes now draw correctly.
