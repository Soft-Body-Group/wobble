# What's new

Shown once, on the first launch of a new version (`app/src/whatsnew.rs`).
Newest first: a `## <version>` heading over `- ` bullets, a `**bold**` lead
drawn strong. Plain words for the people using wobble, not commit messages.

## 0.1.0-alpha.14
- **DiffusionLight** (new): an HDR latlong of the lighting, per frame, from the plate. LuxDiT as a second model on NVIDIA GPUs.
- **Normals** (new): normals for a clip.
- **Relight** (new): relights a plate from an environment, on the GPU.
- **Depth**: Depth Pro 2 models, with their own AI menu entry. Research-only licence, tagged.
- **GenAI**: LTX-2.5 runs locally on this machine's GPU, image- and text-to-video.
- **GenAI**: LTX 2.3 text and image to video, extend, retake, clean plate, render to real. LTX 2.5 upscale, reframe and video matte.
- **GenAI**: Seedance 2.5 text to video.
- **GenAI**: every result keeps its prompt, seed and settings, shown under the takes.
- **GenAI**: image edits take as many reference pictures as the model allows.
- **GenAI Pipeline**: made from a selection, one image input per picked node. A prompt group is a resizable outline.
- **Write**: ProRes and H.264 movie output. Proxy renders are named _prx.
- **Shader**: paste a Matchbox shader and its XML.
- **Quick create**: ctrl+Tab, then one key adds a node (G grade, B blur). Letters are remappable.
- **Graph**: G on several nodes grabs all their noodles. M-swipe a matte over an image to mask it, an image over an image for a Multiply.
- **Graph**: long menus scroll, GenAI steps grouped by what they do.
- **Tracker**: track backward, one frame at a time either way.
- **Transform / Corner Pin**: invert toggle.
- **Planar Track**: matted pixels sit out the solve.
- **Physics**: animated colliders push bodies, holes in colliders are open.
- **Luma Key**: points stop at their neighbours, shift-click a crossfade to shape its slope.
- **Spline**: takes the format of the picture it masks.
- **Slip**: a positive offset now delays the clip. Old comps are converted on load.
- **Project browser**: browse a project's folder tree and load by shot.
- **Clip browser**: Home at the top, a Comp folder for the saved comp's folder.
- **Scrub**: key hints fan out like Wobble's.
- Fixed: the movie build failed against FFmpeg older than 6.

## 0.1.0-alpha.13
- **Refract** (new): a plate or latlong through bumpy glass, from a height or normal map. IOR, dispersion, roughness.
- **Splat** (new): Gaussian splat captures (.ply, .spz) through a Camera, with RGBA and depth out.
- **Morph Dissolve** (new): A and B warp toward each other along their optical flow while they cross-dissolve.
- **Compare** (new): A or B, or a draggable wipe that can be animated.
- **Turn** (new): 90 CW, 90 CCW or 180, the format turned with it, no resampling.
- **Uncrop** (new): puts a render of a Crop's box back into the full format.
- **Project Load** (new): footage picked by project and shot instead of a path. Comps open and save the same way.
- **ACES 1.2**: sRGB and Rec.709 (ACES) views, Output - Rec.709 and sRGB input and output spaces.
- **GenAI**: one node per model in the AI menu. New image, video, upscale, outpaint, depth, normals, relight and image-to-3D models.
- **GenAI**: video models take clip inputs, with masked video inpaint and video edit.
- **GenAI Pipeline**: steps are nodes you dive into. Comp nodes work inside, and a model step runs from its own node.
- **GenAI Pipeline**: templates, app mode, prompt tools, one prompt through several models, batches over prompts, seeds or frames, and prompt groups.
- **GenAI**: take history, a gallery of results on every model step, and progress on the node.
- **GenAI**: a take at another frame rate is conformed by a Speed node.
- **Roto**: pen-tool drawing, drag a new point to pull its tangents. Enter dives in: each shape is a node, Axis nodes group and move them in 3D, and a Camera input projects them.
- **Load**: its own proxy scale, 1, 0.75, 0.5 or 0.25, applied before every cache.
- **Reformat**: size by factor.
- **Sammie**: tracking 4x faster, matting resolution control, weights download shown.
- **Scrub**: now its own app, shipped beside Wobble on all three OSes. The scrub button starts it. A hotkey tour runs on first launch and from help > hotkey tour.
- **Graph**: pasting file paths or script Read blocks makes Load nodes at the cursor. F frames, R is hold+drag rotate only.
- **Viewer**: pixel readout moved to Z, shift+Z pins it.
- **Eyedropper**: a drag follows the pixel under the pointer, shift-drag averages a box.
- **Number fields**: Tab commits the typed value, and typed arithmetic (100/2) commits its answer.
- **Paint**: brush size is in full-res pixels. Clone: hold alt/cmd to move the source.
- **Timeline**: a clear in/out button on the transport row.
- **New comp**: a "new" button and ⌘N, asking first over unsaved changes.
- Fixed: a save or autosave inside a dive wrote only that node, not the whole comp.
- Fixed: the playhead inside a dive left the comp's range.
- Fixed: a GenAI take landing while inside its pipeline was lost.

## 0.1.0-alpha.12
- **Particles** (new): 2D particle system.
- **Plasma** (new): solar corona. Presets: corona, eclipse, prominences.
- **Planet** (new): seamless noise sphere, lit disc or latlong.
- **Starfield** (new): star sky through a Camera, Milky Way glow.
- **Env** (new): latlong background at infinity.
- **Lens Flare** (new): draggable light, can be blocked by the picture.
- **Tone Map / Rolloff** (new): Knee, Exponential, Reinhard, Hable, ACES fit, AgX-like.
- **Contrast** (new): pivot, shadows/highlights, soft toe and shoulder.
- **Grade**: contrast with pivot, master and per zone.
- **LUT**: matches Resolve (tetrahedral, .csp, DaVinci WG).
- **Text**: per-glyph layers, outline/shadow/glow, animators, text on a path.
- **String**: Prompt renamed.
- **Clip browser**: rebuilt (path bar, favourites, folder tree, scrubbing tiles).
- **Layer**: mute and solo.
- **Noise**: tenth-speed steps, 8-bit/single channel out, view inner nodes.
- **Dives**: U goes up a level (unify size is shift+U).
- **Viewer**: F frames, plain R/G/B show a channel.
- **RGB sliders**: movable range.
- **Playback**: Crop-aware decoding, small cache scrubs show pictures.
- Fixed: long Particles catch-up could lose the GPU on Mac.

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
