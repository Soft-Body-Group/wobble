# Blender fork patch: live Cycles into wobble

`wobble-cycles.patch` applies to Blender at commit `9e2066ae`
("Cycles: Bump minimum driver version in UI from 570 to 575", the 5.2 line):

    git checkout 9e2066ae && git apply path/to/wobble-cycles.patch

It adds `_cycles.wobble_target(handle, size, uuid, stamp)` and a display
driver (`WobbleTeeDisplayDriver`) that shows Cycles in Blender's own viewport
as usual and mirrors every display update into a buffer wobble owns, via
Cycles' Vulkan graphics interop (CUDA). wobble's side is
`engine/src/blender/cycles.rs`; the live test is `engine/tests/cycles_live.rs`.

Windows (Win32 handles, CUDA) and macOS (Metal). On macOS there is no import:
wobble shares one POSIX shared memory object (stamp page + pixels), the fork
wraps the pixel pages in an MTLBuffer (`wobble_metal.mm`) and Cycles' Metal
interop writes into it with rows padded to 256 bytes (stamp version 2), so
wobble copies it straight into a texture. Build on the Mac with
`make update` (libraries), then `make ninja`; the app is
`../build_darwin/bin/Blender.app` once `ninja install` has run with
`CMAKE_INSTALL_PREFIX` pointing there.

The patch also carries
`source/creator/wobble_msvc_compat.cc` and a relaxed MSVC version check, so it
links with Visual Studio 17.11; neither is needed elsewhere.
