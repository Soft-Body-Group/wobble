# The Windows twin of release_public.sh: one build for everyone, built with
# --no-default-features (the `commerce` feature never ships), movie support
# through an LGPL-only FFmpeg whose DLLs sit beside wobble.exe (dynamic, so a
# user can swap them, which LGPL asks for). The licences (and FFMPEG.txt) sit
# in licenses\, so the top level is the app and what it needs to run.
# Scrub, the timeline editor (scrub/app), ships beside it as scrub.exe on the
# same DLLs (wobble's scrub button starts the binary beside it).
#
#   powershell -File tools\release_public_win.ps1          # dist\public-win\wobble-<ver>-<git>-windows-x86_64.zip
#                                                         # and scrub-<ver>-<git>-windows-x86_64.zip (same build)
#   $env:WOBBLE_MOVIE="0"; powershell -File tools\release_public_win.ps1   # ...-nomovie, no FFmpeg
#
# FFmpeg comes from tools\build_ffmpeg_lgpl_win.cmd (MSVC + MSYS2), and
# bindgen needs libclang: $env:LIBCLANG_PATH, else target\tools\libclang.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot)
$git = (git rev-parse --short=12 HEAD).Trim()
if (git status --porcelain --untracked-files=no) { $git = "$git-dirty" }
$ver = ((Select-String -Path app\Cargo.toml -Pattern '^version' | Select -First 1).Line -split '"')[1]
$movie = $env:WOBBLE_MOVIE -ne "0"
$suffix = if ($movie) { "" } else { "-nomovie" }
$out = "dist\public-win$suffix"
$name = "wobble-$ver-$git-windows-x86_64$suffix"
$env:CARGO_TARGET_DIR = "target\public"
$env:CARGO_PROFILE_RELEASE_DEBUG = "false"
# No build machine paths in the binary (panic locations, file!()). RUSTFLAGS
# replaces .cargo/config.toml's rustflags, so crt-static is repeated here.
$env:RUSTFLAGS = "-C target-feature=+crt-static --remap-path-prefix=$($env:USERPROFILE)=~ --remap-path-prefix=$((Get-Location).Path)=wobble"
$features = @()
if ($movie) {
    cmd /c "tools\build_ffmpeg_lgpl_win.cmd"
    if ($LASTEXITCODE) { throw "FFmpeg build failed" }
    $env:FFMPEG_DIR = (Resolve-Path target\ffmpeg-lgpl).Path
    if (-not $env:LIBCLANG_PATH) { $env:LIBCLANG_PATH = (Resolve-Path target\tools\libclang).Path }
    $features = @("--features", "app/movie")
}
# The feedback ingest key is compiled in (app/src/report.rs): from the
# environment, else tools\feedback-worker\.secrets (INGEST_KEY=<hex>).
if (-not $env:WOBBLE_INGEST_KEY -and (Test-Path tools\feedback-worker\.secrets)) {
    $k = Select-String -Path tools\feedback-worker\.secrets -Pattern '^INGEST_KEY=(.*)$' | Select-Object -First 1
    if ($k) { $env:WOBBLE_INGEST_KEY = $k.Matches[0].Groups[1].Value }
}
if (-not $env:WOBBLE_INGEST_KEY) { Write-Warning "no WOBBLE_INGEST_KEY; this build will not upload reports" }
cargo build --release -p app --no-default-features @features
if ($LASTEXITCODE) { throw "cargo build failed" }
# Scrub with its own defaults (the clip browser) plus `fold` (clips can be
# Wobble comps, wobble hosted in process); same env, same target dir.
cargo build --release -p scrub_app --features fold
if ($LASTEXITCODE) { throw "cargo build (scrub) failed" }

if (Test-Path $out) { Remove-Item -Recurse -Force $out }
$st = "$out\$name"
New-Item -ItemType Directory -Force "$st\licenses", "$st\plugins" | Out-Null
Copy-Item target\public\release\wobble.exe, target\public\release\scrub.exe $st
Copy-Item LICENSE, NOTICE.md, THIRD_PARTY_LICENSES.md, licenses\LGPL-2.1.txt "$st\licenses"
# One flat folder, so LICENSING.md's link to licenses/LGPL-2.1.txt is a sibling.
(Get-Content LICENSING.md -Raw).Replace("licenses/LGPL-2.1.txt", "LGPL-2.1.txt") |
    Set-Content -Encoding utf8 -NoNewline "$st\licenses\LICENSING.md"
Copy-Item -Recurse plugins\invert, plugins\vignette "$st\plugins"
$movieLine = "This build has no movie (ProRes/H.264) support: read EXR, DPX, PNG or JPEG sequences."
if ($movie) {
    Copy-Item target\ffmpeg-lgpl\bin\*.dll $st
    $movieLine = "Movies (ProRes, H.264 and friends) read through FFmpeg; see licenses\FFMPEG.txt."
    $configure = (Get-Content target\ffmpeg-lgpl\CONFIGURE.txt -Raw).Trim()
    @"
Wobble uses FFmpeg (https://ffmpeg.org) under the GNU Lesser General Public
License 2.1 or later (LGPL-2.1.txt, beside this file). Only FFmpeg's LGPL
parts are built: no --enable-gpl, no --enable-nonfree, no x264, x265 or any
other external codec library. It is linked dynamically: the libraries are the
av*.dll and sw*.dll files beside wobble.exe, and you may replace them with
your own build of the same version.

Version: FFmpeg 8.1, unmodified
Source:  https://download.wobble.ing/ffmpeg-8.1.tar.bz2
         (identical to https://ffmpeg.org/releases/ffmpeg-8.1.tar.bz2)
SHA-256: c07039598df7d64d3c8b42c4e25b1959fc908621c6f6c2946881133f3b27eda2
Built with MSVC (Visual Studio 2022, x64) under MSYS2, then make install:
  $configure
The build script is tools/build_ffmpeg_lgpl_win.sh in Wobble's source.
"@ | Set-Content -Encoding ascii "$st\licenses\FFMPEG.txt"
}
@"
Wobble $ver (Windows, x64)

Unzip anywhere and run wobble.exe. Needs a GPU with Vulkan or DX12.
scrub.exe is Scrub, the timeline editor: an app of its own (run it directly,
or pin it to Start or the taskbar), and also what wobble's scrub button starts.
$movieLine
Plugins in the plugins folder load automatically; your own go in %USERPROFILE%\.wobble\plugins.

The first launch asks whether to join alpha testing and to be told about new
builds; nothing is sent until you answer. Change either in prefs any time.

Wobble is free software under the GPL-3.0-or-later; see the licenses folder
(LICENSING.md says what each file covers).
Built from $git.
"@ | Set-Content -Encoding ascii "$st\README.txt"
# The few home paths the remap misses, blanked in place (same length).
$py = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py" }
& $py tools\scrub_paths.py $st
if ($LASTEXITCODE) { throw "scrub_paths found a home path it could not blank" }
# tar writes forward-slash entries; Compress-Archive on PowerShell 5 does not.
Push-Location $out
tar.exe -a -c -f "$name.zip" $name
Pop-Location
"== $out\$name.zip"
# One build, two names: the same folder as scrub-<...>, led by Scrub, which
# scrub.ing links to (download.scrub.ing serves scrub-*). Not for -nomovie.
if ($movie) {
    $sname = "scrub-$ver-$git-windows-x86_64"
    $sst = "$out\$sname"
    if (Test-Path $sst) { Remove-Item -Recurse -Force $sst }
    Rename-Item $st $sname
    $st = $sst
    @"
Scrub $ver (Windows, x64)

Scrub, the video editor, and Wobble, the compositor, are one app with two
front doors: this folder is the same build as the Wobble download.

Unzip anywhere and run scrub.exe (wobble.exe opens Wobble). Needs a GPU with
Vulkan or DX12. Pin scrub.exe to Start or the taskbar to keep it handy.
$movieLine
Hotkeys and the guide: https://scrub.ing/guide/

Scrub is free software under the GPL-3.0-or-later; see the licenses folder
(LICENSING.md says what each file covers).
Built from $git.
"@ | Set-Content -Encoding ascii "$st\README.txt"
    Push-Location $out
    tar.exe -a -c -f "$sname.zip" $sname
    Pop-Location
    "== $out\$sname.zip"
}
Remove-Item -Recurse -Force $st
