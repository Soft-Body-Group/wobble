#!/bin/sh
# Test media for edify (gitignored). Needs the ffmpeg CLI.
set -e
cd "$(dirname "$0")"
SRC="testsrc2=s=3840x2160:r=24:d=10,noise=alls=12:allf=t"
[ -f prores_hq_4k.mov ] || ffmpeg -v error -y -f lavfi -i "$SRC" -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le prores_hq_4k.mov
[ -f h264_4k.mp4 ] || ffmpeg -v error -y -f lavfi -i "$SRC" -c:v libx264 -preset fast -crf 18 -g 48 -bf 2 -pix_fmt yuv420p h264_4k.mp4
SEQ="testsrc2=s=1920x1080:r=24:d=2,noise=alls=12:allf=t"
mkdir -p exr png dpx
[ -f exr/shot.1001.exr ] || ffmpeg -v error -y -f lavfi -i "$SEQ" -pix_fmt gbrpf32le -c:v exr -compression zip1 -format half -start_number 1001 exr/shot.%04d.exr
[ -f png/shot.1001.png ] || ffmpeg -v error -y -f lavfi -i "$SEQ" -pix_fmt rgb24 -start_number 1001 png/shot.%04d.png
[ -f dpx/shot.1001.dpx ] || ffmpeg -v error -y -f lavfi -i "$SEQ" -pix_fmt gbrp10le -start_number 1001 dpx/shot.%04d.dpx
ls -la . exr | head -20
# A typical delivery/camera bitrate (the CRF clip above is ~550 Mbit/s: grain).
[ -f h264_4k_50m.mp4 ] || ffmpeg -v error -y -f lavfi -i "testsrc2=s=3840x2160:r=24:d=10,noise=alls=4:allf=t" -c:v libx264 -preset fast -b:v 50M -maxrate 60M -bufsize 100M -g 48 -bf 2 -pix_fmt yuv420p h264_4k_50m.mp4
# HEVC 10-bit, the usual phone / mirrorless camera format (the media engine encodes it).
[ -f hevc_4k_10bit.mov ] || ffmpeg -v error -y -f lavfi -i "testsrc2=s=3840x2160:r=24:d=10,noise=alls=4:allf=t" -c:v hevc_videotoolbox -b:v 60M -g 48 -tag:v hvc1 -pix_fmt p010le hevc_4k_10bit.mov
# 1080p H.264 with AAC audio and a start timecode (optimise must keep both).
[ -f h264_1080_audio_tc.mp4 ] || ffmpeg -v error -y -f lavfi -i "testsrc2=s=1920x1080:r=25:d=8" -f lavfi -i "sine=f=440:d=8" -c:v libx264 -preset fast -b:v 20M -g 50 -pix_fmt yuv420p -c:a aac -timecode 01:00:00:00 h264_1080_audio_tc.mp4
# The 4K ProRes with a tone on it (picture copied): sound + picture through cuts.
[ -f prores_hq_4k_audio.mov ] || ffmpeg -v error -y -i prores_hq_4k.mov -f lavfi -i "sine=f=440:d=10:sample_rate=48000" -c:v copy -c:a pcm_s16le -shortest prores_hq_4k_audio.mov
