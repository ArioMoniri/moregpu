#!/usr/bin/env bash
# Render + optimize the MoreGPU Manim animation to a small, high-contrast GIF for the README.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv-manim/bin/activate
manim -qm --format=gif --media_dir /tmp/manim-out -o moregpu scripts/manim/system.py MoreGPUFlow
SRC=/tmp/manim-out/videos/system/720p30/moregpu.gif
ffmpeg -y -i "$SRC" -vf "fps=12,scale=700:-1:flags=lanczos,palettegen=max_colors=128:stats_mode=diff" /tmp/pal.png
ffmpeg -y -i "$SRC" -i /tmp/pal.png -lavfi "fps=12,scale=700:-1:flags=lanczos,paletteuse=dither=sierra2_4a" /tmp/mg.gif
gifsicle -O3 --lossy=45 --colors 128 /tmp/mg.gif -o docs/assets/moregpu-manim.gif
echo "wrote docs/assets/moregpu-manim.gif ($(du -h docs/assets/moregpu-manim.gif | cut -f1))"
