#!/usr/bin/env bash
# Render + optimize the MoreGPU Manim animation to a small GIF for the README.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv-manim/bin/activate
manim -qm --format=gif --media_dir /tmp/manim-out -o moregpu scripts/manim/system.py MoreGPUFlow
SRC=/tmp/manim-out/videos/system/720p30/moregpu.gif
ffmpeg -y -i "$SRC" -vf "fps=10,scale=600:-1:flags=lanczos,palettegen=max_colors=64" /tmp/pal.png
ffmpeg -y -i "$SRC" -i /tmp/pal.png -lavfi "fps=10,scale=600:-1:flags=lanczos,paletteuse=dither=none" /tmp/mg.gif
gifsicle -O3 --lossy=100 --colors 64 /tmp/mg.gif -o docs/assets/moregpu-manim.gif
echo "wrote docs/assets/moregpu-manim.gif ($(du -h docs/assets/moregpu-manim.gif | cut -f1))"
