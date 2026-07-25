"""Generate a clean MoreGPU 'flow' animation as a Lottie JSON (for lottie-web) + a small GIF.
Run in the venv:  python scripts/lottie/gen.py
"""
from lottie import objects, Point, Color
from lottie.exporters.core import export_lottie

W, H, FR, DUR = 640, 220, 30, 60  # 2s loop
an = objects.Animation(DUR)
an.frame_rate = FR
an.width, an.height = W, H

INK = Color(0.055, 0.066, 0.09)      # #0e1117
NODE = Color(0.16, 0.19, 0.28)       # brighter card so it reads
ACC = Color(0.39, 0.40, 0.95)        # indigo
GRN = Color(0.20, 0.83, 0.60)
YEL = Color(0.98, 0.75, 0.14)
GREY = Color(0.545, 0.596, 0.678)


def bg():
    layer = objects.ShapeLayer()
    an.add_layer(layer)
    g = layer.add_shape(objects.Group())
    r = g.add_shape(objects.Rect())
    r.size.value = Point(W, H)
    r.position.value = Point(W / 2, H / 2)
    g.add_shape(objects.Fill(INK))


def node(cx, cy, w, h, color):
    layer = objects.ShapeLayer()
    an.add_layer(layer)
    g = layer.add_shape(objects.Group())
    r = g.add_shape(objects.Rect())
    r.size.value = Point(w, h)
    r.position.value = Point(cx, cy)
    r.rounded.value = 10
    g.add_shape(objects.Fill(NODE))
    st = g.add_shape(objects.Stroke(color, 2.5))
    return layer


def dot(x0, y0, x1, y1, color, delay, r=9):
    layer = objects.ShapeLayer()
    an.add_layer(layer)
    g = layer.add_shape(objects.Group())
    el = g.add_shape(objects.Ellipse())
    el.size.value = Point(r * 2, r * 2)
    # travel x0->x1 over the loop, staggered by delay; fade at ends
    el.position.add_keyframe(delay, Point(x0, y0))
    el.position.add_keyframe(delay + 26, Point(x1, y1))
    g.add_shape(objects.Fill(color))
    tr = g.transform
    tr.opacity.add_keyframe(delay, 0)
    tr.opacity.add_keyframe(delay + 3, 100)
    tr.opacity.add_keyframe(delay + 24, 100)
    tr.opacity.add_keyframe(delay + 27, 0)


# bg() removed — transparent so nodes/dots are visible over the page card
# nodes: admin, coordinator, 3 workers, pool
node(70, H / 2, 90, 46, ACC)      # admin
node(220, H / 2, 100, 52, ACC)    # coordinator
node(410, 55, 110, 40, GRN)       # worker gpu
node(410, H / 2, 110, 40, GRN)    # worker gpu
node(410, 165, 110, 40, YEL)      # worker cpu
node(570, H / 2, 64, 52, GRN)     # pool

# flow: admin->coord
dot(120, H / 2, 175, H / 2, ACC, 0)
# coord->workers (sealed shards, orange-ish accent)
dot(275, H / 2, 358, 55, YEL, 8)
dot(275, H / 2, 358, H / 2, YEL, 10)
dot(275, H / 2, 358, 165, YEL, 12)
# workers->pool (results)
dot(462, 55, 542, H / 2, GRN, 26)
dot(462, H / 2, 542, H / 2, GRN, 28)
dot(462, 165, 542, H / 2, YEL, 30)

export_lottie(an, "docs/assets/moregpu.lottie.json")
print("wrote docs/assets/moregpu.lottie.json")
