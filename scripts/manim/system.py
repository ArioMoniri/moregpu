"""
MoreGPU — a 3blue1brown-style Manim animation of how the pool runs a job.
Render (in the venv):  manim -qm --format=gif -o moregpu scripts/manim/system.py MoreGPUFlow
Uses Text (no LaTeX dependency).
"""
from manim import (
    Scene, Text, RoundedRectangle, VGroup, Dot, Square, Arrow, Line,
    FadeIn, FadeOut, Create, MoveAlongPath, Indicate, Flash,
    LEFT, RIGHT, UP, DOWN, ORIGIN,
    BLUE_D, GREEN_D, YELLOW_D, GREY_B, WHITE, ORANGE, GREEN_B, PURPLE_B,
)

BG = "#0e1117"


def node(label, sub, color, w=2.4, h=1.0):
    box = RoundedRectangle(corner_radius=0.14, width=w, height=h,
                           stroke_color=color, stroke_width=2.5, fill_color="#161b26", fill_opacity=1)
    t = Text(label, font="Sans", weight="BOLD", color=WHITE).scale(0.34)
    s = Text(sub, font="Sans", color=GREY_B).scale(0.24)
    t.move_to(box.get_center() + UP * 0.14)
    s.move_to(box.get_center() + DOWN * 0.18)
    return VGroup(box, t, s)


class MoreGPUFlow(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("MoreGPU", font="Sans", weight="BOLD", color=WHITE).scale(0.7).to_edge(UP)
        sub = Text("submit → shard → seal → compute on GPU/CPU → pool → verify",
                   font="Sans", color=GREY_B).scale(0.32).next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(title, shift=DOWN * 0.2), FadeIn(sub))

        admin = node("Admin", "submit job", BLUE_D).move_to(LEFT * 5.2)
        coord = node("Coordinator", "shard · seal", PURPLE_B).move_to(LEFT * 2.2)
        w1 = node("worker · GPU", "WebGPU", GREEN_D, w=2.3, h=0.85).move_to(RIGHT * 1.6 + UP * 1.7)
        w2 = node("worker · GPU", "WebGPU", GREEN_D, w=2.3, h=0.85).move_to(RIGHT * 1.6)
        w3 = node("worker · CPU", "fallback", YELLOW_D, w=2.3, h=0.85).move_to(RIGHT * 1.6 + DOWN * 1.7)
        pool = node("Pool", "verify ✓", GREEN_D, w=1.9, h=1.0).move_to(RIGHT * 5.1)

        edges = VGroup(
            Line(admin.get_right(), coord.get_left(), stroke_color=GREY_B, stroke_width=2),
            Line(coord.get_right(), w1.get_left(), stroke_color=GREY_B, stroke_width=1.6),
            Line(coord.get_right(), w2.get_left(), stroke_color=GREY_B, stroke_width=1.6),
            Line(coord.get_right(), w3.get_left(), stroke_color=GREY_B, stroke_width=1.6),
            Line(w1.get_right(), pool.get_left(), stroke_color=GREY_B, stroke_width=1.6),
            Line(w2.get_right(), pool.get_left(), stroke_color=GREY_B, stroke_width=1.6),
            Line(w3.get_right(), pool.get_left(), stroke_color=GREY_B, stroke_width=1.6),
        )
        self.play(*[FadeIn(n) for n in (admin, coord, w1, w2, w3, pool)], Create(edges), run_time=1.2)

        # 1) job → coordinator
        job = Dot(color=PURPLE_B).scale(1.3).move_to(admin.get_right())
        self.play(MoveAlongPath(job, Line(admin.get_right(), coord.get_left())), run_time=0.8)
        self.play(Indicate(coord, color=PURPLE_B, scale_factor=1.1), FadeOut(job), run_time=0.5)

        # 2) sealed shards → workers
        seal_lbl = Text("🔒 sealed", font="Sans", color=ORANGE).scale(0.3).next_to(coord, DOWN, buff=0.35)
        self.play(FadeIn(seal_lbl))
        shards = [Square(0.16, color=ORANGE, fill_color=ORANGE, fill_opacity=1).move_to(coord.get_right()) for _ in range(3)]
        self.play(
            *[MoveAlongPath(s, Line(coord.get_right(), w.get_left())) for s, w in zip(shards, (w1, w2, w3))],
            run_time=1.0,
        )
        self.play(*[FadeOut(s) for s in shards],
                  Indicate(w1, color=GREEN_B), Indicate(w2, color=GREEN_B), Indicate(w3, color=YELLOW_D),
                  run_time=0.7)

        # 3) results → pool
        res = [Dot(color=c).move_to(w.get_right()) for w, c in ((w1, GREEN_B), (w2, GREEN_B), (w3, YELLOW_D))]
        self.play(*[MoveAlongPath(r, Line(w.get_right(), pool.get_left())) for r, w in zip(res, (w1, w2, w3))],
                  run_time=1.0)
        self.play(*[FadeOut(r) for r in res], Flash(pool, color=GREEN_D, flash_radius=1.0), run_time=0.7)

        cap = Text("outbound + join-token · AES-256-GCM sealed · duty adapts to each machine's load",
                   font="Sans", color=GREY_B).scale(0.3).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(1.2)
