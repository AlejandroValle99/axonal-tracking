"""Reusable primitives for drawing UML-like sequence diagrams with matplotlib:
lifelines, messages (call/self-call), and loop/alt frames. Used by
generar_diagrama_pipeline_kymobutler.py and
generar_diagrama_pipeline_transformer.py.

Palette: fixed-order prefix of the validated categorical palette from the
dataviz skill (references/palette.md) — blue, aqua, yellow, green, violet.
Color encodes participant *family* (data/preprocessing, neural network,
classical algorithm, kinetics, output); structural chrome (loop/alt frames)
uses a neutral ink color instead, so it never competes with entity identity.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Polygon

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
BORDER = (0.106, 0.106, 0.106, 0.16)
FRAME_COLOR = "#6b6a63"  # neutral structural chrome, distinct from the categorical palette


def tint(hex_color: str, amount: float = 0.85):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r = r + (255 - r) * amount
    g = g + (255 - g) * amount
    b = b + (255 - b) * amount
    return (r / 255, g / 255, b / 255)


class SeqDiagram:
    def __init__(self, participants, box_w=2.05, spacing=2.55, header_h=0.85):
        """participants: list of (name, label, color_hex)"""
        self.participants = participants
        self.box_w = box_w
        self.spacing = spacing
        self.header_h = header_h
        self.xs = {name: 0.9 + i * spacing for i, (name, _, _) in enumerate(participants)}
        self.colors = {name: color for name, _, color in participants}
        self.y = header_h + 0.55
        self._frame_stack = []
        self._frames_to_draw = []

        self.fig, self.ax = plt.subplots(figsize=(spacing * len(participants), 10))
        self.ax.set_facecolor(SURFACE)
        self.fig.patch.set_facecolor(SURFACE)
        x_min = self.xs[participants[0][0]] - box_w / 2 - 0.4
        x_max = self.xs[participants[-1][0]] + box_w / 2 + 0.4
        self.ax.set_xlim(x_min, x_max)
        self.ax.axis("off")

        for name, label, color in participants:
            x = self.xs[name]
            box = FancyBboxPatch(
                (x - box_w / 2, 0),
                box_w,
                header_h,
                boxstyle="round,pad=0.05,rounding_size=0.08",
                linewidth=1.1,
                edgecolor=color,
                facecolor=tint(color, 0.82),
            )
            self.ax.add_patch(box)
            self.ax.text(
                x, header_h / 2, label,
                ha="center", va="center", fontsize=8.8, fontweight="bold", color=INK_PRIMARY,
                linespacing=1.3,
            )

    def _lifeline_stub(self, name, y0, y1):
        x = self.xs[name]
        self.ax.plot([x, x], [y0, y1], linestyle=(0, (2, 2)), linewidth=0.9, color=INK_MUTED, zorder=0)

    def gap(self, amount=0.18):
        self.y += amount

    def call(self, frm, to, label, sublabel=None, row_h=0.62):
        x0, x1 = self.xs[frm], self.xs[to]
        color = self.colors.get(to, INK_PRIMARY)
        self._lifeline_stub(frm, self.y - row_h, self.y)
        self._lifeline_stub(to, self.y - row_h, self.y)
        arrow = FancyArrowPatch(
            (x0, self.y), (x1, self.y),
            arrowstyle="-|>", mutation_scale=13, linewidth=1.3,
            color=color, shrinkA=0, shrinkB=2, zorder=3,
        )
        self.ax.add_patch(arrow)
        text_x = (x0 + x1) / 2
        va = "bottom"
        self.ax.text(text_x, self.y + 0.05, label, ha="center", va=va, fontsize=7.3, color=INK_PRIMARY, zorder=4)
        if sublabel:
            self.ax.text(text_x, self.y + 0.05 - 0.24, sublabel, ha="center", va=va, fontsize=6.5, color=INK_SECONDARY, style="italic", zorder=4)
            self.y += 0.22
        self.y += row_h

    def self_call(self, who, label, sublabel=None, row_h=0.72, loop_w=0.4):
        x = self.xs[who]
        color = self.colors.get(who, INK_PRIMARY)
        y0 = self.y - 0.08
        y1 = self.y + 0.34
        self._lifeline_stub(who, self.y - 0.1, self.y + row_h)
        arrow = FancyArrowPatch(
            (x, y0), (x, y1),
            connectionstyle="arc3,rad=1.5",
            arrowstyle="-|>", mutation_scale=11, linewidth=1.2,
            color=color, zorder=3,
        )
        self.ax.add_patch(arrow)
        self.ax.text(x + loop_w + 0.12, (y0 + y1) / 2, label, ha="left", va="center", fontsize=7.3, color=INK_PRIMARY, zorder=4)
        if sublabel:
            self.ax.text(x + loop_w + 0.12, (y0 + y1) / 2 - 0.22, sublabel, ha="left", va="center", fontsize=6.5, color=INK_SECONDARY, style="italic", zorder=4)
        self.y += row_h

    def open_frame(self, kind, label, span, color=FRAME_COLOR):
        """span: (first_participant_name, last_participant_name)"""
        x_left = self.xs[span[0]] - self.box_w / 2 - 0.18
        x_right = self.xs[span[1]] + self.box_w / 2 + 0.18
        handle = {"kind": kind, "label": label, "x_left": x_left, "x_right": x_right, "y_start": self.y - 0.32, "color": color}
        self._frame_stack.append(handle)
        self.y += 0.28
        return handle

    def close_frame(self, handle):
        handle["y_end"] = self.y + 0.1
        self._frames_to_draw.append(handle)
        self.y += 0.22

    def _draw_frames(self):
        for h in self._frames_to_draw:
            box = FancyBboxPatch(
                (h["x_left"], h["y_start"]), h["x_right"] - h["x_left"], h["y_end"] - h["y_start"],
                boxstyle="square,pad=0", linewidth=1.15, edgecolor=h["color"], facecolor="none",
                linestyle="-", zorder=1,
            )
            self.ax.add_patch(box)
            tab_w = 0.55 + 0.075 * len(h["kind"])
            tab_h = 0.26
            tab = Polygon(
                [
                    (h["x_left"], h["y_start"]),
                    (h["x_left"] + tab_w, h["y_start"]),
                    (h["x_left"] + tab_w, h["y_start"] + tab_h * 0.55),
                    (h["x_left"] + tab_w - 0.09, h["y_start"] + tab_h),
                    (h["x_left"], h["y_start"] + tab_h),
                ],
                closed=True, facecolor=tint(h["color"], 0.75), edgecolor=h["color"], linewidth=1.0, zorder=2,
            )
            self.ax.add_patch(tab)
            self.ax.text(
                h["x_left"] + tab_w / 2, h["y_start"] + tab_h / 2, h["kind"],
                ha="center", va="center", fontsize=6.6, fontweight="bold", color=INK_PRIMARY, zorder=3,
            )
            self.ax.text(
                h["x_left"] + tab_w + 0.12, h["y_start"] + tab_h / 2, h["label"],
                ha="left", va="center", fontsize=6.9, color=INK_SECONDARY, style="italic", zorder=3,
            )

    def finish(self, title, out_path, caption=None, legend=None):
        """legend: optional list of (label, color_hex) — one entry per family,
        since color here encodes identity shared across non-adjacent participants
        (e.g. two neural-net stages), which a legend must make explicit."""
        final_y = self.y + 0.4
        for name, _, _ in self.participants:
            self._lifeline_stub(name, self.header_h, final_y)
        self._draw_frames()
        legend_pad = 0.5 if legend else 0.0
        self.ax.set_ylim(final_y + (0.6 if caption else 0.15) + legend_pad, -0.3)  # inverted: time flows downward
        self.ax.text(
            (self.xs[self.participants[0][0]] + self.xs[self.participants[-1][0]]) / 2,
            -0.28, title, ha="center", va="bottom", fontsize=13.5, fontweight="bold", color=INK_PRIMARY,
        )
        if caption:
            self.ax.text(
                (self.xs[self.participants[0][0]] + self.xs[self.participants[-1][0]]) / 2,
                final_y + 0.55, caption, ha="center", va="top", fontsize=7.8, style="italic", color=INK_SECONDARY,
            )
        if legend:
            # axes-fraction coords are screen-space regardless of the inverted
            # data axis, so "lower center" + a small negative offset still
            # places this just below the rendered diagram.
            handles = [Patch(facecolor=tint(c, 0.82), edgecolor=c, label=lbl) for lbl, c in legend]
            leg = self.ax.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.045),
                ncol=len(handles),
                frameon=False,
                fontsize=7.8,
                handlelength=1.2,
                handleheight=1.2,
                labelcolor=INK_SECONDARY,
            )
            leg.set_zorder(5)
        self.fig.set_size_inches(self.spacing * len(self.participants), 2.4 + 0.62 * (final_y + legend_pad))
        self.fig.tight_layout()
        self.fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        print(f"Saved to {out_path}")
