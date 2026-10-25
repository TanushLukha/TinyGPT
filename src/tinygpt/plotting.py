"""One look for every chart in docs/results/: light surface, quiet axes, a fixed colour per model."""
from __future__ import annotations

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# categorical colours, in a fixed order checked for colour-blind separation
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# a model keeps its colour in every figure, whichever other models are shown with it
FIXED = {"lstm": 0, "gpt-llama": 1, "gpt": 2, "gpt-small": 3, "gpt-rope": 6}   # 6: passes next to 1 and 2


def run_colors(names) -> dict[str, str]:
    """Colour per run name: the fixed one if it has one, otherwise the next unused slot in order."""
    used = {FIXED[n] for n in names if n in FIXED}
    free = iter(i for i in range(len(SLOTS)) if i not in used)
    return {n: SLOTS[FIXED[n]] if n in FIXED else SLOTS[next(free)] for n in names}


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)


def spread_labels(values: list[float], gap: float) -> list[float]:
    """Label positions near `values` (any order) that are at least `gap` apart, keeping their order."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    pos = [0.0] * len(values)
    last = None
    for i in order:
        y = values[i] if last is None else min(values[i], last - gap)
        pos[i] = last = y
    return pos
