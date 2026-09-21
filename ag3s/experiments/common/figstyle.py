"""Shared figure styling for the AG3S verification steps.

One place for the palette and the font so that every step's figures read as one document rather
than as a pile of separately-styled plots, and so that a colour is never re-picked by eye.

The palette is the validated categorical set: the slot order is the colour-vision-deficiency safety
mechanism, not decoration, so hues are assigned by slot index and never cycled. Sequential encodings
(a magnitude, like a rate) use the single blue ramp; identity encodings (which object) use the
categorical slots in fixed order.

Korean labels need a CJK face. Matplotlib registers the Noto CJK collection under its JP name even
though the same file carries Hangul, so the family is looked up rather than assumed -- on a machine
without it the figures fall back to English-only labels instead of drawing boxes.
"""

from __future__ import annotations

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID_INK = "#d8d7d2"

#: Sequential ramp, light -> dark, one hue. For magnitudes.
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

#: Categorical slots in fixed order. Assign by index; never cycle, never reorder per chart.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

_CJK_CANDIDATES = ("Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans CJK SC",
                   "NanumGothic", "Malgun Gothic", "AppleGothic")


def use_korean() -> bool:
    """Point matplotlib at a CJK face. Returns whether one was found."""
    import matplotlib
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
            # A CJK face has no proper minus glyph; without this every negative tick renders as a box.
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


def style_axes(fig, axes) -> None:
    """Recessive frame and ticks on the shared surface, for one or many axes."""
    import numpy as np

    fig.patch.set_facecolor(SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID_INK)
        ax.tick_params(colors=INK_2, labelsize=8, length=3, color=GRID_INK)
        ax.xaxis.label.set_color(INK_2)
        ax.yaxis.label.set_color(INK_2)


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


__all__ = ["CATEGORICAL", "GRID_INK", "INK", "INK_2", "SEQ_BLUE", "SURFACE",
           "sequential_cmap", "style_axes", "use_korean"]
