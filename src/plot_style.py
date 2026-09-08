"""Small shared plotting style for paper-ready experiment figures."""

from __future__ import annotations

import matplotlib


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "dark": "#25313C",
    "mid": "#64748B",
    "light": "#CBD5E1",
    "pale": "#F1F5F9",
}


def apply_paper_style() -> None:
    """Apply a restrained, color-blind-safe style without extra dependencies.

    Only non-font defaults are set so matplotlib's default font is kept.
    """
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": COLORS["dark"],
        "axes.linewidth": 0.7,
        "axes.grid": False,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def try_add_basemap(ax, *, source: str = "hot", alpha: float = 0.9) -> bool:
    """Overlay a real-map tile background when ``contextily`` is available.

    ``source`` is ``"hot"`` (detailed OpenStreetMap HOT tiles) or
    ``"satellite"`` (Esri World Imagery). Returns True when tiles
    were added, False otherwise (plain background kept). Never raises:
    missing dependency or no network falls back silently with a console
    note. Axes are expected in lon/lat (EPSG:4326). Install with
    ``pip install contextily`` (or add it to ``environment.yml``) to enable.
    """
    try:
        import contextily as ctx
    except ImportError:
        print("Basemap skipped: contextily not installed (pip install contextily to enable)")
        return False
    try:
        provider = (
            ctx.providers.Esri.WorldImagery
            if source == "satellite" else ctx.providers.OpenStreetMap.HOT
        )
    except AttributeError:
        print(f"Basemap skipped: unknown tile source {source!r}")
        return False
    try:
        before = set(ax.texts)
        ctx.add_basemap(ax, crs="EPSG:4326", source=provider, alpha=float(alpha), zorder=0)
        # Keep the required tile attribution but move it out of the way:
        # contextily anchors it lower-left, where OD markers usually sit.
        for txt in ax.texts:
            if txt not in before:
                txt.set_position((0.99, 0.01))
                txt.set_ha("right")
                txt.set_va("bottom")
                txt.set_fontsize(5)
                txt.set_alpha(0.85)
        return True
    except Exception as exc:
        print(f"Basemap skipped (offline or tile error): {exc}")
        return False


def clean_axis(ax, *, grid_axis: str | None = "y") -> None:
    """Remove chart clutter and optionally add a subtle reference grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["light"], linewidth=0.55, alpha=0.55)
        ax.set_axisbelow(True)


def save_publication_figure(fig, output_path: str, *, dpi: int = 240) -> None:
    """Save a high-resolution raster figure."""
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
