"""Reproduce the 3-D geometry figures of the Normalization post.

Draws N inputs x = z + c*1 with z ~ N(0, I_3), together with their L2,
RMSNorm and LayerNorm outputs (no affine parameters, eps = 1e-5).

Usage:
    python norm_geometry.py            # writes norm_geometry_c0.pdf and norm_geometry_c2.pdf
    python norm_geometry.py --png      # also writes PNG versions for the blog
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

D, N, EPS, SEED = 3, 100, 1e-5, 0
COLORS = {"input": "#444444", "l2": "#2a7bde", "rms": "#1aab78", "ln": "#ee6a33"}
# Axis ticks and limits for each shift c.
SETTINGS = {0: dict(ticks=[-2.5, 0.0, 2.5], lim=3.2), 2: dict(ticks=[-4, 0, 4], lim=5.2)}

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
})


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def rmsnorm(X):
    return X / np.sqrt((X ** 2).mean(axis=1, keepdims=True) + EPS)


def layernorm(X):
    mu = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True)
    return (X - mu) / np.sqrt(var + EPS)


def ln_circle(n=200):
    """Circle of radius sqrt(d) in the plane orthogonal to 1: the LayerNorm output set for d = 3."""
    u = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    v = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
    t = np.linspace(0, 2 * np.pi, n)
    return np.sqrt(D) * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))


def plot(c, Z, out):
    X = Z + c
    cfg = SETTINGS[c]
    fig = plt.figure(figsize=(3.2, 2.7))
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect((1, 1, 1), zoom=1.1)
    ax.view_init(elev=15, azim=20)

    s = 3.5
    ax.scatter(*X.T, s=s, color=COLORS["input"], label=r"Input $\boldsymbol{x}$", depthshade=False)
    ax.scatter(*l2(X).T, s=s, color=COLORS["l2"], label="L2", depthshade=False)
    ax.scatter(*rmsnorm(X).T, s=s, color=COLORS["rms"], label="RMSNorm", depthshade=False)
    ax.scatter(*layernorm(X).T, s=s, color=COLORS["ln"], label="LayerNorm", depthshade=False)
    ax.plot(*ln_circle().T, color=COLORS["ln"], lw=0.8, ls="--", alpha=0.7)
    t = np.array([-cfg["lim"], cfg["lim"]])
    ax.plot(t, t, t, color="black", lw=0.9, ls=":", label=r"$\mathbf{1}$ direction")

    for axis, name in zip((ax.xaxis, ax.yaxis, ax.zaxis), ("$x_1$", "$x_2$", "$x_3$")):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#dddddd")
        axis._axinfo["grid"].update(color="#e6e6e6", linewidth=0.5)
        axis.set_ticks(cfg["ticks"])
        axis.set_label_text(name)
        axis.labelpad = -4
    ax.tick_params(pad=-2)
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation(0)
    ax.set_xlim(-cfg["lim"], cfg["lim"])
    ax.set_ylim(-cfg["lim"], cfg["lim"])
    ax.set_zlim(-cfg["lim"], cfg["lim"])

    ax.set_title(f"$c$ = {c}", y=1.07, fontsize=11)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=5, frameon=False,
               handletextpad=0.2, columnspacing=0.8, markerscale=1.5, fontsize=8)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.2, dpi=500)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", action="store_true", help="also save PNG files")
    args = parser.parse_args()

    # The same z is used for every c, so the LayerNorm outputs are identical across figures.
    Z = np.random.default_rng(SEED).standard_normal((N, D))
    for c in SETTINGS:
        plot(c, Z, f"norm_geometry_c{c}.pdf")
        if args.png:
            plot(c, Z, f"norm_geometry_c{c}.png")


if __name__ == "__main__":
    main()
