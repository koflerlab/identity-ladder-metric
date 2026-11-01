"""Identity ladder as a picture: one column per rung, the person on top, the compared image below.

Faces are drawn, not photographed: no real person appears. Each face is built from the same variance components as
the tissue model, z = sqrt(h) * genotype + sqrt(1 - h) * development, so a sibling shares half the genotype, a
stranger shares nothing, and the population mean is the pixel average of many drawn people. Session (pose, light,
shirt), camera (colour cast, blur) and ageing are applied on top. RMSE to the person inside a box over one brow and
eye (the analogue of an inpainting mask) is printed under each column; the bilateral image is the mirrored photo, so
its box holds the other eye and only side-specific development differs.
Output: results/figures/identity_ladder_faces.{png,pdf}.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

PROJECT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SEED, N_MEAN, N_DRAWS = 7, 80, 25
# x0, x1, y0, y1 as fractions of width and height (from the top): one brow and eye, so that in the mirrored image the
# box holds the other side's brow and eye and only side-specific development differs
BOX = (0.30, 0.48, 0.355, 0.47)
BG = "#ecebe7"
INK, MUTED, ORANGE = "#0b0b0b", "#52514e", "#eb6834"
SKIN = ["#f6d7c3", "#eac0a0", "#d9a07a", "#b97a55", "#8d5a3b", "#5e3b26"]
HAIR = ["#1d1a18", "#3b2a20", "#5c3b25", "#8a5a33", "#b08850", "#d4b57a"]
IRIS = ["#4b2e1a", "#6b4f2a", "#5a7d4f", "#4f7fa8", "#7c8a96"]
SHIRTS = ["#3d5a80", "#9a4c3c", "#4a7a52", "#6b5b95", "#c08a3e", "#5f7f8f", "#7f4f6f", "#2f4f4f", "#a05a7a"]
# (name, centre, half-range, genetic share h): z -> centre + half_range * tanh(z / 1.6)
TRAITS = [("face_w", 0.235, 0.04, 0.9), ("face_h", 0.30, 0.035, 0.9), ("jaw", 0.30, 0.2, 0.9),
          ("skin", 0.45, 0.45, 0.98), ("hair_col", 0.5, 0.5, 0.98), ("iris", 0.5, 0.5, 0.98),
          ("hair_len", 0.08, 0.14, 0.3), ("hair_vol", 0.03, 0.015, 0.7), ("hairline", 0.68, 0.1, 0.8),
          ("eye_dx", 0.085, 0.014, 0.9), ("eye_y", 0.045, 0.018, 0.9), ("eye_size", 0.03, 0.006, 0.9),
          ("brow_w", 3.2, 1.4, 0.9), ("brow_tilt", 0.0, 0.012, 0.9), ("brow_gap", 0.045, 0.01, 0.9),
          ("nose_len", 0.08, 0.02, 0.9), ("nose_w", 0.028, 0.008, 0.9), ("mouth_w", 0.058, 0.014, 0.9),
          ("mouth_y", -0.125, 0.018, 0.9), ("smile", 0.010, 0.008, 0.5), ("lip", 0.012, 0.006, 0.9),
          ("ear", 0.048, 0.01, 0.9),
          # side-specific development: no genetic share, swaps sides in the mirrored image
          ("asym_eye", 0.0, 0.06, 0.0), ("asym_brow", 0.0, 0.005, 0.0), ("asym_mouth", 0.0, 0.006, 0.0),
          ("part", 0.0, 1.0, 0.0), ("mole_x", 0.0, 0.13, 0.0), ("mole_y", -0.05, 0.1, 0.0)]


def draw_genome(rng):
    return {"g": rng.standard_normal(len(TRAITS)), "e": rng.standard_normal(len(TRAITS))}


def sibling_genome(person, rng):
    g = 0.5 * person["g"] + np.sqrt(0.75) * rng.standard_normal(len(TRAITS))
    return {"g": g, "e": rng.standard_normal(len(TRAITS))}


def traits(person):
    out = {}
    for i, (name, mid, half, h) in enumerate(TRAITS):
        z = np.sqrt(h) * person["g"][i] + np.sqrt(1 - h) * person["e"][i]
        out[name] = mid + half * np.tanh(z / 1.6)
    return out


def palette(colours, t):
    t = float(np.clip(t, 0, 1)) * (len(colours) - 1)
    i = min(int(t), len(colours) - 2)
    a, b = np.array(to_rgb(colours[i])), np.array(to_rgb(colours[i + 1]))
    return tuple(a + (b - a) * (t - i))


def shade(c, f):
    return tuple(np.clip(np.array(c) * f, 0, 1))


def render(p, sess, age=0.0, size=(2.0, 2.4), dpi=110):
    """Draw one face; sess holds pose (dx, dy, rot), shirt colour and light. Returns RGB floats in [0, 1]."""
    fig = plt.figure(figsize=size, dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)
    ax.axis("off")
    cx, cy = 0.5, 0.66
    T = Affine2D().rotate_deg_around(cx, cy, sess.get("rot", 0)).translate(sess.get("dx", 0), sess.get("dy", 0))
    T = T + ax.transData
    skin = palette(SKIN, p["skin"] + 0.08 * age)
    hair = np.array(palette(HAIR, p["hair_col"]))
    hair = tuple(hair + (np.array(to_rgb("#c9c6c0")) - hair) * min(1.0, 0.85 * age))
    a, b = p["face_w"], p["face_h"]

    if p["hair_len"] > 0.08:  # long hair hangs behind the neck and shoulders
        long = 1.6 * p["hair_len"]
        ax.add_patch(Polygon([(cx - a - p["hair_vol"], cy), (cx + a + p["hair_vol"], cy),
                              (cx + a + 0.02, cy - b - long), (cx - a - 0.02, cy - b - long)], color=hair,
                             transform=T))
    ax.add_patch(Ellipse((0.5, 0.02), 0.95, 0.42, color=sess["shirt"]))  # shoulders stay in the frame
    ax.add_patch(Polygon([(cx - 0.075, cy - b * 0.7), (cx + 0.075, cy - b * 0.7), (cx + 0.085, 0.2),
                          (cx - 0.085, 0.2)], color=shade(skin, 0.9), transform=T))
    for s in (-1, 1):
        ax.add_patch(Ellipse((cx + s * a * 0.97, cy + 0.0), p["ear"] * 0.9, p["ear"] * 1.7, color=shade(skin, 0.93),
                             transform=T))
    t = np.linspace(0, 2 * np.pi, 160)
    w = np.where(np.sin(t) < 0, 1 - p["jaw"] * np.sin(t) ** 4, 1.0)
    ax.add_patch(Polygon(np.c_[cx + a * np.cos(t) * w, cy + b * np.sin(t)], color=skin, transform=T))
    for s in (-1, 1):  # cheeks
        ax.add_patch(Ellipse((cx + s * a * 0.55, cy - b * 0.25), 0.07, 0.04, color="#e07a6a", alpha=0.12,
                             transform=T))

    hv = p["hair_vol"]
    ta = np.linspace(-0.12, np.pi + 0.12, 60)
    outer = np.c_[cx + (a + hv) * np.cos(ta), cy + (b + hv) * np.sin(ta)]
    hl = p["hairline"] + 0.18 * age
    xs = np.linspace(cx - a * 0.85, cx + a * 0.85, 30)
    side = 1 if p["part"] >= 0 else -1
    ys = cy + b * hl + side * 0.15 * b * (xs - cx) / a - 0.02 * np.exp(-((xs - cx - side * a * 0.3) / 0.05) ** 2)
    ys = np.minimum(ys, cy + 0.95 * b * np.sqrt(np.clip(1 - ((xs - cx) / a) ** 2, 0, 1)))  # stay inside the head
    inner = np.r_[[[cx - a * 0.99, cy + b * 0.1]], np.c_[xs, ys], [[cx + a * 0.99, cy + b * 0.1]]]
    ax.add_patch(Polygon(np.r_[outer, inner], color=hair, transform=T))

    brow = shade(np.array(palette(HAIR, p["hair_col"])), 0.9)
    for s in (-1, 1):
        ex = cx + s * p["eye_dx"]
        ey = cy + p["eye_y"]
        r = p["eye_size"] * (1 + s * p["asym_eye"])
        by = ey + p["brow_gap"] + s * p["asym_brow"] / 2
        bx = np.linspace(-1, 1, 20)
        ax.plot(ex + s * bx * 0.038, by + 0.01 * (1 - bx ** 2) + p["brow_tilt"] * bx, color=brow,
                lw=p["brow_w"], solid_capstyle="round", transform=T)
        ax.add_patch(Ellipse((ex, ey), 2.1 * r, 1.25 * r, facecolor="white", edgecolor=shade(skin, 0.55), lw=0.8,
                             transform=T))
        ax.add_patch(Circle((ex, ey), 0.5 * r, color=palette(IRIS, p["iris"]), transform=T))
        ax.add_patch(Circle((ex, ey), 0.22 * r, color="#111111", transform=T))
        ax.add_patch(Circle((ex, ey + 0.2 * r), 0.09 * r, color="white", transform=T))  # centred: mirror-safe
        if age > 0.3:  # crow's feet
            for k in (-1, 0, 1):
                ax.plot([ex + s * 1.2 * r, ex + s * 1.75 * r], [ey + k * 0.25 * r, ey + k * 0.45 * r],
                        color=shade(skin, 0.72), lw=0.6, alpha=age, transform=T)

    ny = cy + p["eye_y"]
    nb = ny - p["nose_len"]
    ax.plot([cx + 0.004, cx + 0.008, cx + p["nose_w"] * 0.5], [ny - 0.01, nb + 0.012, nb], color=shade(skin, 0.7),
            lw=1.2, transform=T)
    ax.plot([cx - p["nose_w"] * 0.6, cx, cx + p["nose_w"] * 0.6], [nb + 0.004, nb - 0.006, nb + 0.004],
            color=shade(skin, 0.7), lw=1.2, transform=T)

    my, mw = cy + p["mouth_y"], p["mouth_w"]
    u = np.linspace(-1, 1, 30)
    upper = my + p["smile"] * u ** 2 + p["asym_mouth"] * u
    lower = upper - p["lip"] * (1 - u ** 2) * 1.6
    ax.add_patch(Polygon(np.r_[np.c_[cx + mw * u, upper + p["lip"] * 0.7 * (1 - u ** 2)],
                               np.c_[cx + mw * u, lower][::-1]], color=shade(palette(SKIN, p["skin"]), 0.78),
                         transform=T))
    ax.plot(cx + mw * u, upper, color=shade(skin, 0.45), lw=0.9, transform=T)
    if abs(p["mole_x"]) > 0.03:
        ax.add_patch(Circle((cx + p["mole_x"], cy + p["mole_y"]), 0.006, color="#3a2418", transform=T))
    if age > 0.3:  # forehead and nasolabial lines
        for k in range(3):
            fy = cy + b * (0.42 + 0.09 * k)
            ax.plot(np.linspace(cx - a * 0.4, cx + a * 0.4, 20), fy + 0.006 * np.sin(np.linspace(0, np.pi, 20)),
                    color=shade(skin, 0.75), lw=0.6, alpha=age, transform=T)
        for s in (-1, 1):
            ax.plot([cx + s * p["nose_w"] * 0.9, cx + s * (mw + 0.012)], [nb - 0.005, my - 0.012],
                    color=shade(skin, 0.75), lw=0.7, alpha=age, transform=T)

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float) / 255
    plt.close(fig)
    h, w = img.shape[:2]
    gx = np.linspace(-0.5, 0.5, w)[None, :, None]
    return np.clip(img * (sess.get("light", 1.0) + sess.get("grad", 0.0) * gx), 0, 1)


def camera(img):
    """Another phone camera: warm cast, higher contrast, softer optics, vignetting."""
    out = img * np.array([1.07, 1.0, 0.84])
    out = np.clip((out - 0.5) * 1.18 + 0.5, 0, 1) ** 0.9
    out = np.stack([gaussian_filter(out[..., c], 1.3) for c in range(3)], axis=-1)
    h, w = out.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    return np.clip(out * (1 - 0.18 * r2[..., None]), 0, 1)


def noisy(img, rng, sigma=0.03):
    grain = rng.normal(0, sigma, img.shape[:2])[..., None] + rng.normal(0, sigma / 3, img.shape)
    return np.clip(img + grain, 0, 1)


def face_box(img):
    """Pixels inside BOX, one brow and eye: the analogue of the masked region."""
    h, w = img.shape[:2]
    x0, x1, y0, y1 = BOX
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def rmse(a, b):
    return float(np.sqrt(np.mean((face_box(a) - face_box(b)) ** 2)))


def median_draw(make, top, rng, k=None):
    """Draw k people for one rung and keep the one with the median RMSE, so the column shows a typical relative."""
    k = k or N_DRAWS
    imgs = [make(rng) for _ in range(k)]
    errs = [rmse(top, im) for im in imgs]
    return imgs[int(np.argsort(errs)[k // 2])]


def build(seed=SEED):
    rng = np.random.default_rng(seed)
    person = draw_genome(rng)
    p = traits(person)
    base = {"shirt": SHIRTS[0]}
    clean = render(p, base)
    top = noisy(clean, rng)
    pool = [render(traits(draw_genome(rng)), {"shirt": SHIRTS[rng.integers(len(SHIRTS))],
                                              "dx": rng.normal(0, 0.005), "rot": rng.normal(0, 1)})
            for _ in range(N_MEAN)]
    cols = [
        ("L0", "same person,\nsame session", "acquisition\nnoise only", noisy(clean, rng)),
        ("L0'", "same person,\nanother day", "noise, session\n(pose, light)",
         noisy(render(p, {"shirt": SHIRTS[1], "rot": 1.5, "dx": 0.004, "light": 1.03, "grad": 0.12}), rng)),
        ("L1 *", "same person,\nother camera", "noise, session,\ndevice",
         noisy(camera(render(p, {"shirt": SHIRTS[2], "rot": -1})), rng)),
        ("L2", "same person,\nyears later", "noise, session,\nageing",
         noisy(render(p, {"shirt": SHIRTS[3], "rot": 1}, age=0.8), rng)),
        ("bilateral", "same person,\nother side", "side-specific\ndevelopment", noisy(clean[:, ::-1], rng)),
        ("MZ", "identical twin\n(same genome)", "development,\nside, session", median_draw(
            lambda r: noisy(render(traits({"g": person["g"], "e": r.standard_normal(len(TRAITS))}),
                                   {"shirt": SHIRTS[4], "rot": -1}), r), top, rng)),
        ("DZ / sibling", "brother or sister\n(half the genome)", "half the genome,\ndevelopment, side", median_draw(
            lambda r: noisy(render(traits(sibling_genome(person, r)), {"shirt": SHIRTS[5], "rot": 1}), r), top, rng)),
        ("L4", "unrelated\nperson", "everything", median_draw(
            lambda r: noisy(render(traits(draw_genome(r)), {"shirt": SHIRTS[6], "rot": -1}), r), top, rng)),
        ("population\nmean", f"average of\n{N_MEAN} other people", "the whole\nindividual part",
         np.mean(pool, axis=0)),
    ]
    return top, cols


def main():
    top, cols = build()
    n = len(cols)
    fig = plt.figure(figsize=(1.55 * n, 5.6))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, n, left=0.055, right=0.995, top=0.70, bottom=0.215, wspace=0.06, hspace=0.04)
    errors = {}
    h, w = top.shape[:2]
    x0, x1, y0, y1 = BOX
    for j, (rung, pair, diff, img) in enumerate(cols):
        mean = rung.startswith("population")
        colour = ORANGE if mean else INK
        key = rung.replace("\n", " ")
        for i, im in enumerate((top, img)):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(im, interpolation="lanczos")
            ax.add_patch(plt.Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h, fill=False, ls=(0, (2, 2)),
                                       lw=0.9, ec=ORANGE if mean else "#33322f"))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(ORANGE if (mean and i == 1) else "#c9c7c1")
                sp.set_linewidth(1.6 if (mean and i == 1) else 0.6)
            if j == 0:
                ax.set_ylabel("the person" if i == 0 else "compared with", fontsize=9.5, color=MUTED)
        errors[key] = rmse(top, img)
        pos = gs[0, j].get_position(fig)
        xc = (pos.x0 + pos.x1) / 2
        fig.text(xc, 0.785, rung, ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=colour,
                 linespacing=0.95)
        fig.text(xc, 0.715, pair, ha="center", va="bottom", fontsize=8.3, color=colour)
        low = gs[1, j].get_position(fig)
        fig.text(xc, low.y0 - 0.012, f"differs:\n{diff}", ha="center", va="top", fontsize=7.8, color=MUTED)
        fig.text(xc, low.y0 - 0.115, f"RMSE {errors[key]:.3f}", ha="center", va="top", fontsize=8.8, color=colour,
                 fontweight="bold" if mean else "normal")
    groups = [("same person", 0, 4), ("relatives", 5, 6), ("no shared identity", 7, 8)]
    for label, j0, j1 in groups:
        gx0 = gs[0, j0].get_position(fig).x0 + 0.004
        gx1 = gs[0, j1].get_position(fig).x1 - 0.004
        y = 0.91
        fig.add_artist(plt.Line2D([gx0, gx1], [y, y], color=MUTED, lw=0.9))
        for xe in (gx0, gx1):
            fig.add_artist(plt.Line2D([xe, xe], [y, y - 0.015], color=MUTED, lw=0.9))
        fig.text((gx0 + gx1) / 2, y + 0.01, label, ha="center", va="bottom", fontsize=9.5, color=MUTED)
    fig.text(0.055, 0.01, "Drawn faces, no real person. * the camera changes, so strangers would be matched on the "
             "same camera pair.\nRMSE to the image of the person inside the dashed box (pixels in [0, 1]). Twin, "
             f"sibling and stranger: the median of {N_DRAWS} drawn people by RMSE.",
             fontsize=7.8, color=MUTED, ha="left", va="bottom")
    out = os.path.join(PROJECT, "results", "figures", "identity_ladder_faces")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=170, facecolor="white")
    plt.close(fig)
    print({k: round(v, 4) for k, v in errors.items()})


if __name__ == "__main__":
    main()
