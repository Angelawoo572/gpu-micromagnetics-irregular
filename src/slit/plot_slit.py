#!/usr/bin/env python3
"""
plot_slit_spinwave.py — visualize spin-wave single-slit diffraction output.

Reads output.txt written by 2d_slit.cu and produces:
  1) slit_intensity.png
  2) slit_snapshot.png
  3) slit_lineout.png
  4) slit_animation.mp4 / slit_animation.gif

All outputs are saved to:
  ../../results/slit/
"""

import sys, os, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle


# ─── Output directory ────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../results/slit"))
os.makedirs(OUT_DIR, exist_ok=True)

def out_path(name):
    return os.path.join(OUT_DIR, name)


# ─── I/O ─────────────────────────────────────────────────────────────────────

def read_output(path="output.txt"):
    """
    Parse text output written by WriteFrame():
      # slit spinwave output
      # nx=768 ny=256 ng=256
      # screen_col=128 slit_lo=115 slit_hi=141 src_col=64
      # t=0.000000 nx=768 ny=256 ng=256
      mx my mz       <- one line per cell, row-major j*ng+i
      ...
      # t=5.000000 nx=768 ny=256 ng=256
      ...
    """
    nx = ny = ng = screen_col = slit_lo = slit_hi = src_col = None

    frames = []
    times  = []
    current_rows = []
    current_t    = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                # ── geometry header (first three comment lines) ──────────
                m = re.search(r"nx=(\d+)\s+ny=(\d+)\s+ng=(\d+)", line)
                if m and nx is None:
                    nx, ny, ng = int(m.group(1)), int(m.group(2)), int(m.group(3))

                m = re.search(
                    r"screen_col=(\d+)\s+slit_lo=(\d+)\s+slit_hi=(\d+)\s+src_col=(\d+)",
                    line)
                if m:
                    screen_col = int(m.group(1))
                    slit_lo    = int(m.group(2))
                    slit_hi    = int(m.group(3))
                    src_col    = int(m.group(4))

                # ── per-frame header ─────────────────────────────────────
                m = re.search(r"t=([\d.eE+\-]+)", line)
                if m:
                    # save previous frame if any
                    if current_rows and current_t is not None:
                        arr = np.array(current_rows, dtype=np.float64)
                        # arr shape: (ncell, 3) — take mz (col 2), reshape
                        mz = arr[:, 2].reshape(ny, ng)
                        frames.append(mz)
                        times.append(current_t)
                    current_t    = float(m.group(1))
                    current_rows = []
                continue

            # ── data line: mx my mz ──────────────────────────────────────
            vals = line.split()
            if len(vals) == 3:
                current_rows.append([float(v) for v in vals])

    # flush last frame
    if current_rows and current_t is not None:
        arr = np.array(current_rows, dtype=np.float64)
        mz = arr[:, 2].reshape(ny, ng)
        frames.append(mz)
        times.append(current_t)

    print(f"Read {len(frames)} frames  nx={nx} ny={ny} ng={ng}")
    print(f"  screen_col={screen_col}  slit=[{slit_lo},{slit_hi})  src_col={src_col}")
    print(f"Output dir: {OUT_DIR}")

    return {
        "nx": nx, "ny": ny, "ng": ng,
        "screen_col": screen_col,
        "slit_lo":    slit_lo,
        "slit_hi":    slit_hi,
        "src_col":    src_col,
        "frames":     np.array(frames),
        "times":      np.array(times),
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def overlay_screen(ax, d):
    sc = d["screen_col"]
    lo = d["slit_lo"]
    hi = d["slit_hi"]
    ny = d["ny"]

    ax.add_patch(Rectangle((sc - 0.5, -0.5), 1, lo + 0.5,
                            color="k", alpha=0.8))
    ax.add_patch(Rectangle((sc - 0.5, hi - 0.5), 1, ny - hi + 0.5,
                            color="k", alpha=0.8))


def overlay_pml(ax, d, pml=20):
    ng = d["ng"]
    ax.axvspan(0, pml, alpha=0.25, color="gray", label="PML absorber")
    ax.axvspan(ng - pml, ng, alpha=0.25, color="gray")


def time_avg_intensity(d, last_frac=0.4):
    """
    Average |mz - <mz>|^2 over last last_frac of frames.
    Subtract the local time-mean so static offsets do not dominate.
    """
    frames = d["frames"]
    n_avg = max(1, int(last_frac * len(frames)))
    tail = frames[-n_avg:]
    local_mean = np.mean(tail, axis=0)
    dmz = tail - local_mean[np.newaxis, :, :]
    return np.mean(dmz ** 2, axis=0)


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_intensity(d):
    intensity = time_avg_intensity(d)

    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = np.percentile(intensity, 99) or 0.01

    im = ax.imshow(
        intensity,
        origin="lower",
        cmap="hot",
        vmin=0,
        vmax=vmax,
        aspect="auto",
        extent=[0, d["ng"] - 1, 0, d["ny"] - 1],
    )

    overlay_screen(ax, d)
    overlay_pml(ax, d)

    ax.axvline(d["src_col"], color="cyan", lw=1, ls="--", label="source")
    plt.colorbar(im, ax=ax, label="|δmz|² (time-avg)")

    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    ax.set_title("Spin-wave single-slit diffraction — time-averaged intensity")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path("slit_intensity.png"), dpi=150)
    print(f"Wrote {out_path('slit_intensity.png')}")
    plt.close(fig)

    return intensity


def plot_snapshot(d):
    frame = d["frames"][-1]

    n_avg = max(1, int(0.2 * len(d["frames"])))
    local_mean = np.mean(d["frames"][-n_avg:], axis=0)
    delta = frame - local_mean

    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = np.max(np.abs(delta)) * 0.8 or 0.01

    im = ax.imshow(
        delta,
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        extent=[0, d["ng"] - 1, 0, d["ny"] - 1],
    )

    overlay_screen(ax, d)
    overlay_pml(ax, d)

    plt.colorbar(im, ax=ax, label="δmz (AC)")
    ax.set_title(f"Spin-wave snapshot (AC)  t = {d['times'][-1]:.1f}")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")

    fig.tight_layout()
    fig.savefig(out_path("slit_snapshot.png"), dpi=150)
    print(f"Wrote {out_path('slit_snapshot.png')}")
    plt.close(fig)


def plot_lineout(intensity, d, x_col=None):
    ng = d["ng"]
    sc = d["screen_col"]

    if x_col is None:
        x_col = (sc + ng) // 2

    x_col = min(x_col, ng - 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(intensity[:, x_col], lw=1.5)

    ax.set_xlabel("y (cells)")
    ax.set_ylabel("|δmz|² (time-avg)")
    ax.set_title(f"Diffraction lineout at x={x_col} ({x_col - sc} cells past screen)")
    ax.grid(True, alpha=0.3)

    ax.axvspan(d["slit_lo"], d["slit_hi"],
               alpha=0.15, color="green", label="slit")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path("slit_lineout.png"), dpi=150)
    print(f"Wrote {out_path('slit_lineout.png')}")
    plt.close(fig)


def make_animation(d, fps=12, max_frames=None):
    frames_raw = d["frames"]

    n_avg = max(1, int(0.2 * len(frames_raw)))
    local_mean = np.mean(frames_raw[-n_avg:], axis=0)
    frames = frames_raw - local_mean[np.newaxis, :, :]

    if max_frames:
        frames = frames[:max_frames]

    vmax = np.percentile(np.abs(frames), 99) or 0.01

    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(
        frames[0],
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        extent=[0, d["ng"] - 1, 0, d["ny"] - 1],
    )

    overlay_screen(ax, d)
    overlay_pml(ax, d)

    title = ax.set_title("")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    plt.colorbar(im, ax=ax, label="δmz (AC)")

    def update(k):
        im.set_data(frames[k])
        title.set_text(f"t = {d['times'][k]:.1f}")
        return [im, title]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
    )

    mp4_path = out_path("slit_animation.mp4")

    try:
        anim.save(mp4_path, writer="ffmpeg", fps=fps, dpi=120)
        print(f"Wrote {mp4_path}")
    except Exception as e:
        print(f"ffmpeg unavailable ({e}); saving GIF instead")
        gif_path = out_path("slit_animation.gif")
        anim.save(gif_path, writer="pillow", fps=fps)
        print(f"Wrote {gif_path}")

    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    path = next((a for a in sys.argv[1:] if not a.startswith("--")),
                "output.txt")
    no_anim = "--no-anim" in sys.argv

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    d = read_output(path)

    if len(d["frames"]) == 0:
        print("No frames found in output file.")
        sys.exit(1)

    intensity = plot_intensity(d)
    plot_snapshot(d)
    plot_lineout(intensity, d)

    if not no_anim:
        make_animation(d)
    else:
        print("(animation skipped — use without --no-anim to generate)")


if __name__ == "__main__":
    main()
