#!/usr/bin/env python3
"""
plot_slit.py — visualize the output of slit_fdtd.

Reads output.bin (binary, written by slit_fdtd.cu) and produces:
  1) slit_intensity.png  — time-averaged |Ez|^2 over the last few periods
                            (this is the diffraction pattern your professor wants)
  2) slit_animation.mp4   — animation of Ez(x, y, t)        (optional)

Header of output.bin (in order):
    int    nx
    int    ny
    int    n_frames
    double dx
    double dt_frame
    int    screen_col
    int    slit_lo
    int    slit_hi
Then n_frames repetitions of:
    double t
    int    nx
    int    ny
    double Ez[ny * nx]    (row-major, j*nx + i)
"""

import struct
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle


def read_output(path="output.bin"):
    with open(path, "rb") as fp:
        nx, ny, nf = struct.unpack("iii", fp.read(12))
        dx, dt_frame = struct.unpack("dd", fp.read(16))
        screen_col, slit_lo, slit_hi = struct.unpack("iii", fp.read(12))

        frames = []
        times = []
        for _ in range(nf):
            t, = struct.unpack("d", fp.read(8))
            fnx, fny = struct.unpack("ii", fp.read(8))
            assert fnx == nx and fny == ny
            buf = fp.read(8 * nx * ny)
            arr = np.frombuffer(buf, dtype=np.float64).reshape(ny, nx).copy()
            frames.append(arr)
            times.append(t)
    return {
        "nx": nx, "ny": ny, "n_frames": nf,
        "dx": dx, "dt_frame": dt_frame,
        "screen_col": screen_col,
        "slit_lo": slit_lo, "slit_hi": slit_hi,
        "frames": np.array(frames),  # shape (nf, ny, nx)
        "times": np.array(times),
    }


def overlay_screen(ax, data):
    sc = data["screen_col"]; lo = data["slit_lo"]; hi = data["slit_hi"]
    ny = data["ny"]
    # PEC screen: draw two thin black bars covering everything except the slit
    ax.add_patch(Rectangle((sc - 0.5, -0.5),    1, lo + 0.5,           color="k", alpha=0.85))
    ax.add_patch(Rectangle((sc - 0.5, hi - 0.5), 1, ny - hi + 0.5,     color="k", alpha=0.85))


def plot_intensity(data, n_avg_periods=2.0, freq=1.0e9):
    """Time-average |Ez|^2 over the last ~n_avg_periods periods."""
    period   = 1.0 / freq
    dt_frame = data["dt_frame"]
    n_avg    = max(1, int(round(n_avg_periods * period / dt_frame)))
    n_avg    = min(n_avg, data["n_frames"])
    last     = data["frames"][-n_avg:]
    intensity = (last ** 2).mean(axis=0)  # shape (ny, nx)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(intensity, origin="lower", cmap="inferno", aspect="equal",
                   interpolation="bilinear")
    overlay_screen(ax, data)
    ax.set_title(f"Time-averaged |Ez|² (last {n_avg} frames ≈ {n_avg*dt_frame*freq:.1f} periods)")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    plt.colorbar(im, ax=ax, label="|Ez|²")
    fig.tight_layout()
    fig.savefig("slit_intensity.png", dpi=140)
    print("Wrote slit_intensity.png")
    plt.close(fig)
    return intensity


def plot_lineout(intensity, data, x_offset_cells=None):
    """Take a vertical lineout some distance behind the screen."""
    if x_offset_cells is None:
        # Default: mid-way between screen and right edge
        x_offset_cells = (data["nx"] + data["screen_col"]) // 2
    line = intensity[:, x_offset_cells]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(line, lw=1.5)
    ax.set_xlabel("y (cells)")
    ax.set_ylabel("|Ez|² (time-avg)")
    ax.set_title(f"Diffraction lineout at x = {x_offset_cells} cells "
                 f"({(x_offset_cells - data['screen_col']) * data['dx']:.2f} m past slit)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("slit_lineout.png", dpi=140)
    print("Wrote slit_lineout.png")
    plt.close(fig)


def make_animation(data, fps=15, max_frames=None):
    frames = data["frames"]
    if max_frames is not None:
        frames = frames[:max_frames]

    vmax = np.max(np.abs(frames)) * 0.7
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(frames[0], origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, aspect="equal", interpolation="bilinear")
    overlay_screen(ax, data)
    title = ax.set_title("")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    plt.colorbar(im, ax=ax, label="Ez")

    def update(k):
        im.set_data(frames[k])
        title.set_text(f"frame {k}/{len(frames)}   t = {data['times'][k]*1e9:.2f} ns")
        return [im, title]

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000/fps, blit=False)
    out = "slit_animation.mp4"
    try:
        anim.save(out, writer="ffmpeg", fps=fps, dpi=120)
        print(f"Wrote {out}")
    except Exception as e:
        print(f"ffmpeg not available ({e}); falling back to GIF")
        anim.save("slit_animation.gif", writer="pillow", fps=fps)
        print("Wrote slit_animation.gif")
    plt.close(fig)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "output.bin"
    data = read_output(path)
    print(f"nx={data['nx']}  ny={data['ny']}  n_frames={data['n_frames']}")
    print(f"dx={data['dx']:.4f} m   dt_frame={data['dt_frame']:.3e} s")
    print(f"screen_col={data['screen_col']}   slit=[{data['slit_lo']},{data['slit_hi']})")

    intensity = plot_intensity(data)
    plot_lineout(intensity, data)

    # Animation is optional — comment out if you only want the still frame.
    if "--no-anim" not in sys.argv:
        make_animation(data)


if __name__ == "__main__":
    main()
