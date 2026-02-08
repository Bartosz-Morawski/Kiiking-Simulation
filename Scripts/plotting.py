"""
Plotting and animation utilities for the functional Kiiking simulation.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks


def cartesian_from_solution(sol: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert solution dict to Cartesian (assignment convention).

    Parameters
    ----------
    sol
        Output from solve_kiiking_2state.

    Returns
    -------
    x, y
        x is vertical (positive down), y horizontal.
    """
    r = sol["r"]
    theta = sol["theta"]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def plot_summary(sol: dict) -> plt.Figure:
    """
    Quick 1x3 summary: total energy, amplitude peaks, r(t).

    Parameters
    ----------
    sol
        Output from solve_kiiking_2state.

    Returns
    -------
    fig
        Matplotlib figure.
    """
    t = sol["t"]
    theta = sol["theta"]
    r = sol["r"]
    E = sol["E"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Energy
    ax = axes[0]
    ax.plot(t, E)
    gain = (E[-1] - E[0]) / abs(E[0]) * 100 if E[0] != 0 else np.nan
    ax.set_title(f"Total Energy (gain {gain:+.1f}%)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("J")
    ax.grid(True)

    # Amplitude peaks
    ax = axes[1]
    th_abs = np.abs(theta)
    peaks, _ = find_peaks(th_abs, distance=50)
    if len(peaks) > 0:
        ax.plot(t[peaks], np.degrees(th_abs[peaks]), "o-")
    ax.set_title("Amplitude peaks")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|theta| (deg)")
    ax.grid(True)

    # r(t)
    ax = axes[2]
    ax.plot(t, r)
    ax.set_title("r(t)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("m")
    ax.grid(True)

    fig.tight_layout()
    return fig


def animate_solution(sol: dict, interval_ms: int = 10, trail_frames: int = 50) -> FuncAnimation:
    """
    Animate the motion for a solution dict with a running timer.

    Parameters
    ----------
    sol
        Output from solve_kiiking_2state.
    interval_ms
        Frame interval in milliseconds.
    trail_frames
        Number of previous frames to show in the trail.

    Returns
    -------
    anim
        Matplotlib animation.
    """
    x, y = cartesian_from_solution(sol)
    r = sol["r"]
    t = sol["t"]

    fig, ax = plt.subplots(figsize=(6, 6))
    rmax = float(np.max(r))

    # Set up the plot limits and aspect
    ax.set_xlim(-1.3 * rmax, 1.3 * rmax)
    ax.set_ylim(-1.3 * rmax, 1.3 * rmax)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # +x is down

    # Draw the pivot
    ax.plot(0, 0, "ko", markersize=10)

    # Initialize plot objects: rod, trail, and the timer text
    rod, = ax.plot([], [], "o-", lw=3)
    trail, = ax.plot([], [], "-", lw=1, alpha=0.4)

    # Place timer in top-left corner (0.05, 0.95 relative to axes)
    time_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    def init():
        rod.set_data([], [])
        trail.set_data([], [])
        time_text.set_text("")
        return rod, trail, time_text

    def update(i):
        # Update geometry
        rod.set_data([0, y[i]], [0, x[i]])

        # Update trail
        j0 = max(0, i - trail_frames)
        trail.set_data(y[j0:i + 1], x[j0:i + 1])

        # Update timer
        time_text.set_text(f"t = {t[i]:.2f} s")

        return rod, trail, time_text

    return FuncAnimation(fig, update, init_func=init, frames=len(t), interval=interval_ms, blit=True)
