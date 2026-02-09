"""
Control laws for Kiiking-style variable-length pendulum models.

This module provides r(theta, omega) and its partial derivatives with respect to
theta and omega. These derivatives are used to compute the angular acceleration
from the polar-coordinate equations of motion.

All functions here are designed for use with the two-state constrained model:
    r = f(theta, omega),  omega = dtheta/dt
"""

from __future__ import annotations
import numpy as np


def r_tanh_theta_omega(
    theta: float,
    omega: float,
    r0: float,
    A: float,
    k: float,
    r_min: float,
    r_max: float,
) -> float:
    """
    Compute the effective length r(θ, ω) using a smooth tanh control.

    Parameters
    ----------
    theta
        Angle θ in radians.
    omega
        Angular velocity ω = dθ/dt in rad/s.
    r0
        Baseline length (midpoint).
        For your formula, r0 = r_arm - 0.895.
    A
        Amplitude of length change (meters).
        For your formula, A = 0.145.
    k
        Gain controlling sharpness/strength of pumping.
    r_min, r_max
        Physical limits on r (standing/squatting bounds).

    Returns
    -------
    r
        Effective length (meters), clipped to [r_min, r_max].

    Notes
    -----
    Implements:
        r(θ, ω) = r0 - A * tanh(k sin(θ) ω)
    """
    sign_omega = np.tanh(100*omega)
    z = k * np.sin(theta) * sign_omega
    r = r0 - A * np.tanh(z)
    return float(np.clip(r, r_min, r_max))


def partials_tanh_theta_omega(
    theta: float,
    omega: float,
    r0: float,
    A: float,
    k: float,
    r_min: float,
    r_max: float,
) -> tuple[float, float, float]:
    """
    Compute r and its partial derivatives for the tanh( k θ ω ) control.

    Parameters
    ----------
    theta
        Angle θ in radians.
    omega
        Angular velocity ω in rad/s.
    r0, A, k, r_min, r_max
        As in :func:`r_tanh_theta_omega`.

    Returns
    -------
    r
        Effective length (meters), clipped to [r_min, r_max].
    dr_dtheta
        Partial derivative ∂r/∂θ (meters per radian).
    dr_domega
        Partial derivative ∂r/∂ω (meters per (rad/s)).

    Notes
    -----
    For the *unclipped* function r = r0 - A tanh(z), z = k θ ω:

        ∂r/∂θ = -A * sech^2(z) * k * ω
        ∂r/∂ω = -A * sech^2(z) * k * θ

    Clipping caveat
    -------------
    If r hits r_min or r_max, the physical model is "saturated".
    In saturation, the effective derivatives should be treated as 0 because r is
    no longer responding to θ, ω.

    This function enforces that: if clipping activates, derivatives are set to 0.
    """
    sign_omega = np.tanh(100 * omega)
    z = k * theta * sign_omega
    sech2 = 1.0 / (np.cosh(z) ** 2)
    sech2_2 = 1.0 / (np.cosh(sign_omega) ** 2)

    r_unclipped = r0 - A * np.tanh(z)
    r = float(np.clip(r_unclipped, r_min, r_max))

    if r != r_unclipped:
        # saturated: r no longer changes with theta/omega
        return r, 0.0, 0.0

    dr_dtheta = float(-A * sech2 * k * sign_omega) #CHANGED
    dr_domega = float(-A * sech2 * k * theta * sech2_2) #CHANGED
    return r, dr_dtheta, dr_domega
