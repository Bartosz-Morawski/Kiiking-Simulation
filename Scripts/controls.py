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
    Compute the effective length r(θ, ω) using a "soft-sign" control strategy.

    This strategy uses raw theta (for first-swing optimization) and a steep
    tanh(C*omega) function to approximate a sign switch, forcing the athlete
    to squat/stand aggressively even at low speeds.

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
        Gain controlling the coupling strength.
    r_min, r_max
        Physical limits on r (standing/squatting bounds).

    Returns
    -------
    r
        Effective length (meters), clipped to [r_min, r_max].

    Notes
    -----
    Implements:
        r(θ, ω) = r0 - A * tanh( z )
        where z = k * theta * tanh(C * omega)
        and C = 100 (hardcoded steepness factor).
    """
    C = 5
    sign_omega = np.tanh(C * omega)
    z = k * theta * sign_omega
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
    Compute r and its partial derivatives for the soft-sign control law.

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
    For r = r0 - A * tanh(z), where z = k * theta * tanh(C * omega):

        ∂r/∂θ = -A * sech^2(z) * [k * tanh(C * omega)]
        ∂r/∂ω = -A * sech^2(z) * [k * theta * C * sech^2(C * omega)]

    This includes the chain rule factor 'C' for the inner tanh, which ensures
    the inertial terms are calculated correctly by the solver.
    """
    C = 5

    # Precompute inner terms
    tanh_C_om = np.tanh(C * omega)
    sech2_C_om = 1.0 - tanh_C_om**2  # Identity: sech^2(x) = 1 - tanh^2(x)

    z = k * theta * tanh_C_om

    tanh_z = np.tanh(z)
    sech2_z = 1.0 - tanh_z**2

    r_unclipped = r0 - A * tanh_z
    r = float(np.clip(r_unclipped, r_min, r_max))

    if r != r_unclipped:
        # saturated: r no longer changes with theta/omega
        return r, 0.0, 0.0

    # Chain rule implementation
    dr_dtheta = float(-A * sech2_z * k * tanh_C_om)
    dr_domega = float(-A * sech2_z * k * theta * C * sech2_C_om)

    return r, dr_dtheta, dr_domega