"""
Control laws for Kiiking-style variable-length pendulum models.
Optimized for IMMEDIATE power generation (High Sensitivity).
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
    Compute effective length.
    Uses 'Double Saturation' to force full amplitude even at small angles.
    """
    # Sensitivity gains
    C_theta = 5.0  # High sensitivity to angle
    C_omega = 5.0  # High sensitivity to speed

    # 1. Detect if we are "Away from Bottom" (Saturated)
    # This turns into +/- 1 very quickly, fixing the "weak start" issue
    angle_trigger = np.tanh(C_theta * theta)

    # 2. Detect direction of motion
    speed_trigger = np.tanh(C_omega * omega)

    # 3. Combine triggers
    z = k * angle_trigger * speed_trigger

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
    Compute derivatives with Double Chain Rule (Theta and Omega).
    """
    C_theta = 5.0
    C_omega = 5.0

    # --- Precompute Inner Terms ---

    # Angle terms
    tanh_C_th = np.tanh(C_theta * theta)
    sech2_C_th = 1.0 - tanh_C_th**2

    # Speed terms
    tanh_C_om = np.tanh(C_omega * omega)
    sech2_C_om = 1.0 - tanh_C_om**2

    # --- Outer Control ---
    z = k * tanh_C_th * tanh_C_om

    tanh_z = np.tanh(z)
    sech2_z = 1.0 - tanh_z**2

    # --- Calculate r ---
    r_unclipped = r0 - A * tanh_z
    r = float(np.clip(r_unclipped, r_min, r_max))

    if r != r_unclipped:
        return r, 0.0, 0.0

    # --- Derivatives (Chain Rule) ---

    # 1. dr/dtheta
    # Inner derivative: d/dtheta [tanh(C_th * theta)] = C_th * sech^2(C_th * theta)
    dz_dtheta = k * tanh_C_om * (C_theta * sech2_C_th)
    dr_dtheta = float(-A * sech2_z * dz_dtheta)

    # 2. dr/domega
    # Inner derivative: d/domega [tanh(C_om * omega)] = C_om * sech^2(C_om * omega)
    dz_domega = k * tanh_C_th * (C_omega * sech2_C_om)
    dr_domega = float(-A * sech2_z * dz_domega)

    return r, dr_dtheta, dr_domega