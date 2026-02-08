"""
Two-state constrained Kiiking model (functional version).

We integrate only (theta, omega), and define r = f(theta, omega).

This uses the e_theta equation from the assignment notes (Eq. 10, second line):
    2 r_dot theta_dot + r theta_ddot = -g sin(theta)

With r = f(theta, omega), omega = theta_dot, and r_dot = f_theta * omega + f_omega * alpha,
we obtain:
    alpha = [-(g/r) sin(theta) - (2/r) omega^2 f_theta] / [1 + (2/r) omega f_omega]
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp


def angular_acceleration(
    theta: float,
    omega: float,
    g: float,
    partials_fn,
    partials_args: dict,
    denom_tol: float = 1e-3,
) -> tuple[float, float, float, float, float]:
    """
    Compute angular acceleration alpha = dω/dt for the two-state constrained model.

    Parameters
    ----------
    theta, omega
        Current state (θ, ω).
    g
        Gravitational acceleration.
    partials_fn
        Function returning (r, dr_dtheta, dr_domega) for the chosen control law.
        Example: controls.partials_tanh_theta_omega.
    partials_args
        Dict of keyword arguments passed into partials_fn (e.g. r0, A, k, r_min, r_max).
    denom_tol
        Threshold for detecting near-singularity of the implicit rearrangement.

    Returns
    -------
    alpha
        Angular acceleration dω/dt.
    r
        Effective length.
    dr_dtheta
        ∂r/∂θ.
    dr_domega
        ∂r/∂ω.
    denom
        Denominator value D = 1 + (2/r) ω (∂r/∂ω).

    Raises
    ------
    RuntimeError
        If the denominator becomes too small (model becomes ill-conditioned).
    """
    r, dr_dtheta, dr_domega = partials_fn(theta, omega, **partials_args)

    if r <= 1e-8:
        raise RuntimeError("r became too small; check r_min / control parameters.")

    numerator = (-(g / r) * np.sin(theta)) - ((2.0 / r) * (omega**2) * dr_dtheta)
    denom = 1.0 + ((2.0 / r) * omega * dr_domega)

    if abs(denom) < denom_tol:
        raise RuntimeError(
            f"Singularity/ill-conditioning: denom={denom:.3e}. "
            "Control too aggressive (k/A too large) or r too small, or ω large."
        )

    alpha = numerator / denom
    return float(alpha), float(r), float(dr_dtheta), float(dr_domega), float(denom)


def rhs(
    t: float,
    y: np.ndarray,
    g: float,
    partials_fn,
    partials_args: dict,
) -> np.ndarray:
    """
    Right-hand-side for solve_ivp.

    Parameters
    ----------
    t
        Time (unused; system is autonomous unless partials depend on time via args).
    y
        State vector [theta, omega].
    g, partials_fn, partials_args
        Passed to :func:`angular_acceleration`.

    Returns
    -------
    dydt
        Array [dtheta/dt, domega/dt].
    """
    theta = float(y[0])
    omega = float(y[1])
    alpha, *_ = angular_acceleration(theta, omega, g=g, partials_fn=partials_fn, partials_args=partials_args)
    return np.array([omega, alpha], dtype=float)


def solve_kiiking_2state(
    theta0: float,
    omega0: float,
    t_span: tuple[float, float],
    dt: float,
    g: float,
    m: float,
    partials_fn,
    partials_args: dict,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> dict:
    """
    Solve the two-state constrained Kiiking model and post-compute diagnostics.

    Parameters
    ----------
    theta0, omega0
        Initial conditions.
    t_span
        (t0, tf) integration interval.
    dt
        Output sampling step (seconds).
    g
        Gravitational acceleration.
    m
        Mass (used for energy).
    partials_fn
        Function returning (r, dr_dtheta, dr_domega).
    partials_args
        Arguments for the control law.
    rtol, atol
        solve_ivp tolerances.

    Returns
    -------
    out
        Dictionary with keys:
        - t, theta, omega
        - alpha
        - r, r_dot
        - denom (denominator diagnostic)
        - KE, PE, E (energies)
    """
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    y0 = np.array([theta0, omega0], dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, g=g, partials_fn=partials_fn, partials_args=partials_args),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    t = sol.t
    theta = sol.y[0, :]
    omega = sol.y[1, :]

    n = len(t)
    alpha = np.zeros(n)
    r = np.zeros(n)
    r_dot = np.zeros(n)
    denom = np.zeros(n)

    for i in range(n):
        a, ri, dr_th, dr_om, di = angular_acceleration(
            float(theta[i]),
            float(omega[i]),
            g=g,
            partials_fn=partials_fn,
            partials_args=partials_args,
        )
        alpha[i] = a
        r[i] = ri
        denom[i] = di
        r_dot[i] = dr_th * omega[i] + dr_om * alpha[i]

    # Energies (assignment convention: +x is down)
    v2 = (r * omega) ** 2 + (r_dot ** 2)
    KE = 0.5 * m * v2
    h = -r * np.cos(theta)
    PE = m * g * h
    E = KE + PE

    return {
        "t": t,
        "theta": theta,
        "omega": omega,
        "alpha": alpha,
        "r": r,
        "r_dot": r_dot,
        "denom": denom,
        "KE": KE,
        "PE": PE,
        "E": E,
        "g": g,
        "m": m,
        "partials_args": dict(partials_args),
    }
