import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from Scripts.controls import partials_tanh_theta_omega
from Scripts.model import solve_kiiking_2state
from Scripts.plotting import plot_summary, animate_solution



def main():
    # --- Physical parameters ---
    g = 9.81
    m = 70.0

    # --- Control parameters ---
    # r = (r_arm - 0.895) - 0.145 * tanh(k * sin(theta) * omega)
    r_arm = 7.0 # took 1 minute 40 seconds in video
    r0 = r_arm - 0.895 # Number determined by Noah
    A = 0.145 # Constant determining change in centre of mass
    k = 1.0

    # Physical posture limits
    r_min = r_arm - 1.04   # standing = shorter
    r_max = r_arm - 0.75   # squatting = longer

    partials_args = dict(r0=r0, A=A, k=k, r_min=r_min, r_max=r_max)

    # --- Initial conditions ---
    theta0 = 0.78 # approx pi/4 big push
    omega0 = 0.0 # starts from rest

    sol = solve_kiiking_2state(
        theta0=theta0,
        omega0=omega0,
        t_span=(0.0, 200.0),
        dt=0.01,
        g=g,
        m=m,
        partials_fn=partials_tanh_theta_omega,
        partials_args=partials_args,
    )

    # --- What to check in terminal ---
    r = sol["r"]
    E = sol["E"]
    denom = sol["denom"]
    print(f"r range: {r.min():.3f} .. {r.max():.3f} (Δ={r.max()-r.min():.3f})")
    print(f"E gain: {(E[-1]-E[0]) / abs(E[0]) * 100:+.2f}%")
    print(f"min |denom|: {abs(denom).min():.3e}  (should stay comfortably away from 0)")

    fig = plot_summary(sol)
    anim = animate_solution(sol, interval_ms=10, trail_frames=50)

    plt.show()


if __name__ == "__main__":
    main()
