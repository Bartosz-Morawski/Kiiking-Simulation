"""
TODO create a visualiser over time
TODO plots potential vs kinetic over time
    subplot pumping
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pendulum import SimplePendulum
def plot_motion_2d(pendulum):
    """
    #TODO write up
    :param pendulum:
    :return: fig : matplotlib plot
    """
    if pendulum.theta is None:
        raise   ValueError("no angular data pls check")

    KE, PE, total_E = pendulum.get_energy()
    x,y = pendulum.get_cartesian_position()
    #create fig
    fig, axes = plt.subplots(2, 2)


    # 1 - angle distance v time
    ax = axes[0, 0]
    ## Wrap to find angular distance rather than like 30pi
    theta_wrapped = (pendulum.theta + np.pi) % (2 * np.pi) - np.pi
    theta_deg = np.degrees(theta_wrapped)
    ax.plot(pendulum.t, theta_deg, 'b')
    #TODO this is showing as a line going up as it's not the angular position, its the angle used in the position
    ax.axhline(y=0, linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('deg')
    ax.set_title('Position vs Time')


    # 2 - ang v
    ax = axes[0, 1]
    ax.plot(pendulum.t, pendulum.omega, 'r-', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel(' ω (rad/s)', fontsize=11)
    ax.set_title('Velocity vs Time', )

    # 3 - energy v time
    ax = axes[1, 1]
    ax.plot(pendulum.t, KE, 'b-', label='kin')
    ax.plot(pendulum.t, PE, 'r-', label='pot')
    ax.scatter(pendulum.t, total_E,label='sum')

    b, a = np.polyfit(pendulum.t, total_E, deg=1) #slope and intercept of Total E
    ax.plot(pendulum.t, b*pendulum.t + a, label=f'fit of total E {b}')
    ax.set_xlabel('Time (s)')
    ax.set_xlabel('t')
    ax.set_ylabel('J')
    ax.set_title('Energy')
    ax.legend()

    # 4 - position v time
    ax = axes[1, 0]
    ax.plot(y,-x, linewidth=1.5, color='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Position vs Time')
    ax.set_aspect('equal')
    ax = axes[1, 1]

    return fig


# TODO PHASE Plane Plot
# Phase Plane Plot
def phase_plane_plot(pendulum):
    """
    Should be the plot of theta vs omega
    :param pendulum:
    :return:
    """
    _,theta, omega = pendulum.solve()

    fig = plt.figure(figsize = (5,3))
    plt.plot(theta, omega, 'b-')
    plt.xlabel('theta')
    plt.ylabel('omega (dtheta/dt)')
    return fig


# Sanity check animation
def animate_pendulum(pendulum, interval=10, shadow_frame=50):
    """
    Create animate output  of pendulum.
    blue line
    interval = 10 millisec - matches dt 0.1, so is in 'realtime'
    -> this means the shadow frame is last 0.5s
    TODO add timer in corner of animation pot to have time
    """
    if pendulum.theta is None:
        raise ValueError("Pendulum must be solved before animating!")

    # Get cartesian coordinates (in document system: x=vertical, y=horizontal)
    x, y = pendulum.get_cartesian_position()

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set axis limits with some padding
    r = pendulum.r
    ax.set_xlim(-1.3 * r, 1.3 * r)  # y-axis (horizontal)
    ax.set_ylim(-1.3 * r, 1.3 * r)  # x-axis (vertical, will be inverted)
    ax.invert_yaxis()  # x points the right way
    ax.set_aspect('equal')

    # Labels
    ax.set_xlabel('y  - HORIZONTAL')
    ax.set_ylabel('x  - VERTICAL (down is positive)')
    ax.set_title('hopefully this works')


    # mark (0.0) as pivot
    ax.plot(0, 0, 'ko', markersize=15, zorder=10)

    # Initialize plot elements
    bar, = ax.plot([], [], 'o-', lw=3, markersize=15,
                   color='darkblue', markerfacecolor='red',
                   markeredgecolor='darkred', markeredgewidth=2)
    trail, = ax.plot([], [], '-', lw=1.5, alpha=0.5, color='blue')

    # intialise
    def init():
        bar.set_data([], [])
        trail.set_data([], [])

    def update(frame):
        """frame"""
        # bar plotted y,x
        bar.set_data([0, y[frame]], [0, x[frame]])

        # Update previous
        start_idx = max(0, frame - shadow_frame)
        trail.set_data(y[start_idx:frame + 1], x[start_idx:frame + 1])


    # Create animation
    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(pendulum.t),
                         interval=interval,
                         repeat=True)
    plt.grid(True, alpha=0.3)

    return anim



