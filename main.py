"""
FIXME tidy up the intro comments on this and other pages

This is the main file which includes:
- imports defined objects and functions the other files.
- changeable parameters of the model
    TODO consider moving parameters to pendulum.py incase we want to loop over a set,
        but keep parameters that we are looping over

When it outputs in the IDE (not saved, just shown) :
- animation of the pendulum position over time
- analysis data / plots:
    - GPE vs KE over time
    - speed of pump vs time to reach top
    -

    TODO figure out what the analysis for this needs to be

TODO ask lecturer if we are marked on any code things ( i.e  legibility / testing etc)
TODO ask about use of AI for refactoring etc, or is this an absolute ban
"""

import numpy as np
import matplotlib.pyplot as plt
from plotter import *
t_span = (0,10)
# =============
#Test section 1: just checking it works
# =============
#TODO turn this into a function called test_1, and call it later if needed

def simple(theta_init=np.pi/4, omega_init=0):
    """
    uses the simple pendulum model to output general graph + the animated one
    #TODO I've set the default twice, should change to if not empty then this,
        or should i removed defaul in the class object?
    """

    pendulum = SimplePendulum(theta0=theta_init, omega0=omega_init)

    # Solve the equations
    # I've assigned the solution to var here, but it's also now aaccessible via pendulum.t etc
    # TODO can I make it so that I don't
    t, theta, omega = pendulum.solve()

    # Time series plots
    fig1 = plot_motion_2d(pendulum)
    fig2 = phase_plane_plot(pendulum)


    # Create animation

    anim = animate_pendulum(pendulum)

    plt.show()

# ==========================================
# Tests for simple pendulum in basic sits
# ==========================================
# 1. release from rest
simple(theta_init=np.pi/4, omega_init=0)
# 2. at top of pendulum, no initial velocity
simple(theta_init=np.pi, omega_init=3)
# TODO expect this to be stationary, position seems to, but forces are acting up (small errors 1e-9)
# 3. released at bottom, Stationary as expected,
#simple(theta_init=0, omega_init=0)

