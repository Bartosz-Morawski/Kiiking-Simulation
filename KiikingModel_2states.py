"""
TODO Mod the simple Pendulum to be var r model
"""
import numpy as np
from scipy.integrate import solve_ivp

# ===============================================
# Model Parameters
# ===============================================
#TODO Placeholder parameters below, replace with parameters (or rewrite for clarity, currently cluttered as relationship based)

G = 9.81
L = 8
delta_R = 0.3 #Diffenrence in COM from standing/squatting (m) TODO pls vary this one and leave R_min,max alone

R_max = L     # When Squatting COM at max, as swing is light this seems fine
R_min = L-delta_R # When Standing R goes up 0.5m

M = 70        # Mass (kg) - TODO do we need to consider swing weight and skew higher? not that imp tbh, motion doesnt depend on it, only energy

# ===============================================
# Model Variables
# ===============================================
# aka defaults for IVP, but can de varied when calling class object

# Initial states for IVP solver
# TODO these deffo need to be checked I just picked random numbers
theta_init = 0.3 #Starting Angle (rad)
omega_init = np.pi/10 #Initial Starting Speed TODO most ppl don't stand onto the swing, they carry some movement in

#Period of time IVP runs over (s)
t_span = (0,600)

# ===============================================
# Control Functions
# ===============================================

def control_constant_length(theta, omega, r_min, r_max):
    """
    Should return simple pendulum (fixed r)
    """
    return r_max

def control_velocity_tanh(theta, omega, r_min, r_max, omega_ref=1):
    """
    Velocity-based pumping:
    - High |omega| -> stand (r_min)
    - Low |omega|  -> squat (r_max)
    """
    s = np.tanh(abs(omega) / omega_ref)  # in [0, 1)
    return r_max - (r_max - r_min) * s

def control_stand_bottom_squat_top(theta, omega, r_min, r_max):
    """
    Instantaneous pumping control.

    Physics:
    - Bottom of swing (theta ≈ 0): stand up → shorten pendulum
    - Top of swing (|theta| large): squat → lengthen pendulum

    This breaks time-reversal symmetry and injects energy.
    """
    if np.cos(theta) > 0:
        # Bottom half of swing
        return r_min
    else:
        # Top half of swing
        return r_max


# def control_stand_bottom_squat_top(theta, omega, r_min, r_max):
#     """
#     Optimized pumping control:
#     - Stand (r_min) near the bottom to perform work against centrifugal force.
#     - Squat (r_max) ONLY at the peaks to reset with minimal energy loss.
#     """
#     # Detect the peak: velocity is nearly zero
#     is_at_peak = abs(omega) < 0.1
#
#     # Detect the power stroke: near the bottom and moving fast
#     # Increasing the cosine threshold (0.85 ~ 30 degrees) focuses the work
#     # where centrifugal force is highest.
#     is_near_bottom = np.cos(theta) > 0.85
#
#     if is_near_bottom:
#         return r_min
#     elif is_at_peak:
#         return r_max
#
#     # Hold standing position while climbing to maintain the energy gain
#     # until the peak is reached for a reset.
#     return r_min

# ===============================================
# KiiKing Model
# ===============================================

class KiikingModel:
    """
    Object based pendulum model
    """

    def __init__(self,control,r_min=R_min,r_max=R_max, g=G, m=M,theta0=theta_init, omega0=omega_init):
        """
        Initialize pendulum parameters.
        Used to define the simple pendulum var with parameters

        Parameters:
        -----------
        r : float ( constant )
            Length of pendulum (m)
        G : float
            Gravitational acceleration (m/s^2)
        theta0 : float
            Initial angle (radians, measured from vertical)
            default = 0.25pi
        omega0 : float
            Initial angular velocity (rad/s)
            default = 1.0
        control : function
            function defines how r varies
        """
        #Physical Parameters
        self.r_min = r_min
        self.r_max = r_max
        self.g = g
        self.m = m
        #Model Parameters
        self.theta0 = theta0
        self.omega0 = omega0
        self.control = control


        # Creates empty var for solution, this is hte info we plot
        self.t = None       # time
        self.theta = None   # angular displacement
        self.omega = None   # angular velocity
        self.alpha = None   # angular acceleration
        self.r = None # r from 0,0
        self.r_dot = None # dr/dt

        # Create Empty values for most recent r / r dot
        self._last_r = None
        self._last_r_dot = None

    def constraint_partials(self, theta, omega):
        """
        Returns
        :param theta:
        :param omega:
        :return:
        """
        eps = 1e-5

        # 1. Calculate the r values according to the given constraint
        r = self.control(theta, omega, self.r_min, self.r_max)

        # 2. Calculate the partial differentials using central differencing ( to reduce error)]
        r_th_p = self.control(theta + eps, omega, self.r_min, self.r_max)
        r_th_m = self.control(theta - eps, omega, self.r_min, self.r_max)
        dr_dtheta = (r_th_p - r_th_m) / (2 * eps)

        r_om_p = self.control(theta, omega + eps, self.r_min, self.r_max)
        r_om_m = self.control(theta, omega - eps, self.r_min, self.r_max)
        dr_domega = (r_om_p - r_om_m) / (2 * eps)

        return r, dr_dtheta, dr_domega

    def angular_acceleration(self, theta, omega):
        """
        Computes:
          - r(theta, omega)
          - dr/dtheta
          - dr/domega
          - angular acceleration alpha = dω/dt
        """


        # Constraint
        r, dr_dtheta, dr_domega = self.constraint_partials(theta, omega)

        # Equation of motion (rearranged)
        numerator = (
                -(self.g / r) * np.sin(theta)
                - (2 / r) * omega ** 2 * dr_dtheta
        )

        denominator = 1 + (2 / r) * omega * dr_domega

        # Clip Values so that we don'd divide by something too small
        # TODO Check if these clipped values make sense
        denominator = np.clip(denominator, 0.2, 5.0)

        alpha = numerator / denominator

        return alpha, r, dr_dtheta, dr_domega

    def derivatives(self,t, state):
        """
        Calculate derivatives for the ODE system.
        - Calculates partial derivatives using central differencing
            - Uses numerical methods so we don't need to calculate derivatives


        Input:
            state = [θ, ω] where ω = dθ/dt
        Returns: [dθ/dt, dω/dt] (i.e sencond state vector)
        """
        theta, omega = state

        omega_dot, _, _, _ = self.angular_acceleration(theta, omega)

        return [omega, omega_dot]

    def solve(self, time_range= t_span, dt=0.01):
        """
        Solve the pendulum equation as set of 2 first order ODEs

            theta'(t) = omega(t)
            omega'(t) = -(G/L)*(sin(theta(t)))

        Parameters:
        -----------
        time_range : tuple
            (t_start, t_end) time span for output data
        dt : float
            Time step for output

        Output:
            t - time array
            theta - angle at
        """

        t_eval_points = np.arange(time_range[0], time_range[1], dt)
        initial_state = [self.theta0, self.omega0]

        # Solve ODE system
        solution = solve_ivp(
            self.derivatives,
            t_span=time_range,
            y0= initial_state,
            t_eval=t_eval_points,
            rtol=1e-5, #added as num error was causing total energy to change in a closed system
            atol=1e-5
        )
        if not solution.success:
            raise RuntimeError(f"Integration failed")

        self.t = solution.t
        self.theta = solution.y[0, : ]
        self.omega = solution.y[1, :]

        n = len(self.t)
        self.r =  np.zeros(n)
        self.r_dot =  np.zeros(n)
        self.alpha = np.zeros(n)

        for i, (th, om) in enumerate(zip(self.theta, self.omega)):
            alpha, r, dr_dtheta, dr_domega = self.angular_acceleration(th, om)

            self.alpha[i] = alpha
            self.r[i] = r
            self.r_dot[i] = dr_dtheta * om + dr_domega * alpha

        return self.t, self.theta, self.omega


    def get_cartesian_position(self):
        """
        Convert polar coordinates to Cartesian for visualization.
        From the
            \vec r(t) = r(t)e_r
        TODO: this currently uses a set r, need to make actual conversion later

        Returns: x, y coordinates of the mass
        """
        if self.theta is None:
            raise ValueError("No angles found")


        x = self.r * np.cos(self.theta) # remember for plotting that x points down
        y = self.r * np.sin(self.theta)

        #TODO set x to minus here or later?

        return x, y

    def get_energy(self):
        """
        Calculate kinetic, potential, and total energy.

        Returns: KE, PE, Total Energy (assuming mass = 1)
        assuming point mass
        """
        if self.theta is None:
            raise ValueError("Must call solve() first")

        # Kinetic energy: using standard angular motion kinetic
        KE = 0.5*self.m * (( self.r * self.omega ) ** 2 + self.r_dot **2)


        # PE = self.m * self.g * self.r * (1 - np.cos(self.theta))
        h = -self.r * np.cos(self.theta)
        PE = self.m * self.g * h


        return KE, PE, KE + PE

    def check_energy_gain(self):
        """Check if pumping is increasing energy."""
        if self.t is None:
            raise ValueError("Solve first")

        _, _, E_total = self.get_energy()
        E_initial = E_total[0]
        E_final = E_total[-1]
        E_gain_percent = (E_final - E_initial) / E_initial * 100

        print(f"Energy gain: {E_gain_percent:+.2f}%")
        if E_gain_percent > 0:
            print("✓ Pumping is working!")
        else:
            print("✗ Energy decreasing - check control strategy")

        return E_gain_percent

    def get_max_amplitude(self):
        """Get maximum angle reached."""
        if self.theta is None:
            raise ValueError("Must solve first")

        return np.max(np.abs(self.theta))

    def get_bottom_crossing_velocities(self):
        """
        Detects when the pendulum passes theta = 0 and returns the time and velocity.
        """
        if self.theta is None: raise ValueError("Solve first")

        times = []
        velocities = []

        for i in range(len(self.theta) - 1):
            # Check for zero crossing of theta
            if self.theta[i] * self.theta[i + 1] <= 0:
                # Linear interpolation for better time accuracy
                weight = abs(self.theta[i]) / (abs(self.theta[i]) + abs(self.theta[i + 1]))
                t_cross = self.t[i] + weight * (self.t[i + 1] - self.t[i])
                # Magnitude of tangential velocity at the bottom: v = r * omega
                v_bottom = abs(self.r[i] * self.omega[i])

                times.append(t_cross)
                velocities.append(v_bottom)

        return np.array(times), np.array(velocities)