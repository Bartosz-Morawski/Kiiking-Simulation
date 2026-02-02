"""
TODO simple pendulum model object using the function with no d/r portions
TODO expand to include the dr functions
TODO add 'pumping' object to simplify code
    - 1. instantaneous change
    - 2. change certain
NB - Gravity acts in the positive x axis to match the assignement doc

"""
import numpy as np
from scipy.integrate import odeint, solve_ivp


class SimplePendulum:
    """
    Simple pendulum model based on the equation:
    d²θ/dt² = -(g/r) sin(θ)
    simplified from the polar NII where r is kept the same

    N.B
        - This is only the ODE acting in the e_theta direction
        - Mass doesn't contribute in this direction, so not currently a variable
    """

    def __init__(self,r = 1.0 , g=9.81, m=1.0,theta0=np.pi / 4, omega0=1.0):
        """
        Initialize pendulum parameters.
        Used to define the simple pendulum var with parameters

        Parameters:
        -----------
        r : float ( constant )
            Length of pendulum (m)
        g : float
            Gravitational acceleration (m/s^2)
        theta0 : float
            Initial angle (radians, measured from vertical)
            default = 0.25pi
        omega0 : float
            Initial angular velocity (rad/s)
            default = 1.0
        """

        self.r =r
        self.g = g
        self.m = m
        self.theta0 = theta0
        self.omega0 = omega0
        self.K = g /r  # Constant from the equation
        # TODO K will no longer be constant once we complicate the model, this needs to be adjusted

        # Creates empty var for solution, this is hte info we plot
        self.t = None       # time
        self.theta = None   # angular displacement
        self.omega = None   # angular velocity
        self.alpha = None   # angular acceleration


    def derivatives(self,t, state):
        """
        Calculate derivatives for the ODE system.
        Input:
            state = [θ, ω] where ω = dθ/dt
            t - not used as the derivative is a function of it's self ( autonomous)

        Returns: [dθ/dt, dω/dt]
        """
        theta, omega = state

        dtheta_dt = omega #See state vector on doc
        domega_dt = -self.K * np.sin(theta) #This is the equation of motion

        return [dtheta_dt, domega_dt]

    def solve(self, time_range= (0,10), dt=0.01):
        """
        Solve the pendulum equation as set of 2 first order ODEs

            theta'(t) = omega(t)
            omega'(t) = -(g/L)*(sin(theta(t)))

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
        self.alpha = -(self.g / self.r) * np.sin(self.theta)

        return self.t, self.theta, self.omega

    #TODO define a function to get the angular acceleration of the system as well to plot (alpha) i.e d2theta


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
        KE = 0.5*self.m * self.r ** 2 * self.omega ** 2

        # Potential energy: m g h = m g r (1 - cos(θ))
        PE = self.m * self.g * self.r * (1 - np.cos(self.theta))

        total = KE + PE

        return KE, PE, total

