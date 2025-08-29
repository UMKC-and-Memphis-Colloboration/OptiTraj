
# ## 1. Imports
# Import the necessary libraries and modules
import numpy as np
import matplotlib.pyplot as plt


import casadi as ca
from optitraj.utils.report import Report
from optitraj.utils.data_container import MPCParams
from optitraj.models.casadi_model import CasadiModel
from optitraj.mpc.optimization import OptimalControlProblem
from optitraj.close_loop import CloseLoopSim
from typing import List, Tuple, Dict


class Plane(CasadiModel):
    """
    To begin you must first define the dynamics of your model to to that you must
    first inherit from the CasadiModel class and implement the abstract methods
    for this example we will be using the Plane Model
    """

    def __init__(self,
                 dt_val: float = 0.1,
                 airspeed_tau: float = 0.05,
                 pitch_tau: float = 0.02) -> None:
        super().__init__()
        self.dt_val = dt_val
        self.airspeed_tau = airspeed_tau
        self.pitch_tau = pitch_tau
        self.define_states()
        self.define_controls()
        self.define_state_space()

    def define_states(self) -> None:
        """define the states of your system"""
        # positions ofrom world
        self.x_f = ca.MX.sym('x_f')
        self.y_f = ca.MX.sym('y_f')
        self.z_f = ca.MX.sym('z_f')

        # attitude
        self.phi_f = ca.MX.sym('phi_f')
        self.theta_f = ca.MX.sym('theta_f')
        self.psi_f = ca.MX.sym('psi_f')
        self.v = ca.MX.sym('v')

        self.states = ca.vertcat(
            self.x_f,
            self.y_f,
            self.z_f,
            self.phi_f,
            self.theta_f,
            self.psi_f,
            self.v
        )

        self.n_states = self.states.size()[0]  # is a column vector

    def define_controls(self) -> None:
        """controls for your system"""
        self.u_phi = ca.MX.sym('u_phi')
        self.u_theta = ca.MX.sym('u_theta')
        self.u_psi = ca.MX.sym('u_psi')
        self.v_cmd = ca.MX.sym('v_cmd')

        self.controls = ca.vertcat(
            self.u_phi,
            self.u_theta,
            self.u_psi,
            self.v_cmd
        )
        self.n_controls = self.controls.size()[0]

    def define_state_space(self) -> None:
        """define the state space of your system"""
        self.g = 9.81  # m/s^2
        # #body to inertia frame
        self.x_fdot = self.v_cmd * ca.cos(self.theta_f) * ca.cos(self.psi_f)
        self.y_fdot = self.v_cmd * ca.cos(self.theta_f) * ca.sin(self.psi_f)
        self.z_fdot = -self.v_cmd * ca.sin(self.theta_f)

        self.phi_fdot = -self.u_phi * (1/self.pitch_tau) - self.phi_f
        self.theta_fdot = -self.u_theta * 1/0.5 - self.theta_f
        self.v_dot = ca.sqrt(self.x_fdot**2 + self.y_fdot **
                             2 + self.z_fdot**2)
        self.psi_fdot = self.u_psi

        self.z_dot = ca.vertcat(
            self.x_fdot,
            self.y_fdot,
            self.z_fdot,
            self.phi_fdot,
            self.theta_fdot,
            self.psi_fdot,
            self.v_dot
        )

        # ODE function
        name = 'dynamics'
        self.function = ca.Function(name,
                                    [self.states, self.controls],
                                    [self.z_dot])


class PlaneOptControl(OptimalControlProblem):
    """
    Example of a class that inherits from OptimalControlProblem
    for the Plane model using Casadi
    """

    def __init__(self,
                 mpc_params: MPCParams,
                 casadi_model: CasadiModel,
                 robot_radius: float = 3.0) -> None:
        super().__init__(mpc_params,
                         casadi_model)

        self.robot_radius: float = robot_radius

    def is_valid_obs_params(self) -> bool:
        """
        To use obstacle avoidance the parameters must be
        a list of Obstacle objects

        """
        if self.obs_params is None:
            raise ValueError("obs_params is None")

    def compute_dynamics_cost(self) -> ca.MX:
        """
        Compute the dynamics cost for the optimal control problem
        """
        # initialize the cost
        cost = 0.0
        Q = self.mpc_params.Q
        R = self.mpc_params.R

        x_final = self.P[self.casadi_model.n_states:]

        for k in range(self.N):
            states = self.X[:, k]
            controls = self.U[:, k]
            cost += cost \
                + (states - x_final).T @ Q @ (states - x_final) \
                + controls.T @ R @ controls

        return cost

    def compute_total_cost(self) -> ca.MX:
        cost = self.compute_dynamics_cost()
        return cost


def main() -> None:
    # First define the model
    plane: Plane = Plane()
    # now define the limits for the states and controls for your plane
    # the dictionary must be in the form of {'state_name': {'min': min_value, 'max': max_value}}
    control_limits_dict: dict = {
        'u_phi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
        'u_theta': {'min': -np.deg2rad(30), 'max': np.deg2rad(30)},
        'u_psi': {'min': -np.deg2rad(180), 'max': np.deg2rad(180)},
        'v_cmd': {'min': 15.0, 'max': 30.0}
    }
    state_limits_dict: dict = {
        'x': {'min': -np.inf, 'max': np.inf},
        'y': {'min': -np.inf, 'max': np.inf},
        'z': {'min': 0, 'max': 50},
        'phi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
        'theta': {'min': -np.deg2rad(15), 'max': np.deg2rad(15)},
        'psi': {'min': -np.pi, 'max': np.pi},
        'v': {'min': 15, 'max': 30.0}
    }
    # insert tthe limits into the model
    plane.set_control_limits(control_limits_dict)
    plane.set_state_limits(state_limits_dict)

    # now we will set the MPC weights for the plane
    # 0 means we don't care about the specific state variable 1 means we care about it
    Q: np.diag = np.diag([1, 1, 1, 0, 0, 0, 0])
    R: np.diag = np.diag([0, 0, 0, 1])

    # we will now slot the MPC weights into the MPCParams class
    mpc_params: MPCParams = MPCParams(Q=Q, R=R, N=15, dt=0.1)
    # formulate your optimal control problem
    plane_opt_control: PlaneOptControl = PlaneOptControl(
        mpc_params=mpc_params, casadi_model=plane)

    # now set your initial conditions for this case its the plane
    x0: np.array = np.array([5, 5, 10, 0, 0, 0, 15])
    xF: np.array = np.array([100, 100, 10, 0, 0, 0, 15])
    u_0: np.array = np.array([0, 0, 0, 15])

    def custom_stop_criteria(state: np.ndarray,
                             final_state: np.ndarray) -> bool:
        distance = np.linalg.norm(state[0:2] - final_state[0:2])
        if distance < 0.01:
            return True

    # we can now begin our simulation
    closed_loop_sim: CloseLoopSim = CloseLoopSim(
        optimizer=plane_opt_control, x_init=x0, x_final=xF, u0=u_0,
        N=300, log_data=True, stop_criteria=custom_stop_criteria)

    # we can now run the simulation
    closed_loop_sim.run()
    report: Report = closed_loop_sim.report

    states: Dict = report.current_state
    # we will now plot the trajectory
    # plot a 3D trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states['x'], states['y'], states['z'])
    ax.scatter(xF[0], xF[1], xF[2], c='r', label='Goal')
    plt.show()


if __name__ == '__main__':
    main()
