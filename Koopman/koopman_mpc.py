import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import json

from optitraj.utils.report import Report
from optitraj.utils.data_container import MPCParams
from optitraj.models.casadi_model import CasadiModel
from optitraj.mpc.optimization import OptimalControlProblem
from optitraj.close_loop import CloseLoopSim
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


"""
JSon

"""


@dataclass
class KoopmanStateSpace:
    """
    Slot your dataclass for the koopman state space here,
    this is just a simple wrapper to hold the A and B matrices
    """
    A: np.ndarray
    B: np.ndarray


def load_koopman_json(path: str) -> tuple[KoopmanStateSpace, float, str]:
    """
    TODO: Xander use this function to load that json file you created 
    with the A and B matrices, and then use that to create the KoopmanStateSpace object 
    that is passed into the KoopManModel constructor in the example below.

    - You might need to make a function to upload_koopman values to json as well
    """

    with open(path, "r") as f:
        data = json.load(f)

    model_name = data["model_name"]
    dt = float(data["dt"])
    A = np.array(data["A"], dtype=float)
    B = np.array(data["B"], dtype=float)
    return KoopmanStateSpace(A=A, B=B), dt, model_name


class KoopManModel(CasadiModel):
    def __init__(self,
                 koopman_state_space: KoopmanStateSpace,
                 dt_val: float = 0.1) -> None:
        super().__init__()
        self.koopman_state_space = koopman_state_space
        self.A = koopman_state_space.A
        self.B = koopman_state_space.B
        self.dt_val = dt_val
        self._validate_shapes()

        self.define_states()
        self.define_controls()
        self.define_state_space()

    def _validate_shapes(self) -> None:
        if self.A.ndim != 2 or self.B.ndim != 2:
            raise ValueError("A and B must both be 2D matrices.")

        n_rows_A, n_cols_A = self.A.shape
        n_rows_B, n_cols_B = self.B.shape

        if n_rows_A != n_cols_A:
            raise ValueError(f"A must be square. Got shape {self.A.shape}")

        if n_rows_B != n_rows_A:
            raise ValueError(
                f"B must have same number of rows as A. "
                f"Got A: {self.A.shape}, B: {self.B.shape}"
            )

        self.n_states = n_rows_A
        self.n_controls = n_cols_B

    def define_states(self) -> None:
        """define the states of your system"""
        # positions ofrom world
        self.states = ca.MX.sym('x', self.n_states, 1)

    def define_controls(self) -> None:
        self.controls = ca.MX.sym('u', self.n_controls, 1)

    def define_state_space(self) -> None:
        """
        Mathematical representation of:
        x_{k+1} = A x_k + B u_k 
        The backend already does the RK4 integration, 
        so we can just use the linear dynamics here and it will be integrated for us.
        """
        A_ca = ca.DM(self.A)
        B_ca = ca.DM(self.B)
        self.x_dot = A_ca @ self.states + B_ca @ self.controls

        # the casadimodel base already has a built in rk45 integrator,
        # so we can just use that to get the next state
        self.function = ca.Function('dynamics',
                                    [self.states, self.controls],
                                    [self.x_dot])


class KoopmanOptControl(OptimalControlProblem):
    def __init__(self, mpc_params: MPCParams,
                 casadi_model: CasadiModel) -> None:
        super().__init__(mpc_params=mpc_params, casadi_model=casadi_model)

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

    def _parameter_length(self) -> int:
        return super()._parameter_length()


if __name__ == "__main__":
    koopman_state_space, dt, model_name = load_koopman_json(
        "Koopman/model_results.json")
    model = KoopManModel(koopman_state_space=koopman_state_space)
    print("A matrix:\n", model.A)
    print("B matrix:\n", model.B)
    control_limits = {'u': {'min': -1.0, 'max': 1.0}}
    state_limits = {'x1': {'min': -10.0, 'max':
                           10.0}, 'x2': {'min': -10.0, 'max': 10.0}}

    model.set_control_limits(control_limits)
    model.set_state_limits(state_limits)

    Q: np.ndarray = np.eye(model.n_states)
    R: np.ndarray = np.eye(model.n_controls)
    N = 5  # horizon length
    mpc_params = MPCParams(Q=Q, R=R, N=N, dt=model.dt_val)
    opt_control = KoopmanOptControl(mpc_params=mpc_params,
                                    casadi_model=model)

    # initial conditions
    x0: np.ndarray = np.array([[5.0], [0.0]])
    xF: np.ndarray = np.array([[0.0], [0.0]])
    u0: np.ndarray = np.array([[0.0]])

    def custom_stop_criteria(current_state: np.ndarray, target_state: np.ndarray) -> bool:
        """
        Custom stop criteria for the closed loop simulation.
        In this example, we stop if the state is within a certain distance of the target state.
        """
        distance = np.linalg.norm(current_state - target_state)
        return distance < 0.1  # stop if within 0.1 units of the target

    closed_loop_sim: CloseLoopSim = CloseLoopSim(
        optimizer=opt_control, x_init=x0, x_final=xF, u0=u0,
        N=100, log_data=True, stop_criteria=custom_stop_criteria)

    # use closed_sim.run_single_step() to just run one step of the sim and see the results
    solution: Dict[str, Any] = closed_loop_sim.run_single_step(
        xF=xF,
        x0=x0, u0=u0
    )
    print("Results of single step solution:", solution)
    x1 = solution["states"]["x1"]
    x2 = solution["states"]["x2"]
    u = solution["controls"]["u"]
    print(f"x1: {x1}")
    print(f"x2: {x2}")
    print(f"u: {u}")
    report: Report = closed_loop_sim.report
