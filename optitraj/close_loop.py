"""
Want to implement a close loop control for the robot.
The close loop control will use the MPC to compute the optimal control input.
The MPC will be used to compute the optimal control input.
User can feed in their own dynamics system to put into a plant
"""
import numpy as np
import time
from typing import Tuple, Dict, Any, List, Optional, Callable
from optitraj.utils.data_container import MPCParams
from optitraj.models.casadi_model import CasadiModel
from optitraj.mpc.optimization import OptimalControlProblem
from optitraj.utils.report import Report
from abc import ABC, abstractmethod
from optitraj.dynamics_adapter import DynamicsAdapter


class CloseLoopSim():
    """
    Closed-loop control simulation for a robotic system using Model Predictive Control (MPC).

    This class simulates the closed-loop control of a robot using an MPC controller
    to compute the optimal control inputs. The user can feed their own dynamics system
    into the plant through the dynamics adapter.

    Attributes:
        optimizer (OptimalControlProblem): The MPC optimizer used to compute control inputs.
        x_init (np.ndarray): Initial state of the system.
        x_final (np.ndarray): Desired final state of the system.
        u0 (np.ndarray): Initial control inputs.
        dynamics_adapter (Optional[DynamicsAdapter]): Adapter for the dynamics of the system.
        N (int): Number of control steps in the simulation. Defaults to 100.
        log_data (bool): Whether to log the data during simulation. Defaults to True.
        stop_criteria (Optional[Callable[[np.ndarray], bool]]): Optional stopping criteria based on the current state.
        file_name (str): File name for saving reports. Defaults to an empty string.
        print_every (int): Frequency of printing the current state. Defaults to 10.
        state_limits (Dict): Limits on the system's states.
        ctrl_limits (Dict): Limits on the control inputs.
        control_names (List[str]): Names of the control variables.
        current_step (int): Current simulation step.
        time (float): Current time in the simulation.
        mpc_time (float): Time step for the MPC controller.
        next_time (float): Time for the next control update.
        update_controller (bool): Flag to determine whether to update the controller.
        done (bool): Flag to determine if the simulation is complete.

    Methods:
        get_control_names() -> List[str]:
            Retrieves the control variable names from the optimizer.

        update_x_final(x_final: np.ndarray) -> None:
            Updates the final state of the system.

        update_x_init(x_init: np.ndarray) -> None:
            Updates the initial state of the system.

        update_time() -> None:
            Updates the current simulation time and checks if the MPC control needs updating.

        save_data(sol: Dict) -> None:
            Saves the current simulation data, including state trajectory and control inputs.

        shift_next(sol: Dict) -> None:
            Shifts the initial state and control inputs to the next time step in the trajectory.

        run_single_step(xF: np.ndarray = None) -> Dict:
            Executes a single step of the closed-loop simulation. Optionally updates the final state.

        run() -> None:
            Runs the full closed-loop simulation for N steps using the MPC controller.

        reset(x0: np.ndarray, xF: np.ndarray, u0: np.ndarray, file_name: str) -> None:
            Resets the simulation with new initial conditions and control inputs.
    """

    def __init__(self,
                 optimizer: OptimalControlProblem,
                 x_init: np.ndarray,
                 x_final: np.ndarray,
                 u0: np.ndarray,
                 dynamics_adapter: Optional["DynamicsAdapter"] = None,
                 N: int = 100,
                 log_data: bool = True,
                 stop_criteria: Optional[Callable[[np.ndarray], bool]] = None,
                 file_name: str = '',
                 print_every: int = 10) -> None:

        self.optimizer: OptimalControlProblem = optimizer
        self.state_limits: Dict = self.optimizer.state_limits
        self.ctrl_limits: Dict = self.optimizer.ctrl_limits
        self.optimizer.init_optimization()
        self.x_init: np.ndarray = x_init
        self.x_final: np.ndarray = x_final
        self.u0: np.ndarray = u0
        self.dynamics_adapter: DynamicsAdapter = dynamics_adapter
        self.N: int = N
        self.log_data: bool = log_data
        self.control_names = self.get_control_names()
        self.stop_criteria: Optional[Callable[[
            np.ndarray], bool]] = stop_criteria
        self.ctrl_idx: int = 1  # next control index
        self.print_every = print_every
        # this is used to update the controller, used for the JSBSimAdapter
        self.update_controller: bool = True
        if self.log_data:
            self.report: Report = Report(self.state_limits,
                                         self.ctrl_limits, self.control_names,
                                         file_name=file_name)
        self.current_step: int = 0
        self.time: float = 0.0  # current time in the simulation
        self.mpc_time = self.optimizer.mpc_params.dt  # ctrl frequency
        self.next_time = self.time + self.mpc_time  # next time to update ctrl
        self.done: bool = False

    def get_control_names(self) -> List[str]:
        """
        Retrieve the names of the control variables from the optimizer.

        Returns:
            List[str]: The names of the control variables.
        """
        controls = self.optimizer.casadi_model.controls
        n_controls = self.optimizer.casadi_model.n_controls
        control_names = [str(controls[i].name()) for i in range(n_controls)]

        return control_names

    def update_x_final(self, x_final: np.ndarray) -> None:
        """
        Update the final state of the system.

        Args:
            x_final (np.ndarray): The new final state.

        Raises:
            ValueError: If the shape of x_final is incorrect.
        """
        if x_final.shape[0] != self.x_final.shape[0]:
            raise ValueError("x_final is not the correct shape \
                shape input must be: ", self.x_final.shape[0], "but got: ",
                             x_final.shape[0])
        self.x_final = x_final

    def update_u0(self, u0: np.ndarray) -> None:
        """
        Update the initial control inputs.
        Args:
            u0 (np.ndarray): The new initial control inputs.

        Raises:
            ValueError: If the shape of u0 is incorrect.
        """
        if u0.shape[0] != self.u0.shape[0]:
            raise ValueError("u0 is not the correct shape \
                shape input must be: ", self.u0.shape[0], "but got: ",
                             u0.shape[0])
        self.u0 = u0

    def update_x_init(self, x_init: np.ndarray) -> None:
        """
        Update the initial state of the system.

        Args:
            x_init (np.ndarray): The new initial state.

        Raises:
            ValueError: If the shape of x_init is incorrect.
        """
        # make sure the x_init is the correct shape
        if x_init.shape[0] != self.x_init.shape[0]:
            raise ValueError("x_init is not the correct shape \
                shape input must be: ", self.x_init.shape[0], "but got: ",
                             x_init.shape[0])
        self.x_init = x_init

    def update_time(self) -> None:
        """
        Update the current time in the simulation and check if the controller
        needs to be updated.
        """
        if self.dynamics_adapter is not None:
            self.time += self.dynamics_adapter.simulator.dt
            # check if the time is greater than the mpc time
            if self.time >= self.next_time:
                self.next_time = self.time + self.mpc_time
                self.update_controller = True
        else:
            self.time += self.optimizer.mpc_params.dt

    def save_data(self, sol: Dict) -> None:
        """
        Save the current simulation data, including state trajectory, control inputs, and time.

        Args:
            sol (Dict): The solution dictionary containing state and control information.
        """
        self.report.log_state_traj(sol['states'])
        self.report.log_control_traj(sol['controls'])
        self.report.log_current_state(self.x_init)
        self.report.log_current_control(self.u0)
        self.report.log_time(self.time)

    def shift_next(self, sol: dict) -> None:
        """
        Shift over the next initial state
        """
        states = sol['states']
        controls = sol['controls']
        idx = 1
        x_init = []
        u0 = []
        for i, state_name in enumerate(self.state_limits.keys()):
            x_init.append(states[state_name][idx])

        for i, ctrl_name in enumerate(self.ctrl_limits.keys()):
            u0.append(controls[ctrl_name][idx])

        self.x_init = np.array(x_init)
        self.u0 = np.array(u0)

        return

    def run_single_step(self, xF: np.ndarray = None,
                        x0: np.ndarray = None,
                        u0: np.ndarray = None) -> Dict:
        """
        Execute a single step of the closed-loop simulation.

        This method performs one iteration of the simulation. It updates the state and control
        inputs using the optimizer and optionally allows for updating the final state.

        Args:
            xF (np.ndarray, optional): The updated final state. If None, the existing final state is used.

        Returns:
            Dict: The solution dictionary with state and control information.
        """
        self.current_step += 1

        if xF is not None:
            self.update_x_final(xF)

        if x0 is not None:
            self.update_x_init(x0)

        if u0 is not None:
            self.update_u0(u0)

        if self.dynamics_adapter is not None and self.update_controller:
            # print("Updating controller")
            sol = self.optimizer.solve_and_get_solution(
                self.x_init, self.x_final, self.u0)
            self.dynamics_adapter.set_controls(sol['states'],
                                               sol['controls'],
                                               self.ctrl_idx,
                                               xF=self.x_final)
            self.update_controller = False
        else:
            sol = self.optimizer.solve_and_get_solution(
                self.x_init, self.x_final, self.u0)

        if self.current_step % self.print_every == 0:
            print("step: ", self.x_init)

        self.update_time()

        # run the dynamics system if it is there
        if self.dynamics_adapter is not None:
            self.dynamics_adapter.run()
            self.x_init = self.dynamics_adapter.get_state_information()
            self.u0 = self.dynamics_adapter.get_control_information()
        else:
            self.shift_next(sol)

        if self.log_data:
            self.save_data(sol)

        # def check criteria
        if self.stop_criteria is not None and self.stop_criteria(
                self.x_init, self.x_final):
            print('Stopping criteria met')
            self.report.save_everything()
            self.done = True
            return sol

        return sol

    def run(self) -> None:
        """
        Run the full closed-loop simulation using the MPC controller.

        The simulation proceeds for N steps, updating the state and control inputs 
        at each iteration using the MPC controller.
        """

        for i in range(self.N):
            sol: Dict = self.run_single_step()
            if self.done:
                break

        return

    def reset(self, x0: np.ndarray,
              xF: np.ndarray, u0: np.ndarray,
              file_name: str) -> None:
        """
        Reset the simulation with new initial and final states and control inputs.

        Args:
            x0 (np.ndarray): The initial state of the system.
            xF (np.ndarray): The desired final state of the system.
            u0 (np.ndarray): The initial control inputs.
            file_name (str): The file name for saving the report.
        """
        self.x_init = x0
        self.x_final = xF
        self.u0 = u0
        self.optimizer.init_optimization()
        self.report = Report(self.state_limits,
                             self.ctrl_limits, self.control_names,
                             file_name)
        if self.dynamics_adapter is not None:
            self.dynamics_adapter.reset()
        self.time = 0.0
        self.current_step = 0
        self.next_time = self.time + self.mpc_time
        self.done = False