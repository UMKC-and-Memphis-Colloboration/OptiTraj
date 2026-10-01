import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import torch

from scipy.io import loadmat
from dataclasses import dataclass
from typing import Dict, Any

from optitraj.utils.data_container import MPCParams
from optitraj.models.casadi_model import CasadiModel
from optitraj.mpc.optimization import OptimalControlProblem


# ============================================================
# Koopman data container
# ============================================================

@dataclass
class KoopmanStateSpace:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray

    X_mean: np.ndarray
    X_std: np.ndarray

    U_mean: np.ndarray
    U_std: np.ndarray

    W1: np.ndarray
    W2: np.ndarray


# ============================================================
# Load Koopman model
# ============================================================

def load_koopman_mat(
    mat_path: str,
    checkpoint_path: str = None,
):
    """
    Load Koopman matrices and normalization values from MAT.

    If encoder weights are not present in the MAT file, load them
    from the matching PyTorch checkpoint.
    """

    data = loadmat(
        mat_path,
        squeeze_me=True,
    )

    required = [
        "A",
        "B",
        "C",
        "D",
        "X_mean",
        "X_std",
        "U_mean",
        "U_std",
        "sample_time",
    ]

    for key in required:
        if key not in data:
            raise KeyError(
                f"'{key}' is missing from {mat_path}"
            )

    # --------------------------------------------------------
    # Encoder weights
    # --------------------------------------------------------

    if (
        "encoder_W1" in data
        and "encoder_W2" in data
    ):
        print("Loading encoder weights from MAT file.")

        W1 = np.asarray(
            data["encoder_W1"],
            dtype=float,
        )

        W2 = np.asarray(
            data["encoder_W2"],
            dtype=float,
        )

    else:
        if checkpoint_path is None:
            raise KeyError(
                "encoder_W1 / encoder_W2 are missing from the MAT file "
                "and no checkpoint_path was provided."
            )

        print("Encoder weights not found in MAT.")
        print("Loading encoder weights from:", checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = (
            checkpoint["state_dict"]
            if "state_dict" in checkpoint
            else checkpoint
        )

        encoder_keys = sorted([
            key
            for key in state_dict.keys()
            if (
                key.startswith("lift.")
                and key.endswith(".weight")
            )
        ])

        print("Encoder keys found:", encoder_keys)

        if len(encoder_keys) != 2:
            raise RuntimeError(
                "Expected exactly 2 encoder weight matrices, "
                f"but found {len(encoder_keys)}: {encoder_keys}"
            )

        W1 = (
            state_dict[encoder_keys[0]]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )

        W2 = (
            state_dict[encoder_keys[1]]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )

    koopman = KoopmanStateSpace(
        A=np.asarray(data["A"], dtype=float),
        B=np.asarray(data["B"], dtype=float),
        C=np.asarray(data["C"], dtype=float),
        D=np.asarray(data["D"], dtype=float),

        X_mean=np.asarray(
            data["X_mean"],
            dtype=float,
        ).reshape(-1),

        X_std=np.asarray(
            data["X_std"],
            dtype=float,
        ).reshape(-1),

        U_mean=np.asarray(
            data["U_mean"],
            dtype=float,
        ).reshape(-1),

        U_std=np.asarray(
            data["U_std"],
            dtype=float,
        ).reshape(-1),

        W1=W1,
        W2=W2,
    )

    dt = float(
        np.asarray(
            data["sample_time"]
        ).squeeze()
    )

    n_z = koopman.A.shape[0]
    n_x = koopman.C.shape[0]
    n_features = n_z - n_x

    print()
    print("======================================")
    print("Loaded Koopman model")
    print("======================================")
    print("A:", koopman.A.shape)
    print("B:", koopman.B.shape)
    print("C:", koopman.C.shape)
    print("D:", koopman.D.shape)
    print("W1:", koopman.W1.shape)
    print("W2:", koopman.W2.shape)
    print("X_mean:", koopman.X_mean.shape)
    print("X_std:", koopman.X_std.shape)
    print("U_mean:", koopman.U_mean.shape)
    print("U_std:", koopman.U_std.shape)
    print("dt:", dt)
    print("Physical states:", n_x)
    print("Lifted features:", n_features)
    print("Koopman states:", n_z)
    print("======================================")
    print()

    if koopman.W1.shape[1] != n_x:
        raise ValueError(
            f"W1 expects {koopman.W1.shape[1]} inputs, "
            f"but C indicates {n_x} physical states."
        )

    if koopman.W2.shape[0] != n_features:
        raise ValueError(
            f"W2 produces {koopman.W2.shape[0]} features, "
            f"but Koopman model requires {n_features}."
        )

    return koopman, dt


# ============================================================
# Koopman CasADi model
# ============================================================

class KoopManModel(CasadiModel):

    SIN_IDX = 2
    COS_IDX = 3

    PHYSICAL_STATE_NAMES = [
        "phi",
        "theta",
        "sin_psi",
        "cos_psi",
        "p",
        "q",
        "r",
    ]

    CONTROL_NAMES = [
        "phiD",
        "thetaD",
        "sinpsiD",
        "cospsiD",
        "ThO",
        "C1",
        "C2",
        "C4",
    ]

    def __init__(
        self,
        koopman_state_space: KoopmanStateSpace,
        dt_val: float,
    ):
        super().__init__()

        self.ks = koopman_state_space

        self.A = self.ks.A
        self.B = self.ks.B
        self.C = self.ks.C
        self.D = self.ks.D

        self.dt_val = dt_val

        self._validate_shapes()

        # ====================================================
        # Input normalization absorbed into the dynamics
        #
        # $$
        # u_n = \frac{u-U_{\mathrm{mean}}}{U_{\mathrm{std}}}
        # $$
        #
        # $$
        # z_{k+1}=Az_k+Bu_n
        # $$
        #
        # becomes
        #
        # $$
        # z_{k+1}=Az_k+B_{\mathrm{real}}u_k+b_{\mathrm{real}}
        # $$
        # ====================================================

        inv_u_std = np.diag(
            1.0 / self.ks.U_std
        )

        self.B_real = (
            self.B @ inv_u_std
        )

        self.b_real = (
            -self.B_real @ self.ks.U_mean
        )

        # ====================================================
        # Physical-state reconstruction
        #
        # $$
        # x_n=Cz
        # $$
        #
        # $$
        # x=\mathrm{diag}(X_{\mathrm{std}})Cz+X_{\mathrm{mean}}
        # $$
        # ====================================================

        self.C_real = (
            np.diag(self.ks.X_std)
            @ self.C
        )

        self.x_offset = self.ks.X_mean

        self.define_states()
        self.define_controls()
        self.define_state_space()

    def _validate_shapes(self):

        if self.A.ndim != 2:
            raise ValueError("A must be 2D.")

        if self.B.ndim != 2:
            raise ValueError("B must be 2D.")

        nA1, nA2 = self.A.shape

        if nA1 != nA2:
            raise ValueError(
                f"A must be square. Got {self.A.shape}"
            )

        if self.B.shape[0] != nA1:
            raise ValueError(
                "B must have the same number of rows as A."
            )

        self.n_states = nA1
        self.n_controls = self.B.shape[1]
        self.n_physical_states = self.C.shape[0]

        if self.n_physical_states != 7:
            raise ValueError(
                "This implementation expects 7 physical states."
            )

    def define_states(self):

        self.states = ca.MX.sym(
            "z",
            self.n_states,
            1,
        )

    def define_controls(self):

        self.controls = ca.MX.sym(
            "u",
            self.n_controls,
            1,
        )

    # ========================================================
    # Physical state extraction
    # ========================================================

    def physical_state(self, z):

        C_real = ca.DM(
            self.C_real
        )

        offset = ca.DM(
            self.x_offset
        ).reshape((-1, 1))

        return (
            C_real @ z
            + offset
        )

    def physical_state_numpy(
        self,
        z: np.ndarray,
    ) -> np.ndarray:

        z = np.asarray(
            z,
            dtype=float,
        ).reshape(-1)

        return (
            self.C_real @ z
            + self.x_offset
        )

    # ========================================================
    # Koopman dynamics used INSIDE MPC
    #
    # Keep these dynamics linear/affine.
    #
    # $$
    # z_{k+1}=Az_k+B_{\mathrm{real}}u_k+b_{\mathrm{real}}
    # $$
    #
    # IMPORTANT:
    # No yaw projection is performed inside the optimizer.
    # That avoids square roots/divisions in every horizon step.
    # ========================================================

    def define_state_space(self):

        A = ca.DM(
            self.A
        )

        B_real = ca.DM(
            self.B_real
        )

        b_real = ca.DM(
            self.b_real
        ).reshape((-1, 1))

        z_next = (
            A @ self.states
            + B_real @ self.controls
            + b_real
        )

        self.function = ca.Function(
            "koopman_discrete_dynamics",
            [
                self.states,
                self.controls,
            ],
            [
                z_next,
            ],
        )

    # ========================================================
    # Physical yaw projection OUTSIDE the optimizer
    #
    # The physical yaw representation should satisfy
    #
    # $$
    # \sin^2(\psi)+\cos^2(\psi)=1
    # $$
    #
    # This helper projects the physical yaw pair and then
    # overwrites z[2], z[3] with the corresponding normalized
    # values. The learned latent features are preserved.
    # ========================================================

    def project_yaw_numpy(
        self,
        z: np.ndarray,
    ) -> np.ndarray:

        z = np.asarray(
            z,
            dtype=float,
        ).reshape(-1).copy()

        x = self.physical_state_numpy(
            z
        )

        s = x[self.SIN_IDX]
        c = x[self.COS_IDX]

        norm = np.sqrt(
            s * s
            + c * c
        )

        if norm > 1e-12:

            s /= norm
            c /= norm

            x[self.SIN_IDX] = s
            x[self.COS_IDX] = c

            # Physical -> normalized coordinates used by z[2:4]
            z[self.SIN_IDX] = (
                s
                - self.ks.X_mean[self.SIN_IDX]
            ) / self.ks.X_std[self.SIN_IDX]

            z[self.COS_IDX] = (
                c
                - self.ks.X_mean[self.COS_IDX]
            ) / self.ks.X_std[self.COS_IDX]

        return z

    # ========================================================
    # Physical aircraft state -> Koopman state
    #
    # $$
    # x_n=\frac{x-X_{\mathrm{mean}}}{X_{\mathrm{std}}}
    # $$
    #
    # $$
    # h_1=\tanh(W_1x_n)
    # $$
    #
    # $$
    # h_2=\tanh(W_2h_1)
    # $$
    #
    # $$
    # z=[x_n;h_2]
    # $$
    # ========================================================

    def lift_state(
        self,
        x_real: np.ndarray,
    ) -> np.ndarray:

        x_real = np.asarray(
            x_real,
            dtype=float,
        ).reshape(-1)

        if x_real.size != self.n_physical_states:
            raise ValueError(
                f"x_real must have "
                f"{self.n_physical_states} entries."
            )

        x_norm = (
            x_real
            - self.ks.X_mean
        ) / self.ks.X_std

        h1 = np.tanh(
            self.ks.W1 @ x_norm
        )

        h2 = np.tanh(
            self.ks.W2 @ h1
        )

        z = np.concatenate(
            [
                x_norm,
                h2,
            ]
        )

        if z.size != self.n_states:
            raise ValueError(
                f"Lift produced {z.size} states, "
                f"but A expects {self.n_states}."
            )

        return z

    def aircraft_state(
        self,
        phi,
        theta,
        psi,
        p,
        q,
        r,
    ):

        return np.array(
            [
                phi,
                theta,
                np.sin(psi),
                np.cos(psi),
                p,
                q,
                r,
            ],
            dtype=float,
        )


# ============================================================
# Koopman Optimal Control Problem
# ============================================================

class KoopmanOptControl(
    OptimalControlProblem
):

    def __init__(
        self,
        mpc_params: MPCParams,
        casadi_model: KoopManModel,
    ):

        super().__init__(
            mpc_params=mpc_params,
            casadi_model=casadi_model,
        )

    # ========================================================
    # Parameter vector
    #
    # $$
    # P=[z_0;x_{\mathrm{ref}};u_{\mathrm{prev}}]
    # $$
    # ========================================================

    def _parameter_length(self):

        return (
            self.casadi_model.n_states
            +
            self.casadi_model.n_physical_states
            +
            self.casadi_model.n_controls
        )

    # ========================================================
    # Direct discrete Koopman constraints
    # ========================================================

    def set_dynamic_constraints(self):

        n_z = (
            self.casadi_model.n_states
        )

        # Initial condition
        self.g = (
            self.X[:, 0]
            - self.P[:n_z]
        )

        for k in range(self.N):

            z_k = self.X[:, k]
            u_k = self.U[:, k]

            z_next = (
                self.casadi_model.function(
                    z_k,
                    u_k,
                )
            )

            self.g = ca.vertcat(
                self.g,
                self.X[:, k + 1]
                - z_next,
            )

    # ========================================================
    # MPC cost
    #
    # Physical tracking error:
    #
    # $$
    # e_k=x_k-x_{\mathrm{ref}}
    # $$
    #
    # Smooth-control penalty:
    #
    # $$
    # \Delta u_k=u_k-u_{k-1}
    # $$
    #
    # This does NOT pull the inputs toward U_mean.
    # ========================================================

    def compute_dynamics_cost(self):

        cost = 0.0

        Q = ca.DM(
            self.mpc_params.Q
        )

        R = ca.DM(
            self.mpc_params.R
        )

        n_z = (
            self.casadi_model.n_states
        )

        n_x = (
            self.casadi_model.n_physical_states
        )

        n_u = (
            self.casadi_model.n_controls
        )

        x_ref = self.P[
            n_z:
            n_z + n_x
        ]

        u_prev_param = self.P[
            n_z + n_x:
            n_z + n_x + n_u
        ]

        inv_u_std = ca.DM(
            np.diag(
                1.0
                / self.casadi_model.ks.U_std
            )
        )

        for k in range(self.N):

            z_k = (
                self.X[:, k]
            )

            u_k = (
                self.U[:, k]
            )

            x_k = (
                self.casadi_model
                .physical_state(z_k)
            )

            e_x = (
                x_k - x_ref
            )

            if k == 0:

                du = (
                    u_k
                    - u_prev_param
                )

            else:

                du = (
                    u_k
                    - self.U[:, k - 1]
                )

            du_scaled = (
                inv_u_std @ du
            )

            cost += (
                e_x.T @ Q @ e_x
                +
                du_scaled.T @ R @ du_scaled
            )

        # Terminal tracking cost
        x_terminal = (
            self.casadi_model
            .physical_state(
                self.X[:, self.N]
            )
        )

        e_terminal = (
            x_terminal
            - x_ref
        )

        cost += (
            e_terminal.T
            @ Q
            @ e_terminal
        )

        return cost

    def compute_total_cost(self):

        return (
            self.compute_dynamics_cost()
        )

    # ========================================================
    # Forward-rollout initial guess
    # ========================================================

    def build_initial_guess(
        self,
        z0: np.ndarray,
        u_guess: np.ndarray,
    ):

        n_z = (
            self.casadi_model.n_states
        )

        X0 = np.zeros(
            (
                n_z,
                self.N + 1,
            ),
            dtype=float,
        )

        U0 = np.tile(
            np.asarray(
                u_guess,
                dtype=float,
            ).reshape(-1, 1),
            (
                1,
                self.N,
            ),
        )

        X0[:, 0] = z0

        z_roll = z0.copy()

        for k in range(self.N):

            z_roll = np.asarray(
                self.casadi_model.function(
                    z_roll,
                    U0[:, k],
                )
            ).astype(float).reshape(-1)

            X0[:, k + 1] = z_roll

        return X0, U0

    # ========================================================
    # Solve
    # ========================================================

    def solve(
        self,
        x0: np.ndarray,
        xF: np.ndarray,
        u0: np.ndarray,
        p: np.ndarray = None,
    ):

        x0 = np.asarray(
            x0,
            dtype=float,
        ).reshape(-1)

        xF = np.asarray(
            xF,
            dtype=float,
        ).reshape(-1)

        u0 = np.asarray(
            u0,
            dtype=float,
        ).reshape(-1)

        n_z = (
            self.casadi_model.n_states
        )

        n_x = (
            self.casadi_model.n_physical_states
        )

        n_u = (
            self.casadi_model.n_controls
        )

        # ----------------------------------------------------
        # x0 may be physical or already lifted
        # ----------------------------------------------------

        if x0.size == n_x:

            z0 = (
                self.casadi_model
                .lift_state(x0)
            )

        elif x0.size == n_z:

            z0 = x0

        else:

            raise ValueError(
                f"x0 must have either {n_x} physical states "
                f"or {n_z} lifted states."
            )

        if xF.size != n_x:
            raise ValueError(
                f"xF must contain {n_x} physical states."
            )

        if u0.size != n_u:
            raise ValueError(
                f"u0 must contain {n_u} controls."
            )

        z0_dm = ca.DM(
            z0
        ).reshape((-1, 1))

        xF_dm = ca.DM(
            xF
        ).reshape((-1, 1))

        u0_dm = ca.DM(
            u0
        ).reshape((-1, 1))

        X0_np, U0_np = (
            self.build_initial_guess(
                z0=z0,
                u_guess=u0,
            )
        )

        X0 = ca.DM(
            X0_np
        )

        U0 = ca.DM(
            U0_np
        )

        # Only dynamic equalities remain.
        num_constraints = (
            n_z
            * (self.N + 1)
        )

        lbg = ca.DM.zeros(
            num_constraints,
            1,
        )

        ubg = ca.DM.zeros(
            num_constraints,
            1,
        )

        args = {
            "lbg": lbg,
            "ubg": ubg,

            "lbx":
                self.pack_variables_fn(
                    **self.lbx
                )["flat"],

            "ubx":
                self.pack_variables_fn(
                    **self.ubx
                )["flat"],
        }

        args["p"] = ca.vertcat(
            z0_dm,
            xF_dm,
            u0_dm,
        )

        args["x0"] = ca.vertcat(
            ca.reshape(
                X0,
                n_z * (self.N + 1),
                1,
            ),
            ca.reshape(
                U0,
                n_u * self.N,
                1,
            ),
        )

        return self.solver(
            x0=args["x0"],
            lbx=args["lbx"],
            ubx=args["ubx"],
            lbg=args["lbg"],
            ubg=args["ubg"],
            p=args["p"],
        )

    # ========================================================
    # Physical solution
    # ========================================================

    def get_physical_solution(
        self,
        solution: Dict[str, Any],
    ):

        z, u = (
            self.unpack_solution(
                solution
            )
        )

        z_np = np.asarray(
            z.full()
        )

        u_np = np.asarray(
            u.full()
        )

        x_phys = []

        for k in range(
            z_np.shape[1]
        ):

            xk = (
                self.casadi_model
                .physical_state_numpy(
                    z_np[:, k]
                )
            )

            x_phys.append(
                xk
            )

        x_phys = np.asarray(
            x_phys
        ).T

        return {
            "koopman_states": z_np,
            "physical_states": x_phys,
            "controls": u_np,
        }


# ============================================================
# Main
# ============================================================


# ============================================================
# Hybrid Koopman + Plane Plant
# ============================================================

class KoopmanPlanePlant:
    """
    Hybrid plant model.

    Koopman supplies:
        phi, theta, psi, p, q, r dynamics

    Simple fixed-wing kinematics supply:
        x, y, z translation

    Separate first-order airspeed dynamics supply:
        v

    Full returned plant state:

        [x, y, z, phi, theta, psi, v, p, q, r]

    IMPORTANT:
    The full lifted Koopman state is preserved internally.
    """

    def __init__(
        self,
        koopman_model: KoopManModel,
        dt: float,
        initial_speed: float = 25.0,
        airspeed_tau: float = 0.5,
    ):

        self.koopman_model = koopman_model
        self.dt = float(dt)

        self.airspeed_tau = float(
            airspeed_tau
        )

        self.position = np.zeros(
            3,
            dtype=float,
        )

        self.v = float(
            initial_speed
        )

        self.z_koopman = None

        self.physical_attitude_state = None

    # ========================================================
    # Initialize plant
    # ========================================================

    def initialize(
        self,
        position: np.ndarray,
        phi: float,
        theta: float,
        psi: float,
        p: float = 0.0,
        q: float = 0.0,
        r: float = 0.0,
        airspeed: float = 25.0,
    ):

        self.position = np.asarray(
            position,
            dtype=float,
        ).reshape(3)

        self.v = float(
            airspeed
        )

        x_attitude = np.array([
            phi,
            theta,
            np.sin(psi),
            np.cos(psi),
            p,
            q,
            r,
        ], dtype=float)

        # Lift once for model-in-the-loop propagation.
        self.z_koopman = (
            self.koopman_model
            .lift_state(
                x_attitude
            )
        )

        self.physical_attitude_state = (
            x_attitude.copy()
        )

    # ========================================================
    # Koopman state -> physical aircraft attitude/rates
    # ========================================================

    def get_attitude_state(self):

        if self.z_koopman is None:
            raise RuntimeError(
                "Plant has not been initialized."
            )

        x_att = (
            self.koopman_model
            .physical_state_numpy(
                self.z_koopman
            )
        )

        phi = float(
            x_att[0]
        )

        theta = float(
            x_att[1]
        )

        psi = float(
            np.arctan2(
                x_att[2],
                x_att[3],
            )
        )

        p = float(
            x_att[4]
        )

        q = float(
            x_att[5]
        )

        r = float(
            x_att[6]
        )

        return (
            phi,
            theta,
            psi,
            p,
            q,
            r,
        )

    # ========================================================
    # Return full physical plant state
    # ========================================================

    def get_full_state(self):

        (
            phi,
            theta,
            psi,
            p,
            q,
            r,
        ) = self.get_attitude_state()

        return np.array([
            self.position[0],
            self.position[1],
            self.position[2],
            phi,
            theta,
            psi,
            self.v,
            p,
            q,
            r,
        ], dtype=float)

    # ========================================================
    # One hybrid plant step
    # ========================================================

    def step(
        self,
        koopman_control: np.ndarray,
        v_cmd: float = None,
    ) -> np.ndarray:

        if self.z_koopman is None:
            raise RuntimeError(
                "Plant has not been initialized."
            )

        koopman_control = np.asarray(
            koopman_control,
            dtype=float,
        ).reshape(-1)

        if (
            koopman_control.size
            != self.koopman_model.n_controls
        ):
            raise ValueError(
                "koopman_control must contain "
                f"{self.koopman_model.n_controls} controls."
            )

        # ====================================================
        # 1. Koopman attitude dynamics
        #
        # $$
        # z_{k+1}
        # =
        # A z_k
        # +
        # B_{\mathrm{real}}u_k
        # +
        # b_{\mathrm{real}}
        # $$
        # ====================================================

        self.z_koopman = np.asarray(
            self.koopman_model.function(
                self.z_koopman,
                koopman_control,
            )
        ).astype(float).reshape(-1)

        # Keep physical yaw representation valid OUTSIDE MPC.
        self.z_koopman = (
            self.koopman_model
            .project_yaw_numpy(
                self.z_koopman
            )
        )

        # ====================================================
        # 2. Recover attitude and body rates
        # ====================================================

        (
            phi,
            theta,
            psi,
            p,
            q,
            r,
        ) = self.get_attitude_state()

        # ====================================================
        # 3. Airspeed dynamics
        #
        # $$
        # \dot v
        # =
        # \frac{v_{\mathrm{cmd}}-v}{\tau_v}
        # $$
        # ====================================================

        if v_cmd is not None:

            v_dot = (
                float(v_cmd)
                - self.v
            ) / self.airspeed_tau

            self.v += (
                self.dt
                * v_dot
            )

        # ====================================================
        # 4. Translational plane kinematics
        #
        # Same convention as the original Plane model:
        #
        # $$
        # \dot x
        # =
        # v\cos\theta\cos\psi
        # $$
        #
        # $$
        # \dot y
        # =
        # v\cos\theta\sin\psi
        # $$
        #
        # $$
        # \dot z
        # =
        # -v\sin\theta
        # $$
        # ====================================================

        x_dot = (
            self.v
            * np.cos(theta)
            * np.cos(psi)
        )

        y_dot = (
            self.v
            * np.cos(theta)
            * np.sin(psi)
        )

        z_dot = (
            -self.v
            * np.sin(theta)
        )

        # Forward Euler translational integration
        self.position[0] += (
            self.dt
            * x_dot
        )

        self.position[1] += (
            self.dt
            * y_dot
        )

        self.position[2] += (
            self.dt
            * z_dot
        )

        self.physical_attitude_state = np.array([
            phi,
            theta,
            np.sin(psi),
            np.cos(psi),
            p,
            q,
            r,
        ], dtype=float)

        return self.get_full_state()



if __name__ == "__main__":

    # ========================================================
    # Load learned Koopman model
    # ========================================================

    koopman, dt = load_koopman_mat(
        mat_path="Koopman/Koopman_ABCD.mat",

        checkpoint_path=(
            "Koopman/"
            "Koopman_x=[phi,theta,sinpsi,cospsi,p,q,r]"
            "_u=[phiD,thetaD,sinpsiD,cospsiD,ThO,C1,C2,C4]"
            "-Lift[7, 17, 17]-Act(tanh)-StableA(1)-best.pt"
        ),
    )

    model = KoopManModel(
        koopman_state_space=koopman,
        dt_val=dt,
    )

    print()
    print("Koopman dimension:", model.n_states)
    print("Physical Koopman states:", model.n_physical_states)
    print("Koopman controls:", model.n_controls)
    print("dt:", model.dt_val)

    # ========================================================
    # Lifted-state bounds
    #
    # These are numerical bounds on z, NOT aircraft limits.
    # ========================================================

    state_limits = {}

    for i in range(
        model.n_states
    ):
        state_limits[f"z{i}"] = {
            "min": -20.0,
            "max": 20.0,
        }

    # ========================================================
    # Koopman control bounds
    #
    # Input order:
    #
    # [
    #   phiD,
    #   thetaD,
    #   sinpsiD,
    #   cospsiD,
    #   ThO,
    #   C1,
    #   C2,
    #   C4
    # ]
    # ========================================================

    control_limits = {

        "phiD": {
            "min": np.deg2rad(-45.0),
            "max": np.deg2rad(45.0),
        },

        "thetaD": {
            "min": np.deg2rad(-20.0),
            "max": np.deg2rad(20.0),
        },

        "sinpsiD": {
            "min": -1.0,
            "max": 1.0,
        },

        "cospsiD": {
            "min": -1.0,
            "max": 1.0,
        },

        "ThO": {
            "min": 0.0,
            "max": 100.0,
        },

        "C1": {
            "min": -1.0,
            "max": 1.0,
        },

        "C2": {
            "min": -1.0,
            "max": 1.0,
        },

        "C4": {
            "min": -1.0,
            "max": 1.0,
        },
    }

    model.set_state_limits(
        state_limits
    )

    model.set_control_limits(
        control_limits
    )

    # ========================================================
    # MPC weights
    #
    # Q:
    #
    # [phi, theta, sin(psi), cos(psi), p, q, r]
    #
    # R:
    #
    # penalty on Delta-u
    # ========================================================

    Q = np.diag([
        np.deg2rad(20.0),   # phi
        np.deg2rad(20.0),   # theta
        np.deg2rad(10.0),   # sin psi
        np.deg2rad(10.0),   # cos psi
        np.deg2rad(5.0),    # p
        np.deg2rad(5.0),    # q
        np.deg2rad(5.0),    # r
    ])

    R = np.diag([
        0.10,   # Delta phiD
        0.10,   # Delta thetaD
        0.05,   # Delta sinpsiD
        0.05,   # Delta cospsiD
        0.02,   # Delta ThO
        0.02,   # Delta C1
        0.02,   # Delta C2
        0.02,   # Delta C4
    ])

    # ========================================================
    # Horizon
    #
    # $$
    # T_H = N \Delta t
    # $$
    #
    # With dt = 0.02 s and N = 15:
    #
    # $$
    # T_H = 0.30 \text{ s}
    # $$
    # ========================================================

    N = 15

    mpc_params = MPCParams(
        Q=Q,
        R=R,
        N=N,
        dt=model.dt_val,
    )

    opt_control = KoopmanOptControl(
        mpc_params=mpc_params,
        casadi_model=model,
    )

    opt_control.init_optimization()

    # ========================================================
    # Hybrid plant
    #
    # Koopman:
    #   attitude/rates
    #
    # Plane:
    #   translation
    # ========================================================

    AIRSPEED_CMD = 25.0

    plant = KoopmanPlanePlant(
        koopman_model=model,
        dt=model.dt_val,
        initial_speed=AIRSPEED_CMD,
        airspeed_tau=0.5,
    )

    # ========================================================
    # Initial aircraft condition
    # ========================================================

    phi0 = np.deg2rad(
        10.0
    )

    theta0 = np.deg2rad(
        2.0
    )

    psi0 = np.deg2rad(
        45.0
    )

    plant.initialize(
        position=np.array([
            0.0,
            0.0,
            0.0,
        ]),
        phi=phi0,
        theta=theta0,
        psi=psi0,
        p=0.0,
        q=0.0,
        r=0.0,
        airspeed=AIRSPEED_CMD,
    )

    # ========================================================
    # Attitude target for Koopman MPC
    #
    # This controller currently regulates attitude/rates.
    # Position is propagated by the hybrid plane but is NOT
    # yet included in the MPC objective.
    # ========================================================

    psi_ref = np.deg2rad(
        180.0
    )

    x_target = np.array([
        0.0,                 # phi
        0.0,                 # theta
        np.sin(psi_ref),     # sin psi
        np.cos(psi_ref),     # cos psi
        0.0,                 # p
        0.0,                 # q
        0.0,                 # r
    ], dtype=float)

    # ========================================================
    # Initial input
    # ========================================================

    u_current = (
        koopman.U_mean.copy()
    )

    # Start desired-heading channels consistently.
    u_current[2] = np.sin(
        psi_ref
    )

    u_current[3] = np.cos(
        psi_ref
    )

    # ========================================================
    # Stop criterion
    # ========================================================

    def custom_stop_criteria(
        current_state: np.ndarray,
        target_state: np.ndarray,
    ) -> bool:

        current_state = np.asarray(
            current_state,
            dtype=float,
        ).reshape(-1)

        target_state = np.asarray(
            target_state,
            dtype=float,
        ).reshape(-1)

        phi_error = (
            current_state[0]
            - target_state[0]
        )

        theta_error = (
            current_state[1]
            - target_state[1]
        )

        p_error = (
            current_state[4]
            - target_state[4]
        )

        q_error = (
            current_state[5]
            - target_state[5]
        )

        r_error = (
            current_state[6]
            - target_state[6]
        )

        psi_current = np.arctan2(
            current_state[2],
            current_state[3],
        )

        psi_target_local = np.arctan2(
            target_state[2],
            target_state[3],
        )

        psi_error = (
            psi_current
            - psi_target_local
            + np.pi
        ) % (2.0 * np.pi) - np.pi

        attitude_tol = np.deg2rad(
            2.0
        )

        yaw_tol = np.deg2rad(
            3.0
        )

        rate_tol = np.deg2rad(
            2.0
        )

        return (
            abs(phi_error) < attitude_tol
            and abs(theta_error) < attitude_tol
            and abs(psi_error) < yaw_tol
            and abs(p_error) < rate_tol
            and abs(q_error) < rate_tol
            and abs(r_error) < rate_tol
        )

    # ========================================================
    # Logging
    # ========================================================

    initial_full_state = (
        plant.get_full_state()
    )

    plant_history = [
        initial_full_state.copy()
    ]

    attitude_history = [
        plant.physical_attitude_state.copy()
    ]

    koopman_state_history = [
        plant.z_koopman.copy()
    ]

    control_history = []

    prediction_history = []

    solve_time_history = []

    # ========================================================
    # Closed-loop hybrid simulation
    # ========================================================

    MAX_STEPS = 250

    import time

    for step in range(
        MAX_STEPS
    ):

        print()
        print(
            f"================ STEP {step} ================"
        )

        # ----------------------------------------------------
        # Current physical attitude state:
        #
        # [phi, theta, sinpsi, cospsi, p, q, r]
        # ----------------------------------------------------

        x_current = (
            plant.physical_attitude_state
            .copy()
        )

        # ----------------------------------------------------
        # Current full lifted Koopman state
        # ----------------------------------------------------

        z_current = (
            plant.z_koopman
            .copy()
        )

        if custom_stop_criteria(
            x_current,
            x_target,
        ):

            print(
                "Stop criteria satisfied."
            )
            break

        # ====================================================
        # Solve MPC
        # ====================================================

        solve_start = (
            time.perf_counter()
        )

        solution = (
            opt_control.solve(
                x0=z_current,
                xF=x_target,
                u0=u_current,
            )
        )

        solve_elapsed = (
            time.perf_counter()
            - solve_start
        )

        solve_time_history.append(
            solve_elapsed
        )

        result = (
            opt_control
            .get_physical_solution(
                solution
            )
        )

        X_pred = result[
            "physical_states"
        ]

        U_pred = result[
            "controls"
        ]

        prediction_history.append(
            X_pred.copy()
        )

        # ====================================================
        # Receding-horizon action:
        # apply only first optimized input
        # ====================================================

        u_command = (
            U_pred[:, 0]
            .copy()
        )

        control_history.append(
            u_command.copy()
        )

        # ====================================================
        # Actuate hybrid plane plant
        # ====================================================

        full_state = (
            plant.step(
                koopman_control=u_command,
                v_cmd=AIRSPEED_CMD,
            )
        )

        # ====================================================
        # Log resulting state
        # ====================================================

        plant_history.append(
            full_state.copy()
        )

        attitude_history.append(
            plant.physical_attitude_state
            .copy()
        )

        koopman_state_history.append(
            plant.z_koopman
            .copy()
        )

        # Delta-u reference for next solve.
        u_current = (
            u_command.copy()
        )

        print(
            "Solve time [s]:",
            solve_elapsed,
        )

        print(
            "Position [m]:",
            plant.position,
        )

        print(
            "Attitude [deg]:",
            np.rad2deg(
                full_state[3:6]
            ),
        )

        print(
            "Airspeed [m/s]:",
            full_state[6],
        )

        print(
            "Applied Koopman control:",
            u_command,
        )

    else:

        print(
            "Maximum number of closed-loop iterations reached."
        )

    # ========================================================
    # Convert logs
    # ========================================================

    plant_history = np.asarray(
        plant_history
    )

    attitude_history = np.asarray(
        attitude_history
    )

    koopman_state_history = np.asarray(
        koopman_state_history
    )

    control_history = np.asarray(
        control_history
    )

    solve_time_history = np.asarray(
        solve_time_history
    )

    # Full-state column definitions
    X_POS = 0
    Y_POS = 1
    Z_POS = 2
    PHI = 3
    THETA = 4
    PSI = 5
    SPEED = 6
    P_RATE = 7
    Q_RATE = 8
    R_RATE = 9

    print()
    print("======================================")
    print("HYBRID SIMULATION COMPLETE")
    print("======================================")
    print(
        "Closed-loop iterations:",
        len(control_history),
    )

    print(
        "Final full plant state:",
        plant_history[-1],
    )

    if solve_time_history.size > 0:

        print(
            "Mean MPC solve time [s]:",
            solve_time_history.mean(),
        )

        print(
            "Max MPC solve time [s]:",
            solve_time_history.max(),
        )

    # ========================================================
    # Time vectors
    # ========================================================

    t_state = (
        np.arange(
            plant_history.shape[0]
        )
        * model.dt_val
    )

    t_control = (
        np.arange(
            control_history.shape[0]
        )
        * model.dt_val
    )

    # ========================================================
    # FIGURE 1 — Attitude tracking
    # ========================================================

    fig1, axes1 = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        sharex=True,
    )

    attitude_labels = [
        r"$\phi$ [deg]",
        r"$\theta$ [deg]",
        r"$\psi$ [deg]",
    ]

    refs = [
        0.0,
        0.0,
        np.rad2deg(
            psi_ref
        ),
    ]

    state_indices = [
        PHI,
        THETA,
        PSI,
    ]

    for ax, idx, label, ref in zip(
        axes1,
        state_indices,
        attitude_labels,
        refs,
    ):

        ax.plot(
            t_state,
            np.rad2deg(
                plant_history[:, idx]
            ),
            label="Hybrid plant",
            linewidth=1.8,
        )

        ax.axhline(
            ref,
            linestyle="--",
            label="Reference",
        )

        ax.set_ylabel(
            label
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

    axes1[-1].set_xlabel(
        "Time [s]"
    )

    fig1.suptitle(
        "Koopman MPC + Plane Attitude Tracking"
    )

    fig1.tight_layout()

    # ========================================================
    # FIGURE 2 — Body-rate tracking
    # ========================================================

    fig2, axes2 = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        sharex=True,
    )

    rate_indices = [
        P_RATE,
        Q_RATE,
        R_RATE,
    ]

    rate_labels = [
        r"$p$ [deg/s]",
        r"$q$ [deg/s]",
        r"$r$ [deg/s]",
    ]

    for ax, idx, label in zip(
        axes2,
        rate_indices,
        rate_labels,
    ):

        ax.plot(
            t_state,
            np.rad2deg(
                plant_history[:, idx]
            ),
            label="Hybrid plant",
            linewidth=1.8,
        )

        ax.axhline(
            0.0,
            linestyle="--",
            label="Reference",
        )

        ax.set_ylabel(
            label
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

    axes2[-1].set_xlabel(
        "Time [s]"
    )

    fig2.suptitle(
        "Koopman MPC Body-Rate Tracking"
    )

    fig2.tight_layout()

    # ========================================================
    # FIGURE 3 — Position vs time
    # ========================================================

    fig3, axes3 = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        sharex=True,
    )

    position_labels = [
        "x [m]",
        "y [m]",
        "z [m]",
    ]

    position_indices = [
        X_POS,
        Y_POS,
        Z_POS,
    ]

    for ax, idx, label in zip(
        axes3,
        position_indices,
        position_labels,
    ):

        ax.plot(
            t_state,
            plant_history[:, idx],
            linewidth=1.8,
        )

        ax.set_ylabel(
            label
        )

        ax.grid(
            True,
            alpha=0.3,
        )

    axes3[-1].set_xlabel(
        "Time [s]"
    )

    fig3.suptitle(
        "Hybrid Plane Position"
    )

    fig3.tight_layout()

    # ========================================================
    # FIGURE 4 — XY ground track
    # ========================================================

    fig4, ax4 = plt.subplots(
        figsize=(9, 8)
    )

    ax4.plot(
        plant_history[:, X_POS],
        plant_history[:, Y_POS],
        linewidth=2.0,
    )

    ax4.scatter(
        plant_history[0, X_POS],
        plant_history[0, Y_POS],
        label="Start",
    )

    ax4.scatter(
        plant_history[-1, X_POS],
        plant_history[-1, Y_POS],
        label="End",
    )

    ax4.set_xlabel(
        "x [m]"
    )

    ax4.set_ylabel(
        "y [m]"
    )

    ax4.set_title(
        "Hybrid Plane Ground Track"
    )

    ax4.axis(
        "equal"
    )

    ax4.grid(
        True,
        alpha=0.3,
    )

    ax4.legend()

    fig4.tight_layout()

    # ========================================================
    # FIGURE 5 — Airspeed
    # ========================================================

    fig5, ax5 = plt.subplots(
        figsize=(11, 5)
    )

    ax5.plot(
        t_state,
        plant_history[:, SPEED],
        linewidth=1.8,
        label="Airspeed",
    )

    ax5.axhline(
        AIRSPEED_CMD,
        linestyle="--",
        label="Command",
    )

    ax5.set_xlabel(
        "Time [s]"
    )

    ax5.set_ylabel(
        "Airspeed [m/s]"
    )

    ax5.set_title(
        "Hybrid Plane Airspeed"
    )

    ax5.grid(
        True,
        alpha=0.3,
    )

    ax5.legend()

    fig5.tight_layout()

    # ========================================================
    # FIGURE 6 — Koopman MPC controls
    # ========================================================

    if control_history.shape[0] > 0:

        control_names = [
            r"$\phi_D$",
            r"$\theta_D$",
            r"$\sin(\psi_D)$",
            r"$\cos(\psi_D)$",
            "ThO",
            "C1",
            "C2",
            "C4",
        ]

        fig6, axes6 = plt.subplots(
            4,
            2,
            figsize=(13, 11),
            sharex=True,
        )

        axes6 = axes6.flatten()

        for i in range(
            model.n_controls
        ):

            if i in [0, 1]:

                values = np.rad2deg(
                    control_history[:, i]
                )

                ylabel = (
                    control_names[i]
                    + " [deg]"
                )

            else:

                values = (
                    control_history[:, i]
                )

                ylabel = (
                    control_names[i]
                )

            axes6[i].step(
                t_control,
                values,
                where="post",
                linewidth=1.5,
            )

            axes6[i].set_ylabel(
                ylabel
            )

            axes6[i].grid(
                True,
                alpha=0.3,
            )

        axes6[-2].set_xlabel(
            "Time [s]"
        )

        axes6[-1].set_xlabel(
            "Time [s]"
        )

        fig6.suptitle(
            "Koopman MPC Control Commands"
        )

        fig6.tight_layout()

    # ========================================================
    # FIGURE 7 — Roll prediction horizons
    # ========================================================

    if len(
        prediction_history
    ) > 0:

        fig7, ax7 = plt.subplots(
            figsize=(11, 6)
        )

        ax7.plot(
            t_state,
            np.rad2deg(
                plant_history[:, PHI]
            ),
            linewidth=2.0,
            label="Plant roll",
        )

        prediction_stride = 10

        for step in range(
            0,
            len(prediction_history),
            prediction_stride,
        ):

            X_horizon = (
                prediction_history[step]
            )

            horizon_time = (
                step * model.dt_val
                + np.arange(
                    X_horizon.shape[1]
                ) * model.dt_val
            )

            ax7.plot(
                horizon_time,
                np.rad2deg(
                    X_horizon[0, :]
                ),
                linestyle="--",
                alpha=0.5,
            )

        ax7.axhline(
            0.0,
            linestyle="--",
            label="Roll reference",
        )

        ax7.set_xlabel(
            "Time [s]"
        )

        ax7.set_ylabel(
            r"$\phi$ [deg]"
        )

        ax7.set_title(
            "Koopman MPC Roll Prediction Horizons"
        )

        ax7.grid(
            True,
            alpha=0.3,
        )

        ax7.legend()

        fig7.tight_layout()

    # ========================================================
    # FIGURE 8 — MPC solve time
    # ========================================================

    if solve_time_history.size > 0:

        fig8, ax8 = plt.subplots(
            figsize=(11, 5)
        )

        ax8.plot(
            t_control,
            solve_time_history,
            linewidth=1.5,
        )

        ax8.axhline(
            model.dt_val,
            linestyle="--",
            label=f"dt = {model.dt_val:.3f} s",
        )

        ax8.set_xlabel(
            "Simulation time [s]"
        )

        ax8.set_ylabel(
            "Solve time [s]"
        )

        ax8.set_title(
            "MPC Solve Time"
        )

        ax8.grid(
            True,
            alpha=0.3,
        )

        ax8.legend()

        fig8.tight_layout()
        
    # save all figures
    fig1.savefig("figure1.png")
    fig2.savefig("figure2.png")
    fig3.savefig("figure3.png")
    fig4.savefig("figure4.png")
    fig5.savefig("figure5.png")
    fig6.savefig("figure6.png")
    fig7.savefig("figure7.png")
    fig8.savefig("figure8.png")

    plt.show()
