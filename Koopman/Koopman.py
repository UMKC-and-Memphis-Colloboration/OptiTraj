# ============================================================
# BLOCK 1 — Libraries & Global Settings
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.io import savemat
import json

# Your custom modules (must exist)
from Library import RBFLayer1, ListModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.manual_seed(123)
np.random.seed(123)

# Training flags
Stable_A = 1    # 1: stable-A parameterization, 0: unconstrained A
multi_step = 1    # selection metric: multi-step vs one-step

# ============================================================
# BLOCK 2 — Load CSV + Build (X,Y,U)
#   x = [phi, theta, sin(psi), cos(psi), p, q, r]
#   u = [phi_des, theta_des, sin(psi_des), cos(psi_des), ThO, C1, C2, C4]
# ============================================================

filepath = "Koopman/csv_files/calibration_data.csv"

df = pd.read_csv(filepath).dropna().reset_index(drop=True)

t = df["t"].to_numpy(dtype=float)
dt = float(np.median(np.diff(t)))
print("N =", len(df), "dt~", dt)

# ---- Attitude (deg -> rad) ----
phi = np.deg2rad(df["ATT_Roll"].to_numpy(dtype=float))
theta = np.deg2rad(df["ATT_Pitch"].to_numpy(dtype=float))
psi = np.deg2rad(df["ATT_Yaw"].to_numpy(dtype=float))

sin_psi = np.sin(psi)
cos_psi = np.cos(psi)

# ---- Body rates ----
p = df["IMU_GyrX"].to_numpy(dtype=float)
q = df["IMU_GyrY"].to_numpy(dtype=float)
r = df["IMU_GyrZ"].to_numpy(dtype=float)

x = np.vstack([phi, theta, sin_psi, cos_psi, p, q, r]
              ).T.astype(np.float32)  # (N,7)

# ---- Desired attitude (deg -> rad) ----
phi_d = np.deg2rad(df["ATT_DesRoll"].to_numpy(dtype=float))
theta_d = np.deg2rad(df["ATT_DesPitch"].to_numpy(dtype=float))
psi_d = np.deg2rad(df["ATT_DesYaw"].to_numpy(dtype=float))

sin_psi_d = np.sin(psi_d)
cos_psi_d = np.cos(psi_d)

# ---- Throttle output (leave physical; normalize later) ----
tho = df["CTUN_ThO"].to_numpy(dtype=float)

# ---- Actuator outputs (PWM) -> approx normalized [-1,1] ----
c1 = (df["RCOU_C1"].to_numpy(dtype=float) - 1500.0) / 500.0
c2 = (df["RCOU_C2"].to_numpy(dtype=float) - 1500.0) / 500.0
c4 = (df["RCOU_C4"].to_numpy(dtype=float) - 1500.0) / 500.0

u = np.vstack([phi_d, theta_d, sin_psi_d, cos_psi_d, tho,
              c1, c2, c4]).T.astype(np.float32)  # (N,8)

# One-step pairs
X = x[:-1, :]
Y = x[1:, :]
U = u[:-1, :]

print("Raw pairs:", X.shape, Y.shape, U.shape)

# Normalization
X_mean = X.mean(axis=0, keepdims=True)
X_std = X.std(axis=0, keepdims=True) + 1e-6
U_mean = U.mean(axis=0, keepdims=True)
U_std = U.std(axis=0, keepdims=True) + 1e-6

Xn = (X - X_mean) / X_std
Yn = (Y - X_mean) / X_std
Un = (U - U_mean) / U_std

X_t = torch.from_numpy(Xn).float()
Y_t = torch.from_numpy(Yn).float()
U_t = torch.from_numpy(Un).float()

# Time-ordered split (train/test)
N = X_t.shape[0]
split = int(0.8 * N)

Xtrain, Ytrain, Utrain = X_t[:split], Y_t[:split], U_t[:split]
Xtest,  Ytest,  Utest = X_t[split:], Y_t[split:], U_t[split:]

num_state = Xtrain.shape[1]  # 7
num_input = Utrain.shape[1]  # 8

print("Train:", Xtrain.shape, Utrain.shape,
      "| Test:", Xtest.shape, Utest.shape)
print("num_state =", num_state, "num_input =", num_input)

# ============================================================
# BLOCK 3 — Datasets & DataLoaders
# ============================================================


class PairDataset(Dataset):
    def __init__(self, X, Y, U):
        self.X = X
        self.Y = Y
        self.U = U

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.U[idx]


Train_dataset = PairDataset(Xtrain, Ytrain, Utrain)
Test_dataset = PairDataset(Xtest,  Ytest,  Utest)

num_sample = len(Train_dataset)

learning_rate = 1e-3
batch_size = max(128, int(num_sample/10))

train_loader = DataLoader(Train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    Test_dataset,  batch_size=len(Test_dataset), shuffle=False)

print("batch_size =", batch_size, "learning_rate =", learning_rate)

# ============================================================
# BLOCK 4 — Model Definition (num_state=7, num_input=8)
# ============================================================

lift_width = 128  # gives dimA = 7 + 128 = 135 (normal for Koopman lifts)

params = {
    "num_state": num_state,
    "num_input": num_input,
    "lift_shape": [num_state, lift_width, lift_width],
    "activation": "tanh",
}

SessionName = (
    f"Koopman_x=[phi,theta,sinpsi,cospsi,p,q,r]"
    f"_u=[phiD,thetaD,sinpsiD,cospsiD,ThO,C1,C2,C4]"
    f"-Lift{params['lift_shape']}-Act({params['activation']})"
    f"-StableA({Stable_A})"
)

print("SessionName:", SessionName)


class Encoder(nn.Module):
    def __init__(self, params, name="encoder"):
        super().__init__()
        self.activation = params["activation"]
        self.shape = params["lift_shape"]

        self.aux_layers = ListModule(self, f"{name}")
        if self.activation == "rbf":
            self.rbf = RBFLayer1(
                in_features_dim=params["num_state"],
                num_kernels=params["lift_shape"][-1],
                initial_centers_parameter=False,
                constant_centers_parameter=False,
            )

        for j in range(len(self.shape) - 1):
            self.aux_layers.append(
                nn.Linear(self.shape[j], self.shape[j + 1], bias=False))

    def forward(self, x):
        x_true = x
        if self.activation == "rbf":
            z = self.rbf(x)
        else:
            z = x
            for layer in self.aux_layers:
                if self.activation == "tanh":
                    z = torch.tanh(layer(z))
                elif self.activation == "relu":
                    z = F.relu(layer(z))
                elif self.activation == "sigmoid":
                    z = torch.sigmoid(layer(z))
                else:
                    raise ValueError(f"Unknown activation: {self.activation}")
        return torch.cat((x_true, z), dim=-1)  # z_lift = [x; features]


class MyArchitechture(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.numState = params["num_state"]
        self.numInput = params["num_input"]

        self.lift = Encoder(params, name="liftlayer")
        self.dimA = params["lift_shape"][-1] + self.numState  # 7 + 128 = 135

        # These are used when Stable_A==0 (or for B always)
        self.linA = nn.Linear(self.dimA, self.dimA, bias=False)
        self.linB = nn.Linear(self.numInput, self.dimA, bias=False)

        # Stable A parameterization variables
        self.R = nn.Parameter(torch.rand(self.dimA, self.dimA))
        self.L = nn.Parameter(torch.rand(self.dimA * 2, self.dimA * 2))
        self.register_buffer("epsI", 1e-8 * torch.eye(self.dimA * 2))

    def _stable_A(self):
        dim = self.dimA
        M = self.L @ self.L.T + self.epsI
        Fm = M[dim:, :dim]
        P = M[dim:, dim:]
        Skew = (self.R - self.R.T) / 2.0
        E = (M[:dim, :dim] + P) / 2.0 + Skew
        A = torch.linalg.solve(E, Fm)  # (dimA,dimA)
        return A

    def forward(self, x1, x2, u):
        # lifted target
        y = self.lift(x2)
        z1 = self.lift(x1)

        if Stable_A == 1:
            A = self._stable_A()
            # IMPORTANT: B from linB, A functional
            z_next = z1 @ A.T + self.linB(u)
        else:
            z_next = self.linA(z1) + self.linB(u)

        return y, z_next


model = MyArchitechture(params).to(device)
print(model)
print("dimA =", model.dimA)

# ============================================================
# BLOCK 5 — Training (Stable-A SAFE) + Multi-step + yaw circle + yaw phase loss
# ============================================================

loss_fcn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 80
best_score = float("inf")
best_epoch = None

SIN_IDX = 2
COS_IDX = 3

# Loss weights (tuneable)
alpha_1s_lift = 0.15     # weight on 1-step lifted loss
alpha_ms_state = 1.0     # weight on multi-step state loss
lam_circle = 80.0        # enforces sin^2+cos^2 ~ 1
lam_phase = 2.0          # penalize yaw phase error via atan2(sin,cos)

H = 10
K_windows = 96
print(f"H={H}, K_windows={K_windows}, lam_circle={lam_circle}, lam_phase={lam_phase}")


def project_unit_circle(x):
    s = x[..., SIN_IDX]
    c = x[..., COS_IDX]
    n = torch.sqrt(s*s + c*c + 1e-12)
    s2 = s / n
    c2 = c / n
    parts = []
    for j in range(x.shape[-1]):
        if j == SIN_IDX:
            parts.append(s2.unsqueeze(-1))
        elif j == COS_IDX:
            parts.append(c2.unsqueeze(-1))
        else:
            parts.append(x[..., j].unsqueeze(-1))
    return torch.cat(parts, dim=-1)


def unit_circle_penalty(x_pred):
    s = x_pred[..., SIN_IDX]
    c = x_pred[..., COS_IDX]
    return torch.mean((s*s + c*c - 1.0)**2)


def yaw_phase_loss(x_true, x_pred):

    psi_t = torch.atan2(x_true[..., SIN_IDX], x_true[..., COS_IDX])
    psi_p = torch.atan2(x_pred[..., SIN_IDX], x_pred[..., COS_IDX])
    # wrap to [-pi,pi]
    d = (psi_p - psi_t + np.pi) % (2*np.pi) - np.pi
    return torch.mean(d*d)


# weights for multi-step state MSE (normalized-space)
w_state = torch.tensor([1.0, 1.5, 4.0, 4.0, 2.2, 2.2, 1.8], device=device)


def weighted_state_mse(x_true, x_pred):
    diff2 = (x_true - x_pred) ** 2
    w_view = w_state.view(*([1] * (diff2.ndim - 1)), -1)
    return torch.mean(diff2 * w_view)


def get_A_B(model):
    if Stable_A == 1:
        A = model._stable_A()
    else:
        A = model.linA.weight
    Bm = model.linB.weight
    return A, Bm


def multistep_rollout_pred(model, x0, U_seq):

    z = model.lift(x0)     # (B,dimA)
    A, Bm = get_A_B(model)

    preds = []
    for i in range(U_seq.shape[1]):
        ui = U_seq[:, i, :]          # (B,8)
        z_next = z @ A.T + ui @ Bm.T
        x_next = z_next[:, :num_state]
        x_next = project_unit_circle(x_next)
        preds.append(x_next.unsqueeze(1))
        z = torch.cat([x_next, z_next[:, num_state:]], dim=1)
    return torch.cat(preds, dim=1)


@torch.no_grad()
def strict_eval_multistep(model, X0, U, X_true):

    model.eval()
    z = model.lift(X0)

    if Stable_A == 1:
        A = model._stable_A().detach()
    else:
        A = model.linA.weight.detach()
    Bm = model.linB.weight.detach()

    T = U.shape[0]
    pred = torch.zeros((T, num_state), device=U.device)
    pred[0:1, :] = X0

    for k in range(T - 1):
        z_next = z @ A.T + U[k:k+1, :] @ Bm.T
        x_next = project_unit_circle(z_next[:, :num_state])
        pred[k+1:k+2, :] = x_next
        z = torch.cat([x_next, z_next[:, num_state:]], dim=1)

    mse = loss_fcn(pred, X_true).item()
    sc = torch.mean(pred[:, SIN_IDX]**2 + pred[:, COS_IDX]**2).item()
    return mse, sc, pred


loss_train = []
loss_test_one = []
loss_test_multi = []

for epoch in range(num_epochs):
    # ----------------------------
    # 1) One-step lifted loss (shuffled mini-batches)
    # ----------------------------
    model.train()
    running = 0.0
    for x1, x2, u in train_loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        u = u.to(device)

        y, yhat = model(x1, x2, u)
        loss_1s = loss_fcn(y, yhat)

        optimizer.zero_grad(set_to_none=True)
        (alpha_1s_lift * loss_1s).backward()
        optimizer.step()

        running += (alpha_1s_lift * loss_1s).item()

    avg_train = running / max(1, len(train_loader))
    loss_train.append(avg_train)

    # ----------------------------
    # 2) Multi-step state loss (contiguous windows)
    # ----------------------------
    model.train()
    if Xtrain.shape[0] > (H + 2):
        # contiguous windows start indices
        idx0 = torch.randint(
            low=0, high=Xtrain.shape[0] - (H + 1), size=(K_windows,), device=device)

        x0 = Xtrain[idx0, :].to(device)  # (K,7)
        Uwin = torch.stack([Utrain[idx0 + j, :].to(device)
                           for j in range(H)], dim=1)   # (K,H,8)
        Xtrue = torch.stack([Xtrain[idx0 + j + 1, :].to(device)
                            for j in range(H)], dim=1)  # (K,H,7)

        Xpred = multistep_rollout_pred(model, x0, Uwin)

        loss_ms = weighted_state_mse(Xtrue, Xpred)
        loss_c = unit_circle_penalty(Xpred)
        loss_p = yaw_phase_loss(Xtrue, Xpred)

        loss_total = alpha_ms_state * loss_ms + lam_circle * loss_c + lam_phase * loss_p

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        optimizer.step()

    # ----------------------------
    # 3) Validation
    # ----------------------------
    model.eval()
    with torch.no_grad():
        x1t, x2t, ut = next(iter(test_loader))
        x1t = x1t.to(device)
        x2t = x2t.to(device)
        ut = ut.to(device)
        yt, yhatt = model(x1t, x2t, ut)
        one_step = loss_fcn(yt, yhatt).item()
        loss_test_one.append(one_step)

    X0 = Xtest[0:1, :].to(device)
    mse_multi, sc_pred, _ = strict_eval_multistep(
        model, X0, Utest.to(device), Xtest.to(device))
    loss_test_multi.append(mse_multi)

    score = mse_multi if multi_step == 1 else one_step
    print(f"epoch {epoch:03d} | train={avg_train:.3e} | one-step={one_step:.3e} | multi={mse_multi:.3e} | mean(s^2+c^2)={sc_pred:.4f}")

    if score < best_score:
        best_score = score
        best_epoch = epoch
        torch.save({"epoch": epoch, "state_dict": model.state_dict()},
                   SessionName + "-best.pt")
        print("  -> New best:", best_score)

print("Best epoch:", best_epoch, "best score:", best_score)

# ============================================================
# BLOCK 6 — Strict Validation (PHYSICAL UNITS) + Save A,B,C and stats
# ============================================================

ckpt = torch.load(SessionName + "-best.pt", map_location=device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Extract A,B correctly
with torch.no_grad():
    if Stable_A == 1:
        A_t = model._stable_A().detach().cpu()
    else:
        A_t = model.linA.weight.detach().cpu()
    B_t = model.linB.weight.detach().cpu()

A = A_t.numpy()
B = B_t.numpy()
dimA = A.shape[0]

# C such that x = C z (assuming lift concatenates [x; features])
C = np.zeros((num_state, dimA), dtype=float)
C[:, :num_state] = np.eye(num_state)

savemat("Koopman_" + SessionName + ".mat", {
    "A": A, "B": B, "C": C,
    "X_mean": X_mean, "X_std": X_std,
    "U_mean": U_mean, "U_std": U_std,
})
print("Saved:", "Koopman_" + SessionName + ".mat")
print("A:", A.shape, "B:", B.shape, "C:", C.shape)

# ---- strict open-loop rollout on test set (normalized) ----


@torch.no_grad()
def strict_rollout_full(model, X0, U, T):
    model.eval()
    z = model.lift(X0)

    if Stable_A == 1:
        Aev = model._stable_A().detach()
    else:
        Aev = model.linA.weight.detach()
    Bev = model.linB.weight.detach()

    pred = torch.zeros((T, num_state), device=U.device)
    pred[0:1, :] = X0

    for k in range(T - 1):
        z_next = z @ Aev.T + U[k:k+1, :] @ Bev.T
        x_next = z_next[:, :num_state]
        # project sin/cos
        s = x_next[:, SIN_IDX]
        c = x_next[:, COS_IDX]
        n = torch.sqrt(s*s + c*c + 1e-12)
        x_next = torch.cat([
            x_next[:, 0:2],
            (s/n).unsqueeze(1),
            (c/n).unsqueeze(1),
            x_next[:, 4:7]
        ], dim=1)
        pred[k+1:k+2, :] = x_next
        z = torch.cat([x_next, z_next[:, num_state:]], dim=1)

    return pred


X0 = Xtest[0:1, :].to(device)
Uv = Utest.to(device)
Xtrue = Xtest.to(device)
Tt = Xtrue.shape[0]

pred = strict_rollout_full(model, X0, Uv, Tt)

mse_norm = torch.mean((pred - Xtrue)**2).item()
sc_true = torch.mean(Xtrue[:, SIN_IDX]**2 + Xtrue[:, COS_IDX]**2).item()
sc_pred = torch.mean(pred[:, SIN_IDX]**2 + pred[:, COS_IDX]**2).item()
print("STRICT open-loop MSE (normalized):", mse_norm)
print(
    f"Mean(sin^2+cos^2): true={sc_true:.4f}, pred={sc_pred:.4f} (ideal ~1.0)")

# Denormalize to physical units
pred_np = pred.detach().cpu().numpy()
true_np = Xtrue.detach().cpu().numpy()

pred_phys = pred_np * X_std + X_mean
true_phys = true_np * X_std + X_mean

phi_t,   phi_p = true_phys[:, 0], pred_phys[:, 0]
th_t,    th_p = true_phys[:, 1], pred_phys[:, 1]
sin_t,   sin_p = true_phys[:, 2], pred_phys[:, 2]
cos_t,   cos_p = true_phys[:, 3], pred_phys[:, 3]
p_t,     p_p = true_phys[:, 4], pred_phys[:, 4]
q_t,     q_p = true_phys[:, 5], pred_phys[:, 5]
r_t,     r_p = true_phys[:, 6], pred_phys[:, 6]

psi_t = np.unwrap(np.arctan2(sin_t, cos_t))
psi_p = np.unwrap(np.arctan2(sin_p, cos_p))

t_idx = np.arange(Tt)

series = [
    ("phi [rad]",   phi_t,  phi_p),
    ("theta [rad]", th_t,   th_p),
    ("psi [rad] (unwrap)", psi_t, psi_p),
    ("p [rad/s]",   p_t,    p_p),
    ("q [rad/s]",   q_t,    q_p),
    ("r [rad/s]",   r_t,    r_p),
]

fig, axes = plt.subplots(len(series), 1, sharex=True,
                         figsize=(11, 2.2 * len(series)))
for i, (name, yt, yp) in enumerate(series):
    axes[i].plot(t_idx, yt, label="true", linestyle="solid")
    axes[i].plot(t_idx, yp, label="pred", linestyle="dashed")
    axes[i].set_ylabel(name)
    axes[i].legend(loc="upper right")
axes[0].set_title(
    "STRICT open-loop rollout (physical units) | inputs = desired attitude + throttle + actuators")
axes[-1].set_xlabel("sample index")
fig.tight_layout()
plt.show()

# ============================================================
# BLOCK 7 — Extract Koopman A,B,C (discrete-time) + Save to MAT
# ============================================================


# --- Load best checkpoint ---
ckpt_path = SessionName + "-best.pt"
ckpt = torch.load(ckpt_path, map_location=device)

if "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"])
else:
    raise KeyError("Checkpoint does not contain 'state_dict'.")

model.eval()

# --- Extract A,B from model ---
with torch.no_grad():
    A_t = model.linA.weight.detach().cpu()   # (dimA, dimA)
    B_t = model.linB.weight.detach().cpu()   # (dimA, num_input)

A = A_t.numpy()
B = B_t.numpy()

dimA = A.shape[0]
assert A.shape == (dimA, dimA)
assert B.shape[0] == dimA and B.shape[1] == num_input

# --- Construct C so x = C z ---
# z = [x; lift(x)]  => first num_state elements are x
C = np.zeros((num_state, dimA), dtype=float)
C[:, :num_state] = np.eye(num_state)

# --- Save to MATLAB ---
mat_name = "Koopman_AB_C_" + SessionName + ".mat"
savemat(mat_name, {
    "A": A,
    "B": B,
    "C": C,
    "X_mean": np.asarray(X_mean, dtype=float),
    "X_std":  np.asarray(X_std, dtype=float),
    "U_mean": np.asarray(U_mean, dtype=float),
    "U_std":  np.asarray(U_std, dtype=float),
})

# TODO: Xander -> Convert this to a JSON format like in the matrix_example.json file
print("Saved:", mat_name)
print("A:", A.shape, "B:", B.shape, "C:", C.shape)
print("A = ", A)
print("B = ", B)
print("C = ", C)

data = {
    "model_name": mat_name,
    "dt": dt,
    "A": A.tolist(),
    "B": B.tolist()
}

with open("Koopman/model_results.json", "w") as f:
    json.dump(data, f, indent=2)
print("Saved model results to model_results.json")
