#!/usr/bin/env python
# coding: utf-8

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

# Your custom modules (must exist)
from Library import RBFLayer1, ListModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.manual_seed(123)
np.random.seed(123)

# Training flags
Stable_A   = 1    # 1: stable-A parameterization, 0: unconstrained A
multi_step = 1    # selection metric: multi-step vs one-step


# ============================================================
# BLOCK 2 — Load CSV + Build (X,Y,U)
#   x = [phi, theta, sin(psi), cos(psi), p, q, r]
#   u = [phi_des, theta_des, sin(psi_des), cos(psi_des), ThO, C1, C2, C4]
# ============================================================

df = pd.read_csv("SimulatedData_01142026.csv").dropna().reset_index(drop=True)

t = df["IMU_t"].to_numpy(dtype=float)
dt = float(np.median(np.diff(t)))
print("N =", len(df), "dt~", dt)

# ---- Attitude (deg -> rad) ----
phi   = np.deg2rad(df["ATT_Roll"].to_numpy(dtype=float))
theta = np.deg2rad(df["ATT_Pitch"].to_numpy(dtype=float))
psi   = np.deg2rad(df["ATT_Yaw"].to_numpy(dtype=float))

sin_psi = np.sin(psi)
cos_psi = np.cos(psi)

# ---- Body rates ----
p = df["IMU_GyrX"].to_numpy(dtype=float)
q = df["IMU_GyrY"].to_numpy(dtype=float)
r = df["IMU_GyrZ"].to_numpy(dtype=float)

x = np.vstack([phi, theta, sin_psi, cos_psi, p, q, r]).T.astype(np.float32) 

# ---- Desired attitude (deg -> rad) ----
phi_d   = np.deg2rad(df["ATT_DesRoll"].to_numpy(dtype=float))
theta_d = np.deg2rad(df["ATT_DesPitch"].to_numpy(dtype=float))
psi_d   = np.deg2rad(df["ATT_DesYaw"].to_numpy(dtype=float))

sin_psi_d = np.sin(psi_d)
cos_psi_d = np.cos(psi_d)

# ---- Throttle output (leave physical; normalize later) ----
tho = df["CTUN_ThO"].to_numpy(dtype=float)

# ---- Actuator outputs (PWM) -> approx normalized [-1,1] ----
c1 = (df["RCOU_C1"].to_numpy(dtype=float) - 1500.0) / 500.0
c2 = (df["RCOU_C2"].to_numpy(dtype=float) - 1500.0) / 500.0
c4 = (df["RCOU_C4"].to_numpy(dtype=float) - 1500.0) / 500.0

u = np.vstack([phi_d, theta_d, sin_psi_d, cos_psi_d, tho, c1, c2, c4]).T.astype(np.float32)  # (N,8)

# One-step pairs
X = x[:-1, :]
Y = x[1:,  :]
U = u[:-1, :]

print("Raw pairs:", X.shape, Y.shape, U.shape)

# Normalization
X_mean = X.mean(axis=0, keepdims=True)
X_std  = X.std(axis=0, keepdims=True) + 1e-6
U_mean = U.mean(axis=0, keepdims=True)
U_std  = U.std(axis=0, keepdims=True) + 1e-6

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
Xtest,  Ytest,  Utest  = X_t[split:], Y_t[split:], U_t[split:]

num_state = Xtrain.shape[1]  # 7
num_input = Utrain.shape[1]  # 8

print("Train:", Xtrain.shape, Utrain.shape, "| Test:", Xtest.shape, Utest.shape)
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
Test_dataset  = PairDataset(Xtest,  Ytest,  Utest)

num_sample = len(Train_dataset)

learning_rate = 3e-4
batch_size = max(128, int(num_sample/10))

train_loader = DataLoader(Train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(Test_dataset,  batch_size=len(Test_dataset), shuffle=False)

print("batch_size =", batch_size, "learning_rate =", learning_rate)


# ============================================================
# BLOCK 4 — Model Definition (num_state=7, num_input=8)
# ============================================================

# USER SETTING: desired total dimension of z = [x; learned observables].
# Matrix sizes become A:(N,N), B:(N,num_input), C:(num_state,N),
# and D:(num_state,num_input), where N = desired_koopman_dim.
desired_koopman_dim = 24
if desired_koopman_dim <= num_state:
    raise ValueError(f"desired_koopman_dim must be greater than num_state={num_state}.")
lift_width = desired_koopman_dim - num_state 

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
            self.aux_layers.append(nn.Linear(self.shape[j], self.shape[j + 1], bias=False))

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
        self.dimA = params["lift_shape"][-1] + self.numState

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
        P  = M[dim:, dim:]
        Skew = (self.R - self.R.T) / 2.0
        E = (M[:dim, :dim] + P) / 2.0 + Skew
        A = torch.linalg.solve(E, Fm)  
        return A

    def forward(self, x1, x2, u):
        # lifted target
        y = self.lift(x2)
        z1 = self.lift(x1)

        if Stable_A == 1:
            A = self._stable_A()
            z_next = z1 @ A.T + self.linB(u)  
        else:
            z_next = self.linA(z1) + self.linB(u)

        return y, z_next

model = MyArchitechture(params).to(device)
assert model.dimA == desired_koopman_dim
print(model)
print("dimA =", model.dimA)


# ============================================================
# BLOCK 5 — Training (Stable-A SAFE) + Multi-step + yaw circle + yaw phase loss
# ============================================================

loss_fcn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 120
best_score = float("inf")
best_epoch = None

SIN_IDX = 2
COS_IDX = 3

# Loss weights (tuneable)
alpha_1s_lift = 0.15     # weight on 1-step lifted loss
alpha_1s_state = 1.0     # preserve measured-state accuracy
alpha_1s_yaw = 5.0       # make physical yaw visible in every batch
alpha_ms_state = 1.0     # weight on multi-step state loss
lam_circle = 80.0        # enforces sin^2+cos^2 ~ 1
lam_phase = 10.0         # penalize physical circular yaw error
lam_yaw_increment = 20.0 # suppress sample-to-sample yaw chatter
grad_clip = 1.0

def horizon_schedule(epoch):
    if epoch < 30:
        return 10, 96
    if epoch < 75:
        return 25, 64
    return 50, 32

print(f"H=10->25->50, lam_circle={lam_circle}, lam_phase={lam_phase}")

def project_unit_circle(x):
    # A and B operate on normalized states. Convert the yaw pair to physical
    # sine/cosine before projecting, then convert it back to normalized form.
    sin_mean = torch.as_tensor(X_mean[0, SIN_IDX], dtype=x.dtype, device=x.device)
    cos_mean = torch.as_tensor(X_mean[0, COS_IDX], dtype=x.dtype, device=x.device)
    sin_std = torch.as_tensor(X_std[0, SIN_IDX], dtype=x.dtype, device=x.device)
    cos_std = torch.as_tensor(X_std[0, COS_IDX], dtype=x.dtype, device=x.device)
    s = x[..., SIN_IDX] * sin_std + sin_mean
    c = x[..., COS_IDX] * cos_std + cos_mean
    n = torch.sqrt(s*s + c*c + 1e-12)
    s2 = (s / n - sin_mean) / sin_std
    c2 = (c / n - cos_mean) / cos_std
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
    sin_mean = torch.as_tensor(X_mean[0, SIN_IDX], dtype=x_pred.dtype, device=x_pred.device)
    cos_mean = torch.as_tensor(X_mean[0, COS_IDX], dtype=x_pred.dtype, device=x_pred.device)
    sin_std = torch.as_tensor(X_std[0, SIN_IDX], dtype=x_pred.dtype, device=x_pred.device)
    cos_std = torch.as_tensor(X_std[0, COS_IDX], dtype=x_pred.dtype, device=x_pred.device)
    s = x_pred[..., SIN_IDX] * sin_std + sin_mean
    c = x_pred[..., COS_IDX] * cos_std + cos_mean
    return torch.mean((s*s + c*c - 1.0)**2)

def yaw_phase_loss(x_true, x_pred):
    sin_mean = torch.as_tensor(X_mean[0, SIN_IDX], dtype=x_true.dtype, device=x_true.device)
    cos_mean = torch.as_tensor(X_mean[0, COS_IDX], dtype=x_true.dtype, device=x_true.device)
    sin_std = torch.as_tensor(X_std[0, SIN_IDX], dtype=x_true.dtype, device=x_true.device)
    cos_std = torch.as_tensor(X_std[0, COS_IDX], dtype=x_true.dtype, device=x_true.device)
    sin_t = x_true[..., SIN_IDX] * sin_std + sin_mean
    cos_t = x_true[..., COS_IDX] * cos_std + cos_mean
    sin_p = x_pred[..., SIN_IDX] * sin_std + sin_mean
    cos_p = x_pred[..., COS_IDX] * cos_std + cos_mean
    psi_t = torch.atan2(sin_t, cos_t)
    psi_p = torch.atan2(sin_p, cos_p)
    # wrap to [-pi,pi]
    d = (psi_p - psi_t + np.pi) % (2*np.pi) - np.pi
    return torch.mean(d*d)

def yaw_increment_loss(x0, x_true, x_pred):
    true_seq = torch.cat([x0.unsqueeze(1), x_true], dim=1)
    pred_seq = torch.cat([x0.unsqueeze(1), x_pred], dim=1)

    def physical_yaw(x):
        sin_mean = torch.as_tensor(X_mean[0, SIN_IDX], dtype=x.dtype, device=x.device)
        cos_mean = torch.as_tensor(X_mean[0, COS_IDX], dtype=x.dtype, device=x.device)
        sin_std = torch.as_tensor(X_std[0, SIN_IDX], dtype=x.dtype, device=x.device)
        cos_std = torch.as_tensor(X_std[0, COS_IDX], dtype=x.dtype, device=x.device)
        s = x[..., SIN_IDX] * sin_std + sin_mean
        c = x[..., COS_IDX] * cos_std + cos_mean
        return torch.atan2(s, c)

    psi_true = physical_yaw(true_seq)
    psi_pred = physical_yaw(pred_seq)
    d_true = (psi_true[:, 1:] - psi_true[:, :-1] + np.pi) % (2*np.pi) - np.pi
    d_pred = (psi_pred[:, 1:] - psi_pred[:, :-1] + np.pi) % (2*np.pi) - np.pi
    return torch.mean((d_pred - d_true)**2)

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
        x1 = x1.to(device); x2 = x2.to(device); u = u.to(device)

        y, yhat = model(x1, x2, u)
        x_pred_1s = project_unit_circle(yhat[:, :num_state])
        loss_1s_lift = loss_fcn(y, yhat)
        loss_1s_state = weighted_state_mse(x2, x_pred_1s)
        loss_1s_yaw = yaw_phase_loss(x2, x_pred_1s)
        loss_1s = (alpha_1s_lift * loss_1s_lift
                   + alpha_1s_state * loss_1s_state
                   + alpha_1s_yaw * loss_1s_yaw)

        optimizer.zero_grad(set_to_none=True)
        loss_1s.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running += loss_1s.item()

    avg_train = running / max(1, len(train_loader))
    loss_train.append(avg_train)

    # ----------------------------
    # 2) Multi-step state loss (contiguous windows)
    # ----------------------------
    model.train()
    H, K_windows = horizon_schedule(epoch)
    if Xtrain.shape[0] > (H + 2):
        # contiguous windows start indices
        idx0 = torch.randint(low=0, high=Xtrain.shape[0] - (H + 1), size=(K_windows,))

        x0 = Xtrain[idx0, :].to(device)  # (K,7)
        Uwin = torch.stack([Utrain[idx0 + j, :].to(device) for j in range(H)], dim=1)   # (K,H,8)
        Xtrue = torch.stack([Xtrain[idx0 + j + 1, :].to(device) for j in range(H)], dim=1)  # (K,H,7)

        Xpred = multistep_rollout_pred(model, x0, Uwin)

        loss_ms = weighted_state_mse(Xtrue, Xpred)
        loss_c  = unit_circle_penalty(Xpred)
        loss_p  = yaw_phase_loss(Xtrue, Xpred)
        loss_di = yaw_increment_loss(x0, Xtrue, Xpred)

        loss_total = (alpha_ms_state * loss_ms + lam_circle * loss_c
                      + lam_phase * loss_p + lam_yaw_increment * loss_di)

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    # ----------------------------
    # 3) Validation
    # ----------------------------
    model.eval()
    with torch.no_grad():
        x1t, x2t, ut = next(iter(test_loader))
        x1t = x1t.to(device); x2t = x2t.to(device); ut = ut.to(device)
        yt, yhatt = model(x1t, x2t, ut)
        one_step = loss_fcn(yt, yhatt).item()
        loss_test_one.append(one_step)

    X0 = Xtest[0:1, :].to(device)
    mse_multi, sc_pred, pred_val = strict_eval_multistep(model, X0, Utest.to(device), Xtest.to(device))
    yaw_val = yaw_phase_loss(Xtest.to(device), pred_val).item()
    yaw_val_deg = np.rad2deg(np.sqrt(yaw_val))
    loss_test_multi.append(mse_multi)

    score = (mse_multi + 2.0 * yaw_val) if multi_step == 1 else one_step
    print(f"epoch {epoch:03d} | H={H:02d} | train={avg_train:.3e} | one-step={one_step:.3e} | "
          f"multi={mse_multi:.3e} | yaw={yaw_val_deg:.2f} deg | mean(s^2+c^2)={sc_pred:.4f}")

    if score < best_score:
        best_score = score
        best_epoch = epoch
        torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                    "yaw_rmse_deg": float(yaw_val_deg)}, SessionName + "-best.pt")
        print("  -> New best:", best_score)

print("Best epoch:", best_epoch, "best score:", best_score)


# ============================================================
# BLOCK 6 — Strict Validation (PHYSICAL UNITS) + Save A,B,C and stats
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import torch

# Optional LaTeX-style text rendering in matplotlib
plt.rcParams["text.usetex"] = False   # keep False for portability
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

ckpt = torch.load(SessionName + "-best.pt", map_location=device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Extract A, B correctly
with torch.no_grad():
    if Stable_A == 1:
        A_t = model._stable_A().detach().cpu()
    else:
        A_t = model.linA.weight.detach().cpu()
    B_t = model.linB.weight.detach().cpu()

A = A_t.numpy()
B = B_t.numpy()
dimA = A.shape[0]

# C such that x = C z (the lift concatenates [x; features]).
C = np.zeros((num_state, dimA), dtype=float)
C[:, :num_state] = np.eye(num_state)
# There is no direct input-to-output feedthrough in this model.
D = np.zeros((num_state, num_input), dtype=float)

savemat("Koopman_" + SessionName + ".mat", {
    "A": A,
    "B": B,
    "C": C,
    "D": D,
    "X_mean": X_mean,
    "X_std": X_std,
    "U_mean": U_mean,
    "U_std": U_std,
})

print("Saved:", "Koopman_" + SessionName + ".mat")
print("A:", A.shape, "B:", B.shape, "C:", C.shape, "D:", D.shape)

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
        # Projection must be performed in physical sin/cos coordinates.
        x_next = project_unit_circle(z_next[:, :num_state])

        pred[k+1:k+2, :] = x_next
        z = torch.cat([x_next, z_next[:, num_state:]], dim=1)

    return pred

X0 = Xtest[0:1, :].to(device)
Uv = Utest.to(device)
Xtrue = Xtest.to(device)
Tt = Xtrue.shape[0]

pred_strict = strict_rollout_full(model, X0, Uv, Tt)

# Strict recursive rollout is a long-horizon stress test, not a state estimate.
strict_mse_norm = torch.mean((pred_strict - Xtrue) ** 2).item()
sin_mean_t = torch.as_tensor(X_mean[0, SIN_IDX], dtype=Xtrue.dtype, device=device)
cos_mean_t = torch.as_tensor(X_mean[0, COS_IDX], dtype=Xtrue.dtype, device=device)
sin_std_t = torch.as_tensor(X_std[0, SIN_IDX], dtype=Xtrue.dtype, device=device)
cos_std_t = torch.as_tensor(X_std[0, COS_IDX], dtype=Xtrue.dtype, device=device)
true_s = Xtrue[:, SIN_IDX] * sin_std_t + sin_mean_t
true_c = Xtrue[:, COS_IDX] * cos_std_t + cos_mean_t
strict_s = pred_strict[:, SIN_IDX] * sin_std_t + sin_mean_t
strict_c = pred_strict[:, COS_IDX] * cos_std_t + cos_mean_t
sc_true = torch.mean(true_s**2 + true_c**2).item()
sc_pred = torch.mean(strict_s**2 + strict_c**2).item()
strict_yaw_error = torch.atan2(strict_s * true_c - strict_c * true_s,
                                strict_s * true_s + strict_c * true_c)
strict_yaw_rmse_deg = torch.rad2deg(torch.sqrt(torch.mean(strict_yaw_error**2))).item()

print("STRICT recursive rollout MSE (normalized):", strict_mse_norm)
print(f"STRICT recursive yaw RMSE over {Tt} samples: {strict_yaw_rmse_deg:.3f} deg")
print(f"Physical mean(sin^2+cos^2): true={sc_true:.4f}, pred={sc_pred:.4f} (ideal ~1.0)")

# Block 6 plots one-step estimates: every prediction is initialized from the
# measured state at the preceding sample, which is the relevant estimator view.
with torch.no_grad():
    x_current = Xtest.to(device)
    u_current = Utest.to(device)
    z_current = model.lift(x_current)
    if Stable_A == 1:
        A_one = model._stable_A().detach()
    else:
        A_one = model.linA.weight.detach()
    B_one = model.linB.weight.detach()
    lifted_one_step = z_current @ A_one.T + u_current @ B_one.T
    pred_plot = project_unit_circle(lifted_one_step[:, :num_state])
    true_plot = Ytest.to(device)

one_step_mse = torch.mean((pred_plot - true_plot) ** 2).item()
one_step_yaw_rmse_deg = torch.rad2deg(torch.sqrt(yaw_phase_loss(true_plot, pred_plot))).item()
print("ONE-STEP state MSE (normalized):", one_step_mse)
print(f"ONE-STEP yaw RMSE: {one_step_yaw_rmse_deg:.3f} deg")

# ---- Denormalize the plotted one-step estimates to physical units ----
pred_np = pred_plot.detach().cpu().numpy()
true_np = true_plot.detach().cpu().numpy()

pred_phys = pred_np * X_std + X_mean
true_phys = true_np * X_std + X_mean

phi_t, phi_p = true_phys[:, 0], pred_phys[:, 0]
th_t,  th_p  = true_phys[:, 1], pred_phys[:, 1]
sin_t, sin_p = true_phys[:, 2], pred_phys[:, 2]
cos_t, cos_p = true_phys[:, 3], pred_phys[:, 3]
p_t,   p_p   = true_phys[:, 4], pred_phys[:, 4]
q_t,   q_p   = true_phys[:, 5], pred_phys[:, 5]
r_t,   r_p   = true_phys[:, 6], pred_phys[:, 6]

psi_t_wrapped = np.arctan2(sin_t, cos_t)
psi_p_wrapped = np.arctan2(sin_p, cos_p)
# Unwrap truth and prediction independently; no truth-assisted alignment.
psi_t = np.unwrap(psi_t_wrapped)
psi_p = np.unwrap(psi_p_wrapped)

Tplot = true_phys.shape[0]
t_idx = np.arange(Tplot)

# ============================================================
# Figure 1: phi, theta, psi
# ============================================================
fig1, axes1 = plt.subplots(3, 1, sharex=True, figsize=(11, 8))

attitude_series = [
    (r"$\phi$ [rad]",   phi_t, phi_p),
    (r"$\theta$ [rad]", th_t,  th_p),
    (r"$\psi$ [rad]",   psi_t, psi_p),
]

for ax, (label, yt, yp) in zip(axes1, attitude_series):
    ax.plot(t_idx, yt, label="True", linestyle="solid", linewidth=1.6)
    ax.plot(t_idx, yp, label="Predicted", linestyle="dashed", linewidth=1.6)
    ax.set_ylabel(label, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

#axes1[0].set_title(
#    r"Strict Open-Loop Rollout in Physical Units: $\phi$, $\theta$, $\psi$",
#    fontsize=14
#)
axes1[-1].set_xlabel(r"Sample Index", fontsize=12)
fig1.tight_layout()

# ============================================================
# Figure 2: phi_dot, theta_dot, psi_dot
# Here p, q, r are plotted as roll/pitch/yaw rates
# ============================================================
fig2, axes2 = plt.subplots(3, 1, sharex=True, figsize=(11, 8))

rate_series = [
    (r"$\dot{\phi}$ [rad/s]",   p_t, p_p),
    (r"$\dot{\theta}$ [rad/s]", q_t, q_p),
    (r"$\dot{\psi}$ [rad/s]",   r_t, r_p),
]

for ax, (label, yt, yp) in zip(axes2, rate_series):
    ax.plot(t_idx, yt, label="True", linestyle="solid", linewidth=1.6)
    ax.plot(t_idx, yp, label="Predicted", linestyle="dashed", linewidth=1.6)
    ax.set_ylabel(label, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

#axes2[0].set_title(
#    r"Strict Open-Loop Rollout in Physical Units: $\dot{\phi}$, $\dot{\theta}$, $\dot{\psi}$",
#    fontsize=14
#)
axes2[-1].set_xlabel(r"Sample Index", fontsize=12)
fig2.tight_layout()

plt.show()


# ============================================================
# BLOCK 7 — Extract active Koopman A,B,C,D (discrete-time) + Save to MAT
# ============================================================

import numpy as np
import torch
from scipy.io import savemat

# --- Load best checkpoint ---
ckpt_path = SessionName + "-best.pt"
ckpt = torch.load(ckpt_path, map_location=device)

if "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"])
else:
    raise KeyError("Checkpoint does not contain 'state_dict'.")

model.eval()

# --- Extract the matrices actually used by the trained model ---
with torch.no_grad():
    if Stable_A == 1:
        A_t = model._stable_A().detach().cpu()  # active stable A
    else:
        A_t = model.linA.weight.detach().cpu()  # active unconstrained A
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

# No direct input-to-output feedthrough is present in this architecture.
D = np.zeros((num_state, num_input), dtype=float)

assert C.shape == (num_state, dimA)
assert D.shape == (num_state, num_input)

# --- Save to MATLAB ---
# MATLAB/Simulink export.
# The descriptive archive filename prevents results from different model
# dimensions from silently overwriting one another.
mat_name = "Koopman_ABCD.mat"
archive_mat_name = "Koopman_ABCD_" + SessionName + ".mat"
mat_payload = {
    "A": A,
    "B": B,
    "C": C,
    "D": D,
    "sample_time": np.array([[float(dt)]], dtype=float),
    "koopman_dimension": np.array([[dimA]], dtype=np.int32),
    "state_names": np.asarray(
        ["phi", "theta", "sin_psi", "cos_psi", "p", "q", "r"],
        dtype=object,
    ),
    "input_names": np.asarray(
        ["phiD", "thetaD", "sinpsiD", "cospsiD", "ThO", "C1", "C2", "C4"],
        dtype=object,
    ),
    "X_mean": np.asarray(X_mean, dtype=float),
    "X_std":  np.asarray(X_std, dtype=float),
    "U_mean": np.asarray(U_mean, dtype=float),
    "U_std":  np.asarray(U_std, dtype=float),
}
savemat(mat_name, mat_payload, do_compression=True)
savemat(archive_mat_name, mat_payload, do_compression=True)

print("Saved:", mat_name)
print("Archived as:", archive_mat_name)
print("A:", A.shape, "B:", B.shape, "C:", C.shape, "D:", D.shape)
print("A = ", A)
print("B = ", B)
print("C = ", C)
print("D = ", D)
