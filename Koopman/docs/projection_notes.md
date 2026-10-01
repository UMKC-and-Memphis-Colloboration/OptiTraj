## Using the Koopman Model Directly Inside MPC

The Koopman model is used directly as the prediction model inside the MPC.

The main distinction is that the Koopman model evolves a **lifted state** rather than the raw aircraft state.

The learned discrete-time model is

$$
z_{k+1}=Az_k+Bu_{n,k}
$$

where:

- \(z_k\) is the lifted Koopman state,
- \(A\) is the learned Koopman state-transition matrix,
- \(B\) is the learned Koopman input matrix,
- \(u_{n,k}\) is the normalized control input.

The MPC therefore predicts future behavior using the Koopman model itself.

### 1. Start from the measured aircraft state

The aircraft measurement is

$$
x_{\text{aircraft}}
=
[\phi,\theta,\psi,p,q,r]^T
$$

The Koopman model does not use yaw angle directly. Instead, yaw is represented as

$$
\sin(\psi), \qquad \cos(\psi)
$$

so the physical state used by the learned model is

$$
x=
[
\phi,
\theta,
\sin(\psi),
\cos(\psi),
p,
q,
r
]^T.
$$

### 2. Normalize the physical state

The Koopman model was trained using normalized state data.

Therefore, the same normalization must be applied to the measured aircraft state:

$$
x_n=
\frac{x-X_{\text{mean}}}
{X_{\text{std}}}.
$$

The learned \(A\) and \(B\) matrices are valid in the same normalized representation used during training.

### 3. Lift the state into Koopman space

The normalized physical state is passed through the learned encoder.

For the current encoder architecture,

$$
h_1=
\tanh(W_1x_n)
$$

and

$$
h_2=
\tanh(W_2h_1).
$$

The full Koopman state is

$$
z=
\begin{bmatrix}
x_n\\
h_2
\end{bmatrix}.
$$

The first portion of \(z\) contains the normalized physical aircraft states, while the remaining components are learned nonlinear observables.

The purpose of the lifting operation is to represent nonlinear aircraft behavior using approximately linear dynamics in a higher-dimensional space.

### 4. Use Koopman directly as the MPC model

Once the initial lifted state \(z_0\) has been constructed, the MPC predicts future states using

$$
z_{k+1}=Az_k+Bu_{n,k}.
$$

This is the actual prediction model used throughout the MPC horizon.

No separate nonlinear aircraft model is required.

The Koopman model itself is the MPC model.

### 5. Handle physical control inputs

The learned model was trained using normalized controls:

$$
u_n=
\frac{u-U_{\text{mean}}}
{U_{\text{std}}}.
$$

The MPC can either normalize the controls explicitly at every prediction step or absorb that normalization into the dynamics.

Define

$$
B_{\text{real}}
=
B
\operatorname{diag}
\left(
\frac{1}{U_{\text{std}}}
\right)
$$

and

$$
b=
-B_{\text{real}}U_{\text{mean}}.
$$

Then the dynamics become

$$
z_{k+1}
=
Az_k
+
B_{\text{real}}u_k
+
b.
$$

This form allows the MPC decision variables to remain in physical control units while still using the Koopman model exactly as trained.

### 6. Convert Koopman predictions back to physical states

The optimizer propagates the full lifted state \(z\), but the control objective is normally defined using physical aircraft quantities.

The normalized physical state is extracted using

$$
x_n=Cz.
$$

The physical state is then recovered through denormalization:

$$
x=
x_n\odot X_{\text{std}}
+
X_{\text{mean}}.
$$

Equivalently,

$$
x=
C_{\text{real}}z
+
X_{\text{mean}}
$$

where

$$
C_{\text{real}}
=
\operatorname{diag}(X_{\text{std}})C.
$$

This mapping is only needed when physical aircraft quantities are required for the cost function, constraints, reporting, or actuator interfaces.

### 7. MPC cost is evaluated in physical coordinates

The Koopman state may contain many learned observables that do not have an obvious physical interpretation.

Therefore, the tracking error should normally be evaluated using the reconstructed physical state:

$$
e_k=
x_k-x_{\text{ref}}.
$$

The state-tracking cost can then be written as

$$
J_x=
e_k^TQe_k.
$$

The input penalty is

$$
J_u=
u_k^TRu_k.
$$

The total stage cost is therefore

$$
J_k=
(x_k-x_{\text{ref}})^T
Q
(x_k-x_{\text{ref}})
+
u_k^TRu_k.
$$

The lifted state is used for prediction, while the physical state is used to define what the controller should actually accomplish.

### 8. Do not integrate the Koopman model with RK4

The learned Koopman matrices already define a discrete-time model:

$$
z_{k+1}=Az_k+Bu_k.
$$

They are not a continuous-time model of the form

$$
\dot z=Az+Bu.
$$

Therefore, RK4 should not be applied to the learned Koopman matrices.

The dynamic constraint inside the MPC should instead be imposed directly as

$$
Z_{k+1}
-
\left(
AZ_k+B_{\text{real}}U_k+b
\right)
=
0.
$$

This is why the default OptiTraj RK4 propagation must be overridden for the Koopman implementation.

### 9. Receding-horizon operation

At every MPC update, a new aircraft measurement is used to construct a new lifted initial state.

The process is

$$
x_{\text{measured}}
\rightarrow
x_n
\rightarrow
z_0
\rightarrow
\text{MPC optimization}
\rightarrow
u_0^*.
$$

Only the first optimal control input is applied:

$$
u_{\text{applied}}=u_0^*.
$$

At the next control cycle, the aircraft is measured again and the process repeats.

This means the Koopman state is reinitialized from the real aircraft measurement at every MPC update rather than being allowed to drift indefinitely in open-loop prediction.

### Overall architecture

```text
Measured Aircraft State
(phi, theta, psi, p, q, r)
            |
            v
Construct yaw representation
(phi, theta, sin(psi), cos(psi), p, q, r)
            |
            v
Normalize physical state
            |
            v
Learned Koopman encoder
            |
            v
Lifted state z0
            |
            v
+--------------------------------------+
|               MPC                    |
|                                      |
| z[k+1] = A z[k] + B_real u[k] + b    |
|                                      |
| x[k] = physical-state map from z[k]  |
|                                      |
| tracking cost on physical x[k]       |
|                                      |
| physical input/state constraints     |
+--------------------------------------+
            |
            v
First optimal control
            |
            v
Aircraft
            |
            v
New measurement
            |
            +------ repeat
```

The Koopman model is therefore not being converted into some other MPC model. It is used directly as the MPC prediction model.

The transformations are only required to interface between:

$$
\text{physical aircraft coordinates}
\leftrightarrow
\text{normalized Koopman coordinates}.
$$