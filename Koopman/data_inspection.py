#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD DATA
# ============================================================

csv_path = "Koopman/csv_files/calibration_data.csv"
df = pd.read_csv(csv_path)

t = df["t"]


# ============================================================
# 1. IMU GYROS + DERIVATIVES (SUBPLOTS)
# ============================================================

fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle("IMU Gyros + Angular Acceleration")

axes = ["X", "Y", "Z"]

for i, axis in enumerate(axes):
    axs[i].plot(t, df[f"IMU_Gyr{axis}"], label=f"Gyr{axis}")
    axs[i].plot(t, df[f"DIFF_Gyr{axis}"], linestyle="--", label=f"dGyr{axis}")
    axs[i].set_ylabel(f"{axis}-axis")
    axs[i].grid()
    axs[i].legend()

axs[-1].set_xlabel("Time [s]")


# ============================================================
# 2. ATTITUDE TRACKING (SUBPLOTS)
# ============================================================

fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle("Attitude Tracking (Actual vs Desired)")

axes = ["Roll", "Pitch", "Yaw"]

for i, axis in enumerate(axes):
    axs[i].plot(t, df[f"ATT_{axis}"], label=f"{axis}")
    axs[i].plot(t, df[f"ATT_Des{axis}"], linestyle="--", label=f"Des{axis}")
    axs[i].set_ylabel(f"{axis} [deg]")
    axs[i].grid()
    axs[i].legend()

axs[-1].set_xlabel("Time [s]")


# ============================================================
# 3. WIND / AOA SIGNALS
# ============================================================

fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
fig.suptitle("Wind / Angle of Attack Metrics")

if "AOA_AOA" in df.columns:
    axs[0].plot(t, df["AOA_AOA"], label="AOA")
    axs[0].set_ylabel("AOA [deg]")
    axs[0].legend()
    axs[0].grid()

if "AOA_SSA" in df.columns:
    axs[1].plot(t, df["AOA_SSA"], label="SSA")
    axs[1].set_ylabel("SSA [deg]")
    axs[1].legend()
    axs[1].grid()

axs[-1].set_xlabel("Time [s]")


# ============================================================
# 4. ALTITUDE + AIRSPEED
# ============================================================

fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
fig.suptitle("Altitude and Airspeed")

axs[0].plot(t, df["GPS_Alt"], label="Altitude")
axs[0].set_ylabel("Altitude")
axs[0].grid()
axs[0].legend()

axs[1].plot(t, df["GPS_Spd"], label="Airspeed")
axs[1].set_ylabel("Speed")
axs[1].grid()
axs[1].legend()

axs[-1].set_xlabel("Time [s]")


# ============================================================
# 5. RC COMMANDS (SUBPLOTS)
# ============================================================

fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
fig.suptitle("RC Commands")

channels = ["C1", "C2", "C3", "C4"]

for i, ch in enumerate(channels):
    col = f"RCOU_{ch}"
    if col in df.columns:
        axs[i].plot(t, df[col], label=ch)
        axs[i].set_ylabel(ch)
        axs[i].grid()
        axs[i].legend()

axs[-1].set_xlabel("Time [s]")


# ============================================================
# SHOW ALL FIGURES
# ============================================================

plt.tight_layout()
plt.show()