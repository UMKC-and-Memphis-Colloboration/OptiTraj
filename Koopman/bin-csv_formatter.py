#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from pymavlink import DFReader
from scipy.signal import butter, filtfilt, savgol_filter

class FlightParser:
    def __init__(self, bin_path: Path) -> None:
        self.bin_path = bin_path
        self.binary_log = DFReader.DFReader_binary(filename=str(bin_path))

        print("Message types present:",
              sorted(fmt.name for fmt in self.binary_log.formats.values()))

        self.binary_log.rewind()

    def extract(self, config: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
        rows = {msg_type: [] for msg_type in config.keys()}

        while True:
            m = self.binary_log.recv_msg()
            if m is None:
                break

            msg_type = m.get_type()
            if msg_type not in config:
                continue

            d = m.to_dict()
            if hasattr(m, "TimeUS"):
                d["TimeUS"] = m.TimeUS
            else:
                continue

            rows[msg_type].append(d)

        dfs = {}
        for msg_type, cols in config.items():
            df = pd.DataFrame(rows[msg_type])
            if df.empty:
                dfs[msg_type] = df
                continue

            keep_cols = [c for c in cols if c in df.columns]
            df = df[keep_cols].copy()

            dfs[msg_type] = df

        return dfs

def to_time_seconds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "TimeUS" not in df.columns:
        return df

    df = df.sort_values("TimeUS").copy()
    t0 = df["TimeUS"].iloc[0]
    df["t"] = (df["TimeUS"] - t0) * 1e-6
    return df

def create_time_grid(dfs: Dict[str, pd.DataFrame], hz: float = 50.0) -> np.ndarray:
    all_times = []

    for df in dfs.values():
        if df.empty or "t" not in df.columns:
            continue
        all_times.append(df["t"].values)

    if not all_times:
        return np.array([])

    t_min = max(t[0] for t in all_times)
    t_max = min(t[-1] for t in all_times)

    dt = 1.0 / hz
    return np.arange(t_min, t_max, dt)

def interpolate_to_grid(df: pd.DataFrame, time_grid: np.ndarray, prefix: str) -> pd.DataFrame:
    if df.empty or "t" not in df.columns:
        return pd.DataFrame()

    df = df.sort_values("t")
    df = df.groupby("t", as_index=False).mean()
    df = df.set_index("t")

    # Drop TimeUS after conversion
    value_cols = [c for c in df.columns if c != "TimeUS"]

    interp = pd.DataFrame(index=time_grid)

    for col in value_cols:
        interp[f"{prefix}_{col}"] = df[col].reindex(
            df.index.union(time_grid)
        ).interpolate(method="index").reindex(time_grid)

    return interp

def build_flight_matrix(dfs: Dict[str, pd.DataFrame], hz: float = 50.0) -> pd.DataFrame:
    # Convert all to time-based format
    for k in dfs:
        dfs[k] = to_time_seconds(dfs[k])

    # Build shared time grid
    time_grid = create_time_grid(dfs, hz=hz)
    if len(time_grid) == 0:
        return pd.DataFrame()

    # Interpolate each message type
    aligned = [pd.DataFrame(index=time_grid)]

    for msg_type, df in dfs.items():
        interp = interpolate_to_grid(df, time_grid, msg_type)
        if not interp.empty:
            aligned.append(interp)

    # Merge all
    return pd.concat(aligned, axis=1)

def butter_lowpass_filter(data, cutoff_hz, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def add_gyro_derivatives(df: pd.DataFrame, hz: float = 50.0) -> pd.DataFrame:
    # cutoff_hz = 1.54
    cutoff_hz = 5.04

    window_length = 5   # required
    polyorder = 3

    for axis in ["GyrX", "GyrY", "GyrZ"]:
        col = f"IMU_{axis}"
        out_col = f"DIFF_{axis}"

        if col not in df.columns:
            continue

        signal = df[col].values

        filtered = butter_lowpass_filter(
            signal,
            cutoff_hz=cutoff_hz,
            fs=hz,
            order=2
        )

        deriv = savgol_filter(
            filtered,
            window_length=window_length,
            polyorder=polyorder,
            deriv=1,
            delta=1.0 / hz
        )

        filtered_deriv = butter_lowpass_filter(
            deriv,
            cutoff_hz=cutoff_hz,
            fs=hz,
            order=2
        )

        df[out_col] = filtered_deriv

    return df

def parse_bin(
    binaries_folder: str,
    bin_name: str,
    data_config: Dict[str, List[str]],
    output_name: str = "calibration_data.csv",
) -> None:

    bin_directory = Path(binaries_folder)
    bin_path = bin_directory / f"{bin_name}.BIN"

    if not bin_path.exists():
        raise FileNotFoundError(bin_path)

    parser = FlightParser(bin_path)
    dfs = parser.extract(data_config)

    matrix = build_flight_matrix(dfs, hz=50.0)
    
    matrix = add_gyro_derivatives(matrix, hz=50.0)

    if matrix.empty:
        raise RuntimeError("Failed to build flight matrix.")

    output_file = bin_directory.parent / "csv_files" / output_name
    matrix.to_csv(output_file, index=True, index_label="t")

    print(f"Saved calibrated flight matrix → {output_file}")

if __name__ == "__main__":
    binaries_folder = "Koopman/binaries"
    bin_name = "00000091"

    data_config = {
        "IMU": ["TimeUS", "AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"],
        "RCOU": ["TimeUS", "C1", "C2", "C3", "C4"],
        "ATT": ["TimeUS", "Roll", "Pitch", "Yaw", "DesRoll", "DesPitch", "DesYaw"],
        "CTUN": ["TimeUS", "ThO"],
        "GPS": ["TimeUS", "Lat", "Lng", "Alt", "Spd"],
        "AOA": ["TimeUS", "AOA", "SSA"],
    }

    parse_bin(
        binaries_folder=binaries_folder,
        bin_name=bin_name,
        data_config=data_config,
        output_name="calibration_data.csv",
    )