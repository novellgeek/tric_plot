"""
RIC Compare Tool with:
- Clean structure
- TLE and 3LE support
- Future projection capability (forecast days)
"""

import numpy as np
import matplotlib.pyplot as plt
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import os
import csv
import sys
import glob

def read_tle(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    # Support 3LE by keeping only the first 2 valid TLE lines
    tle_lines = [line for line in lines if line.startswith("1 ") or line.startswith("2 ")]
    if len(tle_lines) < 2:
        raise ValueError(f"TLE format error in file: {file_path}")
    return Satrec.twoline2rv(tle_lines[0], tle_lines[1])

def eci_to_ric(r_ref, v_ref, r_other):
    # Reference direction vectors
    r_ref = np.array(r_ref)
    v_ref = np.array(v_ref)
    r_other = np.array(r_other)

    # Radial unit vector
    R_hat = r_ref / np.linalg.norm(r_ref)

    # Cross-track (normal to orbit plane)
    C_hat = np.cross(r_ref, v_ref)
    C_hat /= np.linalg.norm(C_hat)

    # In-track: complete the orthonormal basis
    I_hat = np.cross(C_hat, R_hat)

    # RIC rotation matrix (columns are R, I, C unit vectors)
    rotation_matrix = np.vstack((R_hat, I_hat, C_hat)).T

    # Position difference
    delta_r = r_other - r_ref

    # Project into RIC frame
    ric_vector = rotation_matrix.T @ delta_r
    return ric_vector


def propagate_ric(master_sat, target_sat, start_time, total_minutes):
    times = [start_time + timedelta(minutes=i) for i in range(total_minutes)]
    ric_list = []
    valid_times = []

    for t in times:
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        e1, r1, v1 = master_sat.sgp4(jd, fr)
        e2, r2, v2 = target_sat.sgp4(jd, fr)

        if e1 == 0 and e2 == 0:
            r1, v1, r2 = np.array(r1), np.array(v1), np.array(r2)
            ric = eci_to_ric(r1, v1, r2)
            ric_list.append(ric)
            valid_times.append(t)

    return np.array(ric_list), valid_times

def save_outputs(base_name, ric_array, times, output_dir):
    time_minutes = np.arange(len(ric_array))

    # Save CSV
    csv_path = os.path.join(output_dir, f"{base_name}_RIC.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time_UTC", "Radial_km", "InTrack_km", "CrossTrack_km"])
        for i, t in enumerate(times):
            writer.writerow([t.isoformat(), *ric_array[i]])

    # Save plot
    plt.figure()
    plt.plot(time_minutes, ric_array[:, 0], label='Radial [km]')
    plt.plot(time_minutes, ric_array[:, 1], label='In-track [km]')
    plt.plot(time_minutes, ric_array[:, 2], label='Cross-track [km]')
    plt.xlabel("Time [minutes]")
    plt.ylabel("Distance [km]")
    plt.title(f"RIC vs Master: {base_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_RIC.png"))
    plt.close()

    # Return RMS values
    rms = np.sqrt(np.mean(ric_array ** 2, axis=0))
    return rms

def main():
    if len(sys.argv) < 4:
        print("Usage: python ric_compare.py master.txt sat1.txt [sat2.txt ...] --forecast-days N")
        sys.exit(1)

    forecast_days = 1
    if "--forecast-days" in sys.argv:
        idx = sys.argv.index("--forecast-days")
        forecast_days = int(sys.argv[idx + 1])
        input_files = sys.argv[1:idx]
    else:
        input_files = sys.argv[1:]

    master_file = input_files[0]
    compare_files = input_files[1:]

    output_dir = "RIC_Output"
    os.makedirs(output_dir, exist_ok=True)

    start_time = datetime.utcnow()
    total_minutes = forecast_days * 24 * 60

    master_sat = read_tle(master_file)

    rms_summary = []
    for f in compare_files:
        base_name = os.path.splitext(os.path.basename(f))[0]
        target_sat = read_tle(f)
        ric_array, times = propagate_ric(master_sat, target_sat, start_time, total_minutes)
        rms = save_outputs(base_name, ric_array, times, output_dir)
        rms_summary.append((base_name, *rms))
        print(f"Processed {base_name}")

    # Save RMS summary
    rms_path = os.path.join(output_dir, "RIC_RMS_Summary.csv")
    with open(rms_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Satellite", "RMS_Radial_km", "RMS_InTrack_km", "RMS_CrossTrack_km"])
        writer.writerows(rms_summary)

    print(f"All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
