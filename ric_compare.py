"""
RIC Compare Tool
- 3LE/TLE support
- Future projection via --forecast-days
- Orbit-style 4-panel plot generation
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
    tle_lines = [line for line in lines if line.startswith("1 ") or line.startswith("2 ")]
    if len(tle_lines) < 2:
        raise ValueError(f"TLE format error in file: {file_path}")
    return Satrec.twoline2rv(tle_lines[0], tle_lines[1])

def eci_to_ric(r_ref, v_ref, r_other):
    R_hat = r_ref / np.linalg.norm(r_ref)
    C_hat = np.cross(r_ref, v_ref)
    C_hat /= np.linalg.norm(C_hat)
    I_hat = np.cross(C_hat, R_hat)
    rot_matrix = np.vstack((R_hat, I_hat, C_hat)).T
    delta_r = r_other - r_ref
    return rot_matrix.T @ delta_r

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
    csv_path = os.path.join(output_dir, f"{base_name}_RIC.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time_UTC", "Radial_km", "InTrack_km", "CrossTrack_km"])
        for i, t in enumerate(times):
            writer.writerow([t.isoformat(), *ric_array[i]])

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

    rms = np.sqrt(np.mean(ric_array ** 2, axis=0))
    return rms

def generate_orbit_style_plot(output_dir):
    files = glob.glob(os.path.join(output_dir, "*_RIC.csv"))
    if len(files) != 2:
        print("Orbit-style plot requires exactly two *_RIC.csv files.")
        return

    def load_csv(file_path):
        t, r, i, c = [], [], [], []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                t.append(datetime.fromisoformat(row[0]))
                r.append(float(row[1]))
                i.append(float(row[2]))
                c.append(float(row[3]))
        return np.array(t), np.array(r), np.array(i), np.array(c)

    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor('black')

    colors = ['cyan', 'yellow']
    for idx, file in enumerate(files):
        name = os.path.basename(file).replace("_RIC.csv", "")
        t, r, i, c = load_csv(file)
        dist = np.linalg.norm(np.vstack((r, i, c)), axis=0)
        min_idx = np.argmin(dist)

        axs[0, 0].plot(i, c, label=name, color=colors[idx])
        axs[0, 0].scatter(i, c, s=10, color=colors[idx])

        axs[0, 1].plot(t, dist, label=name, color=colors[idx])
        axs[0, 1].scatter(t, dist, s=10, color=colors[idx])
        axs[0, 1].plot(t[min_idx], dist[min_idx], 'ro', markersize=6, label=f"{name} Min Dist")

        axs[1, 0].plot(c, r, label=name, color=colors[idx])
        axs[1, 0].scatter(c, r, s=10, color=colors[idx])

        axs[1, 1].plot(i, r, label=name, color=colors[idx])
        axs[1, 1].scatter(i, r, s=10, color=colors[idx])

    axs[0, 0].set_title("Cross-Track Over In-Track")
    axs[0, 1].set_title("Distance")
    axs[1, 0].set_title("Radial Over Cross-Track")
    axs[1, 1].set_title("Radial Over In-Track")

    axs[0, 0].set_xlabel("In-Track (km)")
    axs[0, 0].set_ylabel("Cross-Track (km)")

    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Distance (km)")

    axs[1, 0].set_xlabel("Cross-Track (km)")
    axs[1, 0].set_ylabel("Radial (km)")

    axs[1, 1].set_xlabel("In-Track (km)")
    axs[1, 1].set_ylabel("Radial (km)")

    for ax in axs.flat:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    output_file = os.path.join(output_dir, "Dark_Overlay_RIC_Orbit_Style.png")
    plt.savefig(output_file, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved orbit-style plot to: {output_file}")

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

    rms_csv = os.path.join(output_dir, "RIC_RMS_Summary.csv")
    with open(rms_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Satellite", "RMS_Radial_km", "RMS_InTrack_km", "RMS_CrossTrack_km"])
        writer.writerows(rms_summary)

    if len(compare_files) == 2:
        generate_orbit_style_plot(output_dir)

    print(f"All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
