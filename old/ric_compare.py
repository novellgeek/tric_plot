# RIC Compare Tool
# Paste full script here or replace this file with your final version.
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
        lines = f.readlines()
    return Satrec.twoline2rv(lines[0].strip(), lines[1].strip())

def eci_to_ric(r_ref, v_ref, r_other):
    R_hat = r_ref / np.linalg.norm(r_ref)
    C_hat = np.cross(r_ref, v_ref)
    C_hat /= np.linalg.norm(C_hat)
    I_hat = np.cross(C_hat, R_hat)
    rot_matrix = np.vstack((R_hat, I_hat, C_hat)).T
    delta_r = r_other - r_ref
    ric = rot_matrix.T @ delta_r
    return ric

def process_against_master(master_sat, compare_file, start_time, duration_minutes, output_dir):
    base_name = os.path.splitext(os.path.basename(compare_file))[0]
    sat = read_tle(compare_file)

    times = [start_time + timedelta(seconds=60 * i) for i in range(duration_minutes)]
    ric_list = []
    valid_times = []

    for t in times:
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        e1, r1, v1 = master_sat.sgp4(jd, fr)
        e2, r2, v2 = sat.sgp4(jd, fr)

        if e1 != 0 or e2 != 0:
            continue

        r1, v1, r2 = np.array(r1), np.array(v1), np.array(r2)
        ric = eci_to_ric(r1, v1, r2)
        ric_list.append(ric)
        valid_times.append(t)

    ric_array = np.array(ric_list)
    time_minutes = np.arange(len(ric_array))

    csv_path = os.path.join(output_dir, f"{base_name}_RIC.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time_UTC", "Radial_km", "InTrack_km", "CrossTrack_km"])
        for i, t in enumerate(valid_times):
            writer.writerow([t.isoformat(), *ric_array[i]])

    plt.figure()
    plt.plot(time_minutes, ric_array[:, 0], label='Radial [km]')
    plt.plot(time_minutes, ric_array[:, 1], label='In-track [km]')
    plt.plot(time_minutes, ric_array[:, 2], label='Cross-track [km]')
    plt.xlabel('Time [minutes]')
    plt.ylabel('Distance [km]')
    plt.title(f'RIC Deviation vs Master: {base_name}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    image_path = os.path.join(output_dir, f"{base_name}_RIC.png")
    plt.savefig(image_path)
    plt.close()

    rms_radial = np.sqrt(np.mean(ric_array[:, 0] ** 2))
    rms_intrack = np.sqrt(np.mean(ric_array[:, 1] ** 2))
    rms_crosstrack = np.sqrt(np.mean(ric_array[:, 2] ** 2))
    return base_name, rms_radial, rms_intrack, rms_crosstrack

def main():
    if len(sys.argv) < 3:
        print("Usage: python ric_compare.py master.txt sat1.txt sat2.txt ...")
        sys.exit(1)

    master_file = sys.argv[1]
    compare_files = sys.argv[2:]
    start_time = datetime(2024, 11, 1, 0, 0, 0)
    duration_minutes = 90
    output_dir = "RIC_Master_Comparison"
    os.makedirs(output_dir, exist_ok=True)

    master_sat = read_tle(master_file)
    rms_summary = []

    for compare_file in compare_files:
        base_name, rms_r, rms_i, rms_c = process_against_master(
            master_sat, compare_file, start_time, duration_minutes, output_dir
        )
        rms_summary.append((base_name, rms_r, rms_i, rms_c))
        print(f"Processed: {compare_file}")

    # Save RMS summary
    rms_csv = os.path.join(output_dir, "RIC_RMS_Summary.csv")
    with open(rms_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Satellite", "RMS_Radial_km", "RMS_InTrack_km", "RMS_CrossTrack_km"])
        writer.writerows(rms_summary)

    # Overlay plot
    plt.figure(figsize=(12, 8))
    for csv_file in glob.glob(os.path.join(output_dir, "*_RIC.csv")):
        base_name = os.path.basename(csv_file).replace("_RIC.csv", "")
        t_vals, r_vals, i_vals, c_vals = [], [], [], []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                t_vals.append(row[0])
                r_vals.append(float(row[1]))
                i_vals.append(float(row[2]))
                c_vals.append(float(row[3]))
        t = np.arange(len(r_vals))
        plt.plot(t, r_vals, label=f'{base_name} - Radial')
        plt.plot(t, i_vals, label=f'{base_name} - In-track')
        plt.plot(t, c_vals, label=f'{base_name} - Cross-track')

    plt.title("Overlayed RIC Deviation vs Master TLE")
    plt.xlabel("Time [minutes]")
    plt.ylabel("Distance [km]")
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    overlay_path = os.path.join(output_dir, "All_RIC_Overlay.png")
    plt.savefig(overlay_path)
    plt.close()
    print(f"Overlay saved to {overlay_path}")
    print(f"RMS summary saved to {rms_csv}")

    # If exactly two satellites are provided, generate direct comparison plots
    if len(compare_files) == 2:
        overlay_two_satellites(compare_files, output_dir)

if __name__ == "__main__":
    main()


def overlay_two_satellites(compare_files, output_dir):
    if len(compare_files) != 2:
        print("Overlay of two satellites requires exactly two comparison TLEs.")
        return

    sat_data = []

    for compare_file in compare_files:
        base_name = os.path.splitext(os.path.basename(compare_file))[0]
        csv_file = os.path.join(output_dir, f"{base_name}_RIC.csv")
        if not os.path.exists(csv_file):
            print(f"Missing RIC CSV for {base_name}. Skipping overlay.")
            return

        r_vals, i_vals, c_vals = [], [], []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                r_vals.append(float(row[1]))
                i_vals.append(float(row[2]))
                c_vals.append(float(row[3]))
        sat_data.append((base_name, r_vals, i_vals, c_vals))

    t = np.arange(len(sat_data[0][1]))

    # Radial comparison
    plt.figure()
    for name, r_vals, _, _ in sat_data:
        plt.plot(t, r_vals, label=f'{name} - Radial')
    plt.title("Radial Comparison (RIC) vs Master")
    plt.xlabel("Time [minutes]")
    plt.ylabel("Radial Deviation [km]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Overlay_Radial_Comparison.png"))
    plt.close()

    # In-track comparison
    plt.figure()
    for name, _, i_vals, _ in sat_data:
        plt.plot(t, i_vals, label=f'{name} - In-track')
    plt.title("In-track Comparison (RIC) vs Master")
    plt.xlabel("Time [minutes]")
    plt.ylabel("In-track Deviation [km]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Overlay_Intrack_Comparison.png"))
    plt.close()

    # Cross-track comparison
    plt.figure()
    for name, _, _, c_vals in sat_data:
        plt.plot(t, c_vals, label=f'{name} - Cross-track')
    plt.title("Cross-track Comparison (RIC) vs Master")
    plt.xlabel("Time [minutes]")
    plt.ylabel("Cross-track Deviation [km]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Overlay_Crosstrack_Comparison.png"))
    plt.close()

    print("Overlay comparison between two satellites saved as:")
    print("- Overlay_Radial_Comparison.png")
    print("- Overlay_Intrack_Comparison.png")
    print("- Overlay_Crosstrack_Comparison.png")



if __name__ == "__main__":
    main()

