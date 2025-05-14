"""
Streamlit RIC Deviation Analyzer with:
- TLE/3LE upload
- Forecast propagation
- Time series + orbit-style plot
- Min distance + threshold alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
import io

st.set_page_config(layout="wide")
st.title("🛰️ Satellite RIC Deviation Analyzer")

def parse_tle(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tle_lines = [line for line in lines if line.startswith("1 ") or line.startswith("2 ")]
    if len(tle_lines) < 2:
        raise ValueError("Invalid TLE/3LE format")
    return Satrec.twoline2rv(tle_lines[0], tle_lines[1])

def eci_to_ric(r_ref, v_ref, r_other):
    R_hat = r_ref / np.linalg.norm(r_ref)
    C_hat = np.cross(r_ref, v_ref)
    C_hat /= np.linalg.norm(C_hat)
    I_hat = np.cross(C_hat, R_hat)
    rot_matrix = np.vstack((R_hat, I_hat, C_hat)).T
    delta_r = r_other - r_ref
    return rot_matrix.T @ delta_r

def compute_ric(master_sat, target_sat, start_time, duration_minutes):
    times, ric_vals = [], []
    for i in range(duration_minutes):
        t = start_time + timedelta(minutes=i)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        e1, r1, v1 = master_sat.sgp4(jd, fr)
        e2, r2, v2 = target_sat.sgp4(jd, fr)
        if e1 == 0 and e2 == 0:
            ric = eci_to_ric(np.array(r1), np.array(v1), np.array(r2))
            ric_vals.append(ric)
            times.append(t)
    return np.array(times), np.array(ric_vals)

def plot_orbit_style(data_dict):
    plt.style.use("dark_background")
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor("black")

    colors = ["cyan", "yellow", "magenta", "lime", "orange"]

    for idx, (name, (times, ric)) in enumerate(data_dict.items()):
        r, i, c = ric[:, 0], ric[:, 1], ric[:, 2]
        dist = np.linalg.norm(ric, axis=1)
        min_idx = np.argmin(dist)

        axs[0, 0].plot(i, c, color=colors[idx], label=name)
        axs[0, 0].scatter(i, c, s=10, color=colors[idx])

        axs[0, 1].plot(times, dist, color=colors[idx], label=name)
        axs[0, 1].scatter(times, dist, s=10, color=colors[idx])
        axs[0, 1].plot(times[min_idx], dist[min_idx], 'ro', markersize=6)

        axs[1, 0].plot(c, r, color=colors[idx], label=name)
        axs[1, 0].scatter(c, r, s=10, color=colors[idx])

        axs[1, 1].plot(i, r, color=colors[idx], label=name)
        axs[1, 1].scatter(i, r, s=10, color=colors[idx])

    axs[0, 0].set_title("Cross-Track vs In-Track")
    axs[0, 1].set_title("Distance to Master vs Time")
    axs[1, 0].set_title("Radial vs Cross-Track")
    axs[1, 1].set_title("Radial vs In-Track")

    for ax in axs.flat:
        ax.set_xlabel(ax.get_xlabel())
        ax.set_ylabel(ax.get_ylabel())
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    return fig

# Sidebar
st.sidebar.header("Upload TLE/3LE Files")
master_file = st.sidebar.file_uploader("Master Satellite TLE", type=["txt"])
target_files = st.sidebar.file_uploader("Compare With (multiple)", type=["txt"], accept_multiple_files=True)
forecast_days = st.sidebar.slider("Forecast Duration (days)", 1, 7, 3)
threshold_km = st.sidebar.slider("Alert Threshold (km)", 1, 10000, 500)

# Process
if master_file and target_files:
    try:
        master_text = master_file.read().decode("utf-8")
        master_sat = parse_tle(master_text)
        duration_minutes = forecast_days * 24 * 60
        start_time = datetime.utcnow()

        data_dict = {}

        for target in target_files:
            target_text = target.read().decode("utf-8")
            name = target_text.splitlines()[0].strip().split()[1] if target_text.strip().startswith("1 ") else target.name.replace(".txt", "")

            target_sat = parse_tle(target_text)
            times, ric = compute_ric(master_sat, target_sat, start_time, duration_minutes)
            data_dict[name] = (times, ric)

        tabs = st.tabs(["📊 Time Series", "🧭 Orbit-Style Plot", "🧠 Alerts"])

        with tabs[0]:
            for name, (times, ric) in data_dict.items():
                df = pd.DataFrame({
                    "UTC Time": times,
                    "Radial (km)": ric[:, 0],
                    "In-Track (km)": ric[:, 1],
                    "Cross-Track (km)": ric[:, 2]
                }).set_index("UTC Time")
                st.subheader(f"RIC Plot: {name}")
                st.line_chart(df)

                rms = np.sqrt(np.mean(ric ** 2, axis=0))
                st.markdown(f"**RMS for {name}:**")
                st.write(f"- Radial: {rms[0]:.2f} km | In-Track: {rms[1]:.2f} km | Cross-Track: {rms[2]:.2f} km")

        with tabs[1]:
            st.pyplot(plot_orbit_style(data_dict))

        with tabs[2]:
            for name, (times, ric) in data_dict.items():
                dist = np.linalg.norm(ric, axis=1)
                min_dist = np.min(dist)
                breach = dist > threshold_km
                st.markdown(f"### {name}")
                st.write(f"📉 Minimum Distance: {min_dist:.2f} km")
                if any(breach):
                    st.error(f"⚠️ {np.sum(breach)} points exceed {threshold_km} km threshold!")
                else:
                    st.success("✅ No deviations exceed threshold.")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a master TLE and at least one comparison TLE.")
