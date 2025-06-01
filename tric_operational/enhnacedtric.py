# ric_analyzer_enhanced.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from sgp4.api import Satrec, jday
import plotly.graph_objects as go
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import RendererAgg
import pymap3d as pm
import streamlit.components.v1 as components
from io import StringIO

st.set_page_config(layout="wide")
st.title("🛰️ RIC Deviation Analyzer with NORAD Lookup")

TLE_FILE_PATH = "C:/Users/HP/Scripts/my_satellites.txt"

# ------------------- TLE Loader ----------------------
def load_tle_dict(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    tle_dict = {}
    i = 0
    while i < len(lines) - 2:
        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]
        try:
            raw_id = line1.split()[1]
            numeric_id = ''.join(filter(str.isdigit, raw_id))
            tle_dict[numeric_id] = (name, line1, line2)
        except Exception:
            pass
        i += 3
    return tle_dict

# ------------------- RIC Calculation ----------------------
def eci_to_ric(r_ref, v_ref, r_other):
    R_hat = r_ref / np.linalg.norm(r_ref)
    C_hat = np.cross(r_ref, v_ref)
    C_hat /= np.linalg.norm(C_hat)
    I_hat = np.cross(C_hat, R_hat)
    rot_matrix = np.vstack((R_hat, I_hat, C_hat)).T
    delta_r = r_other - r_ref
    return rot_matrix.T @ delta_r

def compute_ric(master_sat, target_sat, start_time, duration_minutes):
    times, ric_vals, eci_r1, eci_r2 = [], [], [], []
    for i in range(duration_minutes):
        t = start_time + timedelta(minutes=i)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        e1, r1, v1 = master_sat.sgp4(jd, fr)
        e2, r2, v2 = target_sat.sgp4(jd, fr)
        if e1 == 0 and e2 == 0:
            ric = eci_to_ric(np.array(r1), np.array(v1), np.array(r2))
            ric_vals.append(ric)
            times.append(t)
            eci_r1.append(r1)
            eci_r2.append(r2)
    return np.array(times), np.array(ric_vals), np.array(eci_r1), np.array(eci_r2)

# ------------------- Plotting ----------------------
def plot_3d_orbits(r1, r2):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=r1[:, 0], y=r1[:, 1], z=r1[:, 2], mode='lines', name='Master'))
    fig.add_trace(go.Scatter3d(x=r2[:, 0], y=r2[:, 1], z=r2[:, 2], mode='lines', name='Target'))
    fig.update_layout(title='3D Orbit Track', scene=dict(xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)'))
    return fig

def fig_to_png(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

def eci_to_ecef_batch(r_eci, times):
    from astropy.time import Time
    from astropy import units as u
    from astropy.coordinates import GCRS, ITRS, CartesianRepresentation

    ecef_coords = []
    for ri, ti in zip(r_eci, times):
        t_astropy = Time(ti, format='datetime', scale='utc')
        gcrs = GCRS(CartesianRepresentation(ri[0] * u.km, ri[1] * u.km, ri[2] * u.km), obstime=t_astropy)
        itrs = gcrs.transform_to(ITRS(obstime=t_astropy))
        ecef = itrs.cartesian.xyz.to(u.km).value
        ecef_coords.append(ecef)
    return np.array(ecef_coords)

# ------------------- Load TLEs ----------------------
try:
    tle_dict = load_tle_dict(TLE_FILE_PATH)
except FileNotFoundError:
    st.error(f"TLE file not found at {TLE_FILE_PATH}")
    st.stop()

# ------------------- Inputs ----------------------
master_id = st.sidebar.text_input("Enter Master NORAD ID").strip()
target_id = st.sidebar.text_input("Enter Target NORAD ID").strip()
forecast_days = st.sidebar.slider("Forecast Duration (days)", 1, 7, 3)
threshold_km = st.sidebar.slider("Deviation Alert Threshold (km)", 1, 10000, 500)

# ------------------- Processing ----------------------
if master_id in tle_dict and target_id in tle_dict:
    master_name, m1, m2 = tle_dict[master_id]
    target_name, t1, t2 = tle_dict[target_id]
    master_sat = Satrec.twoline2rv(m1, m2)
    target_sat = Satrec.twoline2rv(t1, t2)

    st.success(f"Master: {master_name} ({master_id}) | Target: {target_name} ({target_id})")

    start_time = datetime.utcnow()
    duration_minutes = forecast_days * 24 * 60
    times, ric, r1, r2 = compute_ric(master_sat, target_sat, start_time, duration_minutes)

    tabs = st.tabs(["RIC Plot", "4-Panel View", "3D Orbit", "CSV Export", "Alerts", "3D RIC Deviation"])

    # --- Tab 0 ---
    with tabs[0]:
        df = pd.DataFrame({
            "UTC Time": times,
            "Radial (km)": ric[:, 0],
            "In-Track (km)": ric[:, 1],
            "Cross-Track (km)": ric[:, 2]
        })
        start_idx, end_idx = st.slider("Select time range (minutes)", 0, len(df)-1, (0, len(df)-1))
        df_filtered = df.iloc[start_idx:end_idx+1].set_index("UTC Time")
        st.line_chart(df_filtered)

    # --- Tab 1 ---
    with tabs[1]:
        plt.style.use('dark_background')
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        fig.patch.set_facecolor("#1e1e1e")

        r, i, c = ric[:, 0], ric[:, 1], ric[:, 2]
        dist = np.linalg.norm(ric, axis=1)
        min_idx = np.argmin(dist)

        axs[0, 0].plot(i, c, label="Cross vs In-Track", linewidth=0.8, color='cyan')
        axs[0, 0].scatter(i, c, s=4, color='white')

        axs[0, 1].plot(times, dist, linewidth=0.8, label="Distance", color='lime')
        axs[0, 1].axhline(threshold_km, color='red', linestyle='--', label='Threshold')
        axs[0, 1].scatter(times, dist, s=4, c=np.where(dist > threshold_km, 'red', 'lime'))

        axs[1, 0].plot(c, r, linewidth=0.8, color='orange')
        axs[1, 0].scatter(c, r, s=4, color='orange')

        axs[1, 1].plot(i, r, linewidth=0.8, color='magenta')
        axs[1, 1].scatter(i, r, s=4, color='magenta')

        for ax in axs.flat:
            ax.set_facecolor("#2e2e2e")
            ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
            ax.legend(fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)

    # --- Tab 2 ---
    with tabs[2]:
        frame = st.radio("Coordinate Frame", options=["ECI", "ECEF"], horizontal=True)
        if frame == "ECI":
            fig3d = plot_3d_orbits(r1[start_idx:end_idx+1], r2[start_idx:end_idx+1])
        else:
            r1_ecef = eci_to_ecef_batch(r1[start_idx:end_idx+1], times[start_idx:end_idx+1])
            r2_ecef = eci_to_ecef_batch(r2[start_idx:end_idx+1], times[start_idx:end_idx+1])
            fig3d = plot_3d_orbits(r1_ecef, r2_ecef)
        st.plotly_chart(fig3d, use_container_width=True)

    # --- Tab 3 ---
    with tabs[3]:
        st.download_button("Download Filtered CSV", df_filtered.reset_index().to_csv(index=False), file_name="filtered_RIC.csv")

    # --- Tab 4 ---
    with tabs[4]:
        dist = np.linalg.norm(ric, axis=1)
        st.write(f"Min Distance: {np.min(dist):.2f} km")
        if np.any(dist > threshold_km):
            st.error(f"{np.sum(dist > threshold_km)} points exceed {threshold_km} km")
        else:
            st.success("No threshold breaches")

    # --- Tab 5 ---
    with tabs[5]:
        fig_ric_3d = go.Figure()
        fig_ric_3d.add_trace(go.Scatter3d(
            x=ric[start_idx:end_idx+1, 0],
            y=ric[start_idx:end_idx+1, 1],
            z=ric[start_idx:end_idx+1, 2],
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=2),
            name='RIC Vector'
        ))
        fig_ric_3d.update_layout(
            title="3D RIC Deviation",
            scene=dict(xaxis_title="Radial (km)", yaxis_title="In-Track (km)", zaxis_title="Cross-Track (km)"),
            height=700
        )
        st.plotly_chart(fig_ric_3d, use_container_width=True)

else:
    st.info("Enter valid NORAD IDs for master and target to begin analysis.")
