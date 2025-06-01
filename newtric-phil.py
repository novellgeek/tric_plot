# RIC Analyzer with Manual NORAD ID Entry and File Lookup

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import RendererAgg
import streamlit.components.v1 as components
from io import StringIO

st.set_page_config(layout="wide")
st.title("🛰️ RIC Deviation Analyzer with NORAD Lookup")

TLE_FILE_PATH = "C:/Users/HP/Scripts/my_satellites.txt"

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

def plot_3d_orbits(r1, r2):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=r1[:, 0], y=r1[:, 1], z=r1[:, 2], mode='lines', name='Master'))
    fig.add_trace(go.Scatter3d(x=r2[:, 0], y=r2[:, 1], z=r2[:, 2], mode='lines', name='Target'))
    fig.update_layout(title='3D ECI Orbits', scene=dict(xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)'))
    return fig

def fig_to_png(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Load TLEs into dictionary
try:
    tle_dict = load_tle_dict(TLE_FILE_PATH)
except FileNotFoundError:
    st.error(f"TLE file not found at {TLE_FILE_PATH}")
    st.stop()

# Inputs
master_id = st.sidebar.text_input("Enter Master NORAD ID").strip()
target_id = st.sidebar.text_input("Enter Target NORAD ID").strip()
forecast_days = st.sidebar.slider("Forecast Duration (days)", 1, 7, 3)
threshold_km = st.sidebar.slider("Deviation Alert Threshold (km)", 1, 10000, 500)

if master_id:
    if master_id not in tle_dict:
        st.sidebar.error(f"Master ID {master_id} not found in file.")
if target_id:
    if target_id not in tle_dict:
        st.sidebar.error(f"Target ID {target_id} not found in file.")

if master_id in tle_dict and target_id in tle_dict:
    master_name, m1, m2 = tle_dict[master_id]
    target_name, t1, t2 = tle_dict[target_id]
    master_sat = Satrec.twoline2rv(m1, m2)
    target_sat = Satrec.twoline2rv(t1, t2)

    st.success(f"Master: {master_name} ({master_id}) | Target: {target_name} ({target_id})")

    start_time = datetime.utcnow()
    duration_minutes = forecast_days * 24 * 60
    times, ric, r1, r2 = compute_ric(master_sat, target_sat, start_time, duration_minutes)

    tabs = st.tabs(["📊 RIC Plot", "🧭 Orbit-Style 4-Panel", "🌍 3D Orbit", "📄 CSV Export", "⚠️ Alerts"])

    # --- Tab 0: RIC Plot ---
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

        rms = np.sqrt(np.mean(df_filtered.values ** 2, axis=0))
        st.markdown(f"**RMS (filtered):** R: {rms[0]:.2f} km | I: {rms[1]:.2f} km | C: {rms[2]:.2f} km")

        st.markdown("### 📋 Summary")
        st.write({
            "Start Time": df_filtered.index[0],
            "End Time": df_filtered.index[-1],
            "Duration (min)": end_idx - start_idx + 1,
            "Min Distance (km)": float(np.min(np.linalg.norm(ric[start_idx:end_idx+1], axis=1))),
            "Max Distance (km)": float(np.max(np.linalg.norm(ric[start_idx:end_idx+1], axis=1))),
            "Avg Speed (km/min)": float(np.mean(np.linalg.norm(np.diff(ric[start_idx:end_idx+1], axis=0), axis=1)))
        })

    # --- Tab 1: 4-Panel Plot ---
    with tabs[1]:
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        fig.patch.set_facecolor("white")

        r, i, c = ric[:, 0], ric[:, 1], ric[:, 2]
        dist = np.linalg.norm(ric, axis=1)
        min_idx = np.argmin(dist)

        axs[0, 0].plot(i, c, label="Cross vs In-Track", color='blue')
        axs[0, 0].scatter(i, c, s=5)

        axs[0, 1].plot(times, dist, label="Distance to Master", color='green')
        axs[0, 1].axhline(threshold_km, color='red', linestyle='--', label='Threshold')
        axs[0, 1].scatter(times, dist, s=5, c=np.where(dist > threshold_km, 'red', 'green'))
        axs[0, 1].plot(times[min_idx], dist[min_idx], 'ro', markersize=6)
        axs[0, 1].annotate(f'Min: {dist[min_idx]:.2f} km', xy=(times[min_idx], dist[min_idx]), xytext=(15, 15),
                         textcoords='offset points', arrowprops=dict(arrowstyle='->'), fontsize=8, color='red')

        axs[1, 0].plot(c, r, label="Radial vs Cross", color='orange')
        axs[1, 0].scatter(c, r, s=5, c='orange')

        axs[1, 1].plot(i, r, label="Radial vs In-Track", color='purple')
        axs[1, 1].scatter(i, r, s=5, c='purple')

        axs[0, 0].set_title("Cross-Track vs In-Track")
        axs[0, 1].set_title("Distance to Master vs Time")
        axs[1, 0].set_title("Radial vs Cross-Track")
        axs[1, 1].set_title("Radial vs In-Track")

        for ax in axs.flat:
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        st.pyplot(fig)

        st.download_button("📷 Download All 4-Panel Plot (PNG)", data=fig_to_png(fig), file_name=f"RIC_{master_id}_4panel.png")

    # --- Tab 2: 3D Orbit ---
    with tabs[2]:
        fig3d = plot_3d_orbits(r1[start_idx:end_idx+1], r2[start_idx:end_idx+1])
        st.plotly_chart(fig3d, use_container_width=True)

    # --- Tab 3: CSV Export ---
    with tabs[3]:
        filtered_csv = df_filtered.reset_index().to_csv(index=False)
        st.download_button("📥 Download Filtered RIC CSV", filtered_csv, file_name=f"RIC_filtered_{master_id}_vs_{target_id}.csv")

        full_csv = df.set_index("UTC Time").to_csv()
        st.download_button("📥 Download Full RIC CSV", full_csv, file_name=f"RIC_{master_id}_vs_{target_id}.csv")

    # --- Tab 4: Alerts ---
    with tabs[4]:
        dist = np.linalg.norm(ric, axis=1)
        min_dist = np.min(dist)
        breach = dist > threshold_km
        st.write(f"📉 Minimum Distance: {min_dist:.2f} km")
        if any(breach):
            st.error(f"⚠️ {np.sum(breach)} points exceed {threshold_km} km threshold!")
        else:
            st.success("✅ No deviations exceed threshold.")

else:
    st.info("Enter valid NORAD IDs for master and target to begin RIC analysis.")

# --- Optional: Multi-target comparison ---
if st.sidebar.checkbox("Compare with multiple targets (batch mode)"):
    target_ids = st.sidebar.text_area("Enter comma-separated NORAD IDs for targets").split(',')
    target_ids = [tid.strip() for tid in target_ids if tid.strip() in tle_dict and tid.strip() != master_id]

    if master_id in tle_dict and target_ids:
        master_name, m1, m2 = tle_dict[master_id]
        master_sat = Satrec.twoline2rv(m1, m2)
        start_time = datetime.utcnow()
        duration_minutes = forecast_days * 24 * 60

        report_lines = []
        for tid in target_ids:
            tname, t1, t2 = tle_dict[tid]
            target_sat = Satrec.twoline2rv(t1, t2)
            times, ric, _, _ = compute_ric(master_sat, target_sat, start_time, duration_minutes)
            min_dist = np.min(np.linalg.norm(ric, axis=1))
            breach_count = np.sum(np.linalg.norm(ric, axis=1) > threshold_km)
            report_lines.append(f"Target: {tname} ({tid}) | Min Dist: {min_dist:.2f} km | Breaches: {breach_count}")

        st.markdown("### 📑 Batch Report")
        st.text("\n".join(report_lines))
        st.download_button("📄 Download Report", data="\n".join(report_lines), file_name="RIC_Batch_Report.txt")