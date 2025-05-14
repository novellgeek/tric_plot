"""
Streamlit Web App for Satellite RIC Analysis
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

# Function to parse TLE or 3LE
def parse_tle(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tle_lines = [line for line in lines if line.startswith("1 ") or line.startswith("2 ")]
    if len(tle_lines) < 2:
        raise ValueError("Invalid TLE/3LE format")
    return Satrec.twoline2rv(tle_lines[0], tle_lines[1])

# Compute RIC from ECI vectors
def eci_to_ric(r_ref, v_ref, r_other):
    R_hat = r_ref / np.linalg.norm(r_ref)
    C_hat = np.cross(r_ref, v_ref)
    C_hat /= np.linalg.norm(C_hat)
    I_hat = np.cross(C_hat, R_hat)
    rot_matrix = np.vstack((R_hat, I_hat, C_hat)).T
    delta_r = r_other - r_ref
    return rot_matrix.T @ delta_r

# Propagate and compute RIC
def compute_ric(master_sat, target_sat, start_time, duration_minutes):
    times = [start_time + timedelta(minutes=i) for i in range(duration_minutes)]
    ric_values = []
    time_labels = []

    for t in times:
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        e1, r1, v1 = master_sat.sgp4(jd, fr)
        e2, r2, v2 = target_sat.sgp4(jd, fr)
        if e1 == 0 and e2 == 0:
            r1, v1, r2 = np.array(r1), np.array(v1), np.array(r2)
            ric = eci_to_ric(r1, v1, r2)
            ric_values.append(ric)
            time_labels.append(t)
    return np.array(time_labels), np.array(ric_values)

# Sidebar: file upload
st.sidebar.header("Upload TLE/3LE Files")
master_file = st.sidebar.file_uploader("Master Satellite TLE", type=["txt"])
target_files = st.sidebar.file_uploader("Compare With (multiple)", type=["txt"], accept_multiple_files=True)
forecast_days = st.sidebar.slider("Forecast Duration (days)", 1, 7, 3)

# Main action
if master_file and target_files:
    try:
        master_text = master_file.read().decode("utf-8")
        master_sat = parse_tle(master_text)

        duration_minutes = forecast_days * 24 * 60
        start_time = datetime.utcnow()

        for target in target_files:
            name = target.name.replace(".txt", "")
            target_text = target.read().decode("utf-8")
            target_sat = parse_tle(target_text)
            times, ric = compute_ric(master_sat, target_sat, start_time, duration_minutes)

            df = pd.DataFrame({
                "UTC Time": times,
                "Radial (km)": ric[:, 0],
                "In-Track (km)": ric[:, 1],
                "Cross-Track (km)": ric[:, 2]
            })

            st.subheader(f"📈 RIC Plot: {name}")
            st.line_chart(df.set_index("UTC Time"))

            rms = np.sqrt(np.mean(ric**2, axis=0))
            st.markdown(f"**RMS Deviation for {name}:**")
            st.write(f"- Radial: {rms[0]:.3f} km")
            st.write(f"- In-Track: {rms[1]:.3f} km")
            st.write(f"- Cross-Track: {rms[2]:.3f} km")

            min_dist = np.min(np.linalg.norm(ric, axis=1))
            st.markdown(f"**🧠 Minimum Distance to Master:** {min_dist:.2f} km")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a master TLE and at least one comparison TLE to begin.")
