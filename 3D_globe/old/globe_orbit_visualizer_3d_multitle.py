"""
3D Globe Orbit Visualizer (Multi-TLE Single File Support)
- Accepts a single file with many 2-line or 3-line TLEs
- Renders orbital tracks around Earth using Plotly 3D
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
import plotly.graph_objects as go
import argparse
import os

def parse_multi_tle_file(tle_path):
    sats = []
    with open(tle_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    i = 0
    while i < len(lines) - 1:
        if lines[i].startswith("1 ") and lines[i+1].startswith("2 "):
            satnum = lines[i].split()[1]
            sats.append((satnum, Satrec.twoline2rv(lines[i], lines[i+1])))
            i += 2
        elif i < len(lines) - 2 and lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
            satnum = lines[i+1].split()[1]
            sats.append((satnum, Satrec.twoline2rv(lines[i+1], lines[i+2])))
            i += 3
        else:
            i += 1
    return sats

def propagate_eci(sat, start_time, duration_min=1440, step_sec=60):
    times, xs, ys, zs = [], [], [], []
    for i in range(0, duration_min * 60, step_sec):
        t = start_time + timedelta(seconds=i)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        e, r, _ = sat.sgp4(jd, fr)
        if e == 0:
            xs.append(r[0])
            ys.append(r[1])
            zs.append(r[2])
            times.append(t)
    return times, xs, ys, zs

def create_earth_surface():
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 6371 * np.outer(np.cos(u), np.sin(v))
    y = 6371 * np.outer(np.sin(u), np.sin(v))
    z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tle_file", help="Path to TLE file with multiple entries")
    parser.add_argument("--minutes", type=int, default=1440, help="Duration to propagate in minutes")
    args = parser.parse_args()

    start_time = datetime.utcnow()
    fig = go.Figure()

    satellites = parse_multi_tle_file(args.tle_file)
    if not satellites:
        print("❌ No valid TLEs found.")
        return

    for sat_id, sat in satellites:
        times, xs, ys, zs = propagate_eci(sat, start_time, args.minutes)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            name=f"SAT {sat_id}",
            line=dict(width=2),
            hoverinfo='name'
        ))

    xe, ye, ze = create_earth_surface()
    fig.add_trace(go.Surface(x=xe, y=ye, z=ze,
                             colorscale=[[0, 'rgb(10,10,40)'], [1, 'rgb(10,10,40)']],
                             opacity=0.3, showscale=False, hoverinfo='skip'))

    fig.update_layout(
        title="3D Globe Orbit Visualization (Multi-TLE)",
        scene=dict(
            xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)',
            aspectmode='data', bgcolor='black',
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font_color="white",
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()

if __name__ == "__main__":
    main()
