"""
Orbital Dashboard v3
- Clean display with hover-only labels
- Earth visibility improved
- Always re-reads fresh TLE file
"""

import numpy as np
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

def propagate_eci(sat, start_time, duration_min=1440, step_sec=300):
    times, xs, ys, zs, labels = [], [], [], [], []
    for i in range(0, duration_min * 60, step_sec):
        t = start_time + timedelta(seconds=i)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        e, r, _ = sat.sgp4(jd, fr)
        if e == 0:
            xs.append(r[0])
            ys.append(r[1])
            zs.append(r[2])
            labels.append(t.strftime("%Y-%m-%d %H:%M:%S UTC"))
            times.append(t)
    return times, xs, ys, zs, labels

def create_earth_surface(res=100):
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(0, np.pi, res)
    x = 6371 * np.outer(np.cos(u), np.sin(v))
    y = 6371 * np.outer(np.sin(u), np.sin(v))
    z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tle_file", help="Path to TLE file (multi-sat)")
    parser.add_argument("--minutes", type=int, default=1440, help="Duration to propagate")
    parser.add_argument("--html", default="orbit_dashboard_v3.html", help="Output HTML file")
    args = parser.parse_args()

    start_time = datetime.utcnow()
    satellites = parse_multi_tle_file(args.tle_file)
    fig = go.Figure()

    for sat_id, sat in satellites:
        times, xs, ys, zs, labels = propagate_eci(sat, start_time, args.minutes)

        # Orbit trail
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            name=f"Orbit {sat_id}",
            line=dict(width=2),
            hoverinfo='skip'
        ))

        # Dot at end of arc with hover text only
        if xs and ys and zs:
            fig.add_trace(go.Scatter3d(
                x=[xs[-1]], y=[ys[-1]], z=[zs[-1]],
                mode='markers',
                name=f"SAT {sat_id}",
                marker=dict(size=4, color='orange'),
                hovertext=[f"SAT {sat_id}<br>{labels[-1]}"],
                hoverinfo='text'
            ))

    # Earth
    xe, ye, ze = create_earth_surface()
    fig.add_trace(go.Surface(
        x=xe, y=ye, z=ze,
        surfacecolor=np.zeros_like(xe),
        colorscale=[[0, 'rgb(20,20,50)'], [1, 'rgb(20,20,50)']],
        opacity=0.6,
        showscale=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title="🛰️ Orbital Globe Dashboard v3 (Clean)",
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

    fig.write_html(args.html)
    print(f"✅ Clean orbital dashboard saved as: {args.html}")

if __name__ == "__main__":
    main()
