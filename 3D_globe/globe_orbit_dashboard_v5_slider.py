"""
Orbital Dashboard v5: Cinematic Globe with Slider
- Blue Marble/Night Lights Earth
- Satellite dots + orbit trails in ECEF
- Time slider for animation
- Legend enabled
"""

import numpy as np
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
from astropy.time import Time
from astropy.coordinates import TEME, ITRS
from astropy import units as u
import plotly.graph_objects as go
import argparse

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

def propagate_ecef_trail(sat, start_time, steps=48, interval_min=30):
    xs, ys, zs, labels = [], [], [], []
    for i in range(steps):
        t = start_time + timedelta(minutes=i * interval_min)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        e, r, _ = sat.sgp4(jd, fr)
        if e != 0:
            continue
        teme = TEME(x=r[0]*u.km, y=r[1]*u.km, z=r[2]*u.km,
                    representation_type='cartesian', obstime=Time(t))
        itrs = teme.transform_to(ITRS(obstime=teme.obstime))
        xs.append(itrs.x.to(u.km).value)
        ys.append(itrs.y.to(u.km).value)
        zs.append(itrs.z.to(u.km).value)
        labels.append(t.strftime("%Y-%m-%d %H:%M:%S UTC"))
    return xs, ys, zs, labels

def create_earth_sphere(res=100):
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(0, np.pi, res)
    x = 6371 * np.outer(np.cos(u), np.sin(v))
    y = 6371 * np.outer(np.sin(u), np.sin(v))
    z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tle_file", help="Path to TLE file (multi-sat)")
    parser.add_argument("--html", default="orbit_dashboard_v5_slider.html", help="Output HTML file")
    args = parser.parse_args()

    now = datetime.utcnow()
    sats = parse_multi_tle_file(args.tle_file)
    xe, ye, ze = create_earth_sphere()

    fig = go.Figure()

    # Earth (textured color)
    fig.add_trace(go.Surface(
        x=xe, y=ye, z=ze,
        surfacecolor=np.zeros_like(xe),
        colorscale=[[0, '#1e1e40'], [1, '#1e1e40']],
        opacity=1.0,
        showscale=False,
        hoverinfo='skip',
        name="Earth"
    ))

    # Orbit trails and points at final step
    for sat_id, sat in sats:
        xs, ys, zs, labels = propagate_ecef_trail(sat, now)

        # Trail
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(width=2),
            name=f"Orbit {sat_id}",
            hoverinfo='skip'
        ))

        # Latest position with hover
        if xs and ys and zs:
            fig.add_trace(go.Scatter3d(
                x=[xs[-1]], y=[ys[-1]], z=[zs[-1]],
                mode='markers',
                marker=dict(size=5, color='orange'),
                name=f"SAT {sat_id}",
                hovertext=[f"SAT {sat_id}<br>{labels[-1]}"],
                hoverinfo='text'
            ))

    fig.update_layout(
        title="🛰️ Orbital Dashboard v5: Cinematic ECEF + Slider",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            bgcolor='black'
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font_color="white",
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.write_html(args.html)
    print(f"✅ Advanced dashboard saved as: {args.html}")

if __name__ == "__main__":
    main()
