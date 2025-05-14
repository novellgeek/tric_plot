"""
Orbital Dashboard v4: Cinematic ECEF Globe
- ECEF coordinates (Earth-fixed)
- Satellite dots only (no trails)
- Gridless globe with NASA-style texture
"""

import numpy as np
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
from astropy.time import Time
from astropy.coordinates import TEME, ITRS
from astropy import units as u
import pymap3d as pm
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

def propagate_ecef(sat, start_time):
    jd, fr = jday(start_time.year, start_time.month, start_time.day,
                  start_time.hour, start_time.minute, start_time.second + start_time.microsecond * 1e-6)
    e, r, _ = sat.sgp4(jd, fr)
    if e != 0:
        return None
    # TEME → ITRF (ECEF)
    teme = TEME(x=r[0]*u.km, y=r[1]*u.km, z=r[2]*u.km,
                representation_type='cartesian', obstime=Time(start_time))
    itrs = teme.transform_to(ITRS(obstime=teme.obstime))
    return itrs.x.to(u.km).value, itrs.y.to(u.km).value, itrs.z.to(u.km).value

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
    parser.add_argument("--html", default="orbit_dashboard_v4_ecef.html", help="Output HTML file")
    args = parser.parse_args()

    sats = parse_multi_tle_file(args.tle_file)
    now = datetime.utcnow()
    xs, ys, zs, labels = [], [], [], []

    for sat_id, sat in sats:
        result = propagate_ecef(sat, now)
        if result:
            x, y, z = result
            xs.append(x)
            ys.append(y)
            zs.append(z)
            labels.append(f"SAT {sat_id}<br>{now.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    xe, ye, ze = create_earth_sphere()
    fig = go.Figure()

    # Earth textured surface
    fig.add_trace(go.Surface(
        x=xe, y=ye, z=ze,
        surfacecolor=np.zeros_like(xe),
        colorscale=[[0, 'rgb(15,15,50)'], [1, 'rgb(15,15,50)']],
        opacity=0.9,
        showscale=False,
        hoverinfo='skip'
    ))

    # Satellite dots
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=5, color='orange'),
        hovertext=labels,
        hoverinfo='text',
        name="Satellites"
    ))

    fig.update_layout(
        title="🛰️ ECEF Orbital Dashboard (v4 Cinematic)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            bgcolor='black'
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.write_html(args.html)
    print(f"✅ Cinematic ECEF dashboard saved to: {args.html}")

if __name__ == "__main__":
    main()
