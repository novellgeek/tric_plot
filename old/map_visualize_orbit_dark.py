"""
Map Visualize Orbit (Dark Theme)
- Loads a TLE or 3LE
- Propagates with SGP4
- Converts ECI to Lat/Lon
- Plots ground track with enhanced visual styling
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
from astropy.time import Time
from astropy.coordinates import TEME, ITRS
from astropy import units as u
import pymap3d as pm
import plotly.express as px
import argparse

def load_tle(file_path):
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    tle_lines = [line for line in lines if line.startswith("1 ") or line.startswith("2 ")]
    if len(tle_lines) < 2:
        raise ValueError("Invalid TLE/3LE file")
    return Satrec.twoline2rv(tle_lines[0], tle_lines[1])

def propagate_latlon(sat, start_time, duration_min=1440, step_sec=60):
    times, lats, lons, alts = [], [], [], []

    for i in range(0, duration_min * 60, step_sec):
        t = start_time + timedelta(seconds=i)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            continue

        teme = TEME(x=r[0]*u.km, y=r[1]*u.km, z=r[2]*u.km, representation_type='cartesian', obstime=Time(t))
        itrs = teme.transform_to(ITRS(obstime=teme.obstime))
        x, y, z = itrs.x.to(u.m).value, itrs.y.to(u.m).value, itrs.z.to(u.m).value
        lat, lon, alt = pm.ecef2geodetic(x, y, z)

        times.append(t)
        lats.append(lat)
        lons.append(lon)
        alts.append(alt / 1000)  # km

    return pd.DataFrame({
        "Time_UTC": times,
        "Latitude": lats,
        "Longitude": lons,
        "Altitude_km": alts
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tle_file", help="Path to TLE or 3LE text file")
    parser.add_argument("--minutes", type=int, default=1440, help="Duration to propagate in minutes")
    args = parser.parse_args()

    sat = load_tle(args.tle_file)
    start = datetime.utcnow()
    df = propagate_latlon(sat, start, args.minutes)

    fig = px.scatter_geo(df, lat="Latitude", lon="Longitude",
                         hover_name="Time_UTC", color="Altitude_km",
                         title="Satellite Ground Track", projection="natural earth")

    fig.update_geos(
        showcoastlines=True, coastlinecolor="white",
        showland=True, landcolor="black",
        showocean=True, oceancolor="midnightblue",
        showlakes=True, lakecolor="blue"
    )

    fig.update_layout(
        paper_bgcolor="black",
        plot_bgcolor="black",
        font_color="white"
    )

    fig.update_traces(marker=dict(size=6, line=dict(width=0), colorscale="Viridis"))

    fig.show()

if __name__ == "__main__":
    main()
