# app_with_orbit_style_satid_final.py

🛰️ Satellite RIC Deviation Analyzer
This is a Streamlit-based application designed to analyze and visualize satellite deviations in the Radial-In-Track-Cross-Track (RIC) frame. It provides tools for uploading satellite Two-Line Element (TLE) data, propagating orbits, and generating insightful visualizations and alerts for satellite operations.

Features
TLE/3LE Upload: Upload TLE files for both a master satellite and one or more target satellites for comparison.
Forecast Propagation: Predict satellite positions over a user-defined forecast duration (up to 7 days).
Orbit-Style 4-Panel Visualization: Generate professional orbit-style plots to analyze deviations in RIC coordinates.
Satellite ID Extraction: Automatically extract Satellite IDs from TLE files.
Deviation Alerts: Detect and alert when satellite deviations exceed a customizable threshold.
How It Works
Upload TLE Files:

Upload a TLE file for the master satellite.
Upload one or more TLE files for target satellites to compare against the master.
Set Parameters:

Choose the forecast duration in days (1 to 7).
Set an alert threshold for deviations (in kilometers).
Analyze:

Time-Series Analysis: View charts of RIC deviations over time.
Orbit-Style Visualization: Inspect deviations using a 4-panel orbit-style plot.
Deviation Alerts: Check for deviations exceeding the threshold and get notified.
Visualization Panels
The application generates the following plots:

Cross-Track vs In-Track: Analyze deviations in the cross-track and in-track directions.
Distance to Master vs Time: Monitor distance deviations over the forecast duration.
Radial vs Cross-Track: Inspect radial and cross-track deviations.
Radial vs In-Track: Examine deviations in radial and in-track directions.
Dependencies
Python Libraries:
streamlit
pandas
numpy
matplotlib
sgp4 (for satellite propagation)
Install the dependencies using:

bash
pip install -r requirements.txt
How to Run
Clone this repository:

bash
git clone https://github.com/novellgeek/tric_plot.git
Navigate to the directory:


cd tric_plot
Run the application:

streamlit run app_with_orbit_style_satid_final.py

Open your browser and navigate to the local Streamlit server (usually http://localhost:8501).

File Structure
This application is implemented in app_with_orbit_style_satid_final.py. Key functions include:

parse_tle: Parses TLE files and initializes satellite objects.
compute_ric: Computes RIC deviations between the master and target satellites.
plot_orbit_style: Generates orbit-style visualizations.
Example Usage
Upload TLE files for the master and target satellites.
Set parameters such as forecast duration and deviation threshold.
View visualizations and alerts to understand satellite behavior.
