ELECTRIC VEHICLE DATA ANALYSIS PROJECT
======================================

This is the runnable Python project. It is not a report.

DATASET
-------
Washington State Electric Vehicle Population Data
Official source:
https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2

The program downloads up to 50,000 records automatically the first time it runs.
Internet access is required for the first run. The CSV is retained in the data
folder, so subsequent runs can use the local copy.

SETUP (WINDOWS / VS CODE)
-------------------------
1. Extract the ZIP file.
2. Open the extracted EV_ML_Data_Analysis_Project folder in VS Code.
3. Open the VS Code terminal in that folder.
4. Install the required libraries:

   py -m pip install -r requirements.txt

5. Run the project:

   py main.py

OUTPUTS
-------
The program creates an outputs folder containing:
- cleaned_ev_data.csv
- statistical_summary.csv
- covariance_matrix.csv
- correlation_matrix.csv
- range_by_ev_type.csv
- probability distribution CSV files
- interpretation.txt
- seven PNG charts

DATA HANDLING NOTE
------------------
The source states that an electric-range value of zero may mean that the range
has not been researched. The project therefore treats zero range as unavailable
(missing) instead of treating it as a real zero-mile range. Electric-range
statistics and plots are based only on records with known positive range.

SOURCE ATTRIBUTION
------------------
Dataset provider: Washington State Department of Licensing.
Dataset portal: Washington State Open Data.
Dataset URL: https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2

The project uses the public dataset for educational analysis. It does not train
or claim to validate a predictive machine-learning model; it performs the
specified data import, preprocessing, statistical analysis, probability
analysis, and visualization.
