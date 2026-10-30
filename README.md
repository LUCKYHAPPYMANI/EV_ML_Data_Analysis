Electric Vehicle Population Data Analysis
A Python-based data analysis project that explores electric vehicle (EV) population records from the Washington State Open Data portal. The project performs data cleaning, descriptive statistics, covariance and correlation analysis, empirical probability analysis, and data visualization. It also includes an interactive Streamlit dashboard for exploring the dataset.
Project scope: This project is focused on data analysis and visualization. It does not train or claim to validate an EV prediction model.

Project Highlights
- Downloads up to 50,000 records from the public dataset the first time the project is run.
- Stores the downloaded CSV locally so subsequent runs can reuse it.
- Cleans and validates model-year, vehicle, and electric-range fields.
- Calculates mean, median, variance, standard deviation, minimum, and maximum.
- Generates covariance and correlation matrices.
- Calculates empirical probability distributions for EV type and model year.
- Compares recorded electric range across EV types.
- Creates seven Matplotlib/Seaborn visualizations.
- Provides an interactive Streamlit dashboard with filters for EV type, manufacturer, and model year.
Dataset
Washington State Electric Vehicle Population Data
- Official dataset page: https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2
- CSV API endpoint: https://data.wa.gov/resource/f6w7-q2d2.csv
- Provider: Washington State Department of Licensing, published through Washington State Open Data.
The script requests up to 50,000 rows for the analysis. Internet access is required for the first run unless the CSV has already been placed in the data/ directory.
Important electric-range handling
The source metadata explains that an electric-range value of 0 may indicate that the range has not been researched. For this reason, the project converts zero or negative electric-range values to missing values instead of interpreting them as genuine zero-mile ranges. Range statistics and range visualizations use records with known positive electric range; records with unknown range remain available for other analyses such as EV-type and model-year counts.
Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Requests
- Streamlit
Project Structure
EV_ML_Data_Analysis_Project/
├── app.py              # Interactive Streamlit dashboard
├── main.py             # Data download, preprocessing, analysis, and charts
├── requirements.txt    # Python dependencies
├── run.bat             # Windows launcher for the analysis script
├── data/
│   └── electric_vehicle_population.csv  # Downloaded dataset (created on first run)
└── outputs/            # Generated CSV results, interpretation, and charts
The data/ and outputs/ folders may initially be empty. The dataset and output files are created when the relevant scripts run.
Getting Started
1. Clone the repository
git clone <YOUR_REPOSITORY_URL>
cd EV_ML_Data_Analysis_Project
Replace <YOUR_REPOSITORY_URL> with the URL of your GitHub repository.
2. (Recommended) Create a virtual environment
Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
python -m pip install -r requirements.txt
4. Run the data analysis
python main.py
On the first run, the script downloads the dataset and saves it to data/electric_vehicle_population.csv. It then preprocesses the records, performs the statistical and probability analysis, and creates the output files in outputs/.
5. Launch the interactive dashboard
Run the analysis script once before starting the dashboard so that the local dataset is available. Then run:
python -m streamlit run app.py
Streamlit will display a local URL in the terminal, typically:
http://localhost:8501
Open that URL in your browser. The dashboard allows you to filter the records by EV type, manufacturer, and model year, and explore the resulting summaries and charts.

On Windows, you can also use run.bat to run main.py.
Analysis Performed
Data preprocessing
- Checks that the required columns are present.
- Removes duplicate vehicle records using dol_vehicle_id when available.
- Converts model year and electric range to numeric values.
- Marks model years outside the accepted range as missing and removes records without a valid model year.
- Fills missing categorical labels with Unknown.
- Treats zero or negative electric-range values as unavailable, in line with the source metadata.
Statistical analysis
For model year and records with a known positive electric range, the project calculates:
- Count
- Mean
- Median
- Variance
- Standard deviation
- Minimum and maximum
- Covariance matrix
- Correlation matrix
It also calculates electric-range summary statistics by EV type.
Probability analysis
The project calculates empirical probability distributions for:
- EV type
- Model year
It also produces a descriptive summary of the recorded electric-range distribution and visualizes the distribution using a histogram with a kernel density estimate.
Visualizations
The analysis script generates these seven charts:
1. 01_ev_type_distribution.png — number of records by EV type.
2. 02_top_10_manufacturers.png — ten manufacturers with the most records.
3. 03_model_year_distribution.png — records by model year.
4. 04_electric_range_distribution.png — distribution of known electric-range values.
5. 05_range_by_ev_type_boxplot.png — electric-range comparison by EV type.
6. 06_model_year_vs_range.png — model year versus recorded electric range.
7. 07_correlation_heatmap.png — correlation between model year and recorded electric range.
Generated Output Files
The outputs/ directory contains:
File	Description
cleaned_ev_data.csv	Cleaned dataset used in the analysis
statistical_summary.csv	Descriptive statistics
covariance_matrix.csv	Covariance matrix
correlation_matrix.csv	Correlation matrix
range_by_ev_type.csv	Electric-range statistics grouped by EV type
ev_type_probability_distribution.csv	Empirical probability by EV type
model_year_probability_distribution.csv	Empirical probability by model year
electric_range_distribution_summary.csv	Descriptive summary of known electric-range values
interpretation.txt	Automatically generated correlation interpretation and data note
01_...png to 07_...png	Seven generated visualizations


Interpretation Notes and Limitations
- The dataset represents vehicles in Washington State; it should not be interpreted as a complete picture of EV adoption worldwide or across all of the United States.
- The program downloads at most 50,000 records. Results therefore describe the downloaded dataset, not necessarily every record available in the source portal.
- The source dataset can be updated. A new download at a later date may produce different statistics and charts.
- A recorded association between model year and electric range does not establish causation.
- Range analysis excludes records whose electric range is unknown or recorded as zero, based on the source's data note.
- The dashboard reads the locally downloaded CSV. Run main.py first if the dataset file does not exist.
Acknowledgements
Dataset provided by the Washington State Department of Licensing through the Washington State Open Data portal. This project uses the public dataset for educational and analytical purposes.
License
No separate software license is specified in this project. Check the source dataset's terms and add an appropriate license file before redistributing the project.
