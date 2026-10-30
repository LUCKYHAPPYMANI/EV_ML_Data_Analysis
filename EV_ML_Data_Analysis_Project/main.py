"""Electric Vehicle Population Data Analysis

Downloads a public EV-registration dataset (if it is not already present),
preprocesses it, performs statistical and probability-distribution analysis,
and saves explanatory visualizations and result files.

Dataset: Washington State Electric Vehicle Population Data
Source: https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2

Run: python main.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests


# -----------------------------------------------------------------------------
# 1. Project settings
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_FILE = DATA_DIR / "electric_vehicle_population.csv"

# Download a substantial sample to keep first-time setup manageable.
# The official dataset contains approximately 300,000 vehicle records.
DATASET_API = "https://data.wa.gov/resource/f6w7-q2d2.csv"
DOWNLOAD_LIMIT = 50_000

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11


# -----------------------------------------------------------------------------
# 2. Download and import the dataset
# -----------------------------------------------------------------------------

def download_dataset() -> None:
    """Download the public CSV only when it is not already available locally."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists() and DATA_FILE.stat().st_size > 0:
        print(f"Using existing dataset: {DATA_FILE}")
        return

    print("Dataset file not found. Downloading EV data from Washington Open Data...")
    params = {
        "$select": (
            "county,city,state,model_year,make,model,ev_type,"
            "cafv_type,electric_range,dol_vehicle_id"
        ),
        "$limit": DOWNLOAD_LIMIT,
    }

    try:
        response = requests.get(DATASET_API, params=params, timeout=120)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "csv" not in content_type and not response.text.lstrip().startswith(
            ("county,", '"county",')
        ):
            raise RuntimeError(
                "The dataset server did not return a CSV file. "
                "Download the CSV manually from the official source and save it as "
                f"{DATA_FILE.name} inside the data folder."
            )

        DATA_FILE.write_bytes(response.content)
        print(f"Downloaded {len(response.content):,} bytes to: {DATA_FILE}")

    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not download the EV dataset. Check your internet connection.\n"
            "Alternatively, download the CSV from:\n"
            "https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2\n"
            f"and place it here: {DATA_FILE}"
        ) from exc


def load_dataset() -> pd.DataFrame:
    """Read the source CSV and validate the required fields."""
    download_dataset()
    df = pd.read_csv(DATA_FILE, low_memory=False)

    required_columns = {
        "model_year", "make", "model", "ev_type", "electric_range"
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "The CSV is missing expected columns: " + ", ".join(sorted(missing_columns))
        )

    return df


# -----------------------------------------------------------------------------
# 3. Data inspection
# -----------------------------------------------------------------------------

def inspect_dataset(df: pd.DataFrame) -> None:
    print("\n" + "=" * 76)
    print("3. DATA INSPECTION")
    print("=" * 76)
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst five records:")
    print(df.head().to_string(index=False))
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values before preprocessing:")
    print(df.isna().sum().to_string())
    print(f"\nExact duplicate rows: {df.duplicated().sum():,}")


# -----------------------------------------------------------------------------
# 4. Data preprocessing
# -----------------------------------------------------------------------------

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean records without pretending that unknown range values are zero."""
    print("\n" + "=" * 76)
    print("4. DATA PREPROCESSING")
    print("=" * 76)

    data = df.copy()
    original_rows = len(data)

    # Remove repeated records. Prefer the official vehicle ID when available.
    if "dol_vehicle_id" in data.columns:
        data = data.drop_duplicates(subset=["dol_vehicle_id"], keep="first")
    else:
        data = data.drop_duplicates()

    # Convert the two numerical analysis fields safely.
    data["model_year"] = pd.to_numeric(data["model_year"], errors="coerce")
    data["electric_range"] = pd.to_numeric(
        data["electric_range"], errors="coerce"
    )

    # Invalid model years are not useful for this analysis.
    data.loc[~data["model_year"].between(1990, 2035), "model_year"] = np.nan

    # Official metadata explains that a zero range can mean it was not researched.
    # Therefore, zero/negative range values are converted to missing, not retained.
    data.loc[data["electric_range"] <= 0, "electric_range"] = np.nan

    # Retain records with a valid model year for adoption/year analyses.
    data = data.dropna(subset=["model_year"])
    data["model_year"] = data["model_year"].astype(int)

    # Fill missing categorical labels so category counts remain explicit.
    for col in ["make", "model", "ev_type", "cafv_type"]:
        if col in data.columns:
            data[col] = data[col].fillna("Unknown").astype(str).str.strip()
            data.loc[data[col] == "", col] = "Unknown"

    # A separate range-valid subset is used for range statistics and graphs.
    # This prevents unknown electric-range values from distorting the results.
    range_data = data.dropna(subset=["electric_range"]).copy()

    print(f"Rows before preprocessing: {original_rows:,}")
    print(f"Rows after duplicate removal and year validation: {len(data):,}")
    print(f"Rows with a known positive electric range: {len(range_data):,}")
    print(
        "Rows with unknown electric range retained for categorical/year analysis: "
        f"{data['electric_range'].isna().sum():,}"
    )
    print("\nMissing values after preprocessing:")
    print(data.isna().sum().to_string())

    return data, range_data


# -----------------------------------------------------------------------------
# 5. Statistical analysis: mean, variance, covariance and correlation
# -----------------------------------------------------------------------------

def perform_statistical_analysis(
    data: pd.DataFrame, range_data: pd.DataFrame
) -> None:
    print("\n" + "=" * 76)
    print("5. STATISTICAL ANALYSIS")
    print("=" * 76)

    numeric_data = range_data[["model_year", "electric_range"]].copy()
    summary = numeric_data.agg(["count", "mean", "median", "var", "std", "min", "max"]).T
    summary.index.name = "Attribute"
    summary.columns = [
        "Count", "Mean", "Median", "Variance", "Std_Deviation", "Minimum", "Maximum"
    ]
    summary.to_csv(OUTPUT_DIR / "statistical_summary.csv")

    print("\nDescriptive statistics (range-valid records):")
    print(summary.round(3).to_string())

    covariance = numeric_data.cov()
    correlation = numeric_data.corr()
    covariance.to_csv(OUTPUT_DIR / "covariance_matrix.csv")
    correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    print("\nCovariance matrix:")
    print(covariance.round(3).to_string())
    print("\nCorrelation matrix:")
    print(correlation.round(3).to_string())

    # Compare the measured electric range across EV categories.
    type_comparison = (
        range_data.groupby("ev_type")["electric_range"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .sort_values("count", ascending=False)
    )
    type_comparison.to_csv(OUTPUT_DIR / "range_by_ev_type.csv")
    print("\nElectric range comparison by EV type (dataset unit: miles):")
    print(type_comparison.round(2).to_string())

    # Generate a cautious interpretation from the computed correlation.
    corr_value = correlation.loc["model_year", "electric_range"]
    abs_corr = abs(corr_value)
    if abs_corr < 0.20:
        strength = "very weak"
    elif abs_corr < 0.40:
        strength = "weak"
    elif abs_corr < 0.60:
        strength = "moderate"
    elif abs_corr < 0.80:
        strength = "strong"
    else:
        strength = "very strong"

    direction = "positive" if corr_value >= 0 else "negative"
    interpretation = (
        f"The correlation between model year and recorded electric range is "
        f"{corr_value:.3f}, indicating a {strength} {direction} linear association "
        f"among records with a known positive range. Correlation does not establish causation."
    )
    (OUTPUT_DIR / "interpretation.txt").write_text(
        interpretation + "\n\n"
        "Important data limitation: zero electric-range values are treated as unknown "
        "because the source states that zero may mean the range has not been researched. "
        "Consequently, range statistics describe only records with a known positive range, "
        "not every registered EV.\n",
        encoding="utf-8",
    )
    print("\nInterpretation:")
    print(interpretation)


# -----------------------------------------------------------------------------
# 6. Probability distribution analysis
# -----------------------------------------------------------------------------

def perform_probability_analysis(data: pd.DataFrame, range_data: pd.DataFrame) -> None:
    print("\n" + "=" * 76)
    print("6. PROBABILITY DISTRIBUTION ANALYSIS")
    print("=" * 76)

    # Empirical categorical probability distribution for EV type.
    type_probability = data["ev_type"].value_counts(normalize=True).rename("Probability")
    type_probability_table = type_probability.reset_index()
    type_probability_table.columns = ["EV_Type", "Probability"]
    type_probability_table.to_csv(
        OUTPUT_DIR / "ev_type_probability_distribution.csv", index=False
    )

    print("\nEmpirical probability distribution of EV type:")
    print(type_probability_table.to_string(index=False, formatters={
        "Probability": "{:.4f}".format
    }))

    # Empirical distribution of model years.
    year_probability = (
        data["model_year"].value_counts(normalize=True)
        .sort_index()
        .rename_axis("Model_Year")
        .reset_index(name="Probability")
    )
    year_probability.to_csv(
        OUTPUT_DIR / "model_year_probability_distribution.csv", index=False
    )

    # Numeric range distribution summary; histogram/KDE are saved in visualizations.
    range_distribution = range_data["electric_range"].describe()
    range_distribution.to_csv(OUTPUT_DIR / "electric_range_distribution_summary.csv")
    print("\nElectric range distribution summary (miles):")
    print(range_distribution.round(2).to_string())


# -----------------------------------------------------------------------------
# 7. Visualization
# -----------------------------------------------------------------------------

def save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()


def create_visualizations(data: pd.DataFrame, range_data: pd.DataFrame) -> None:
    print("\n" + "=" * 76)
    print("7. CREATING VISUALIZATIONS")
    print("=" * 76)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 7.1 EV type frequency distribution
    plt.figure(figsize=(9, 5))
    order = data["ev_type"].value_counts().index
    sns.countplot(data=data, x="ev_type", order=order, color="steelblue")
    plt.title("Number of Registered Vehicles by EV Type")
    plt.xlabel("Electric Vehicle Type")
    plt.ylabel("Number of vehicle records")
    plt.xticks(rotation=10, ha="right")
    save_figure("01_ev_type_distribution.png")

    # 7.2 Top 10 manufacturers
    top_makes = data["make"].value_counts().head(10).sort_values()
    plt.figure(figsize=(9, 6))
    top_makes.plot(kind="barh")
    plt.title("Top 10 EV Manufacturers by Registration Count")
    plt.xlabel("Number of vehicle records")
    plt.ylabel("Manufacturer")
    save_figure("02_top_10_manufacturers.png")

    # 7.3 Model-year distribution
    year_counts = data["model_year"].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=year_counts.index.astype(str), y=year_counts.values, color="steelblue")
    plt.title("Electric Vehicle Records by Model Year")
    plt.xlabel("Model year")
    plt.ylabel("Number of vehicle records")
    plt.xticks(rotation=45, ha="right")
    save_figure("03_model_year_distribution.png")

    # 7.4 Continuous probability distribution of recorded electric range
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=range_data,
        x="electric_range",
        bins=30,
        stat="density",
        kde=True,
        color="steelblue",
    )
    plt.title("Probability Density Distribution of Recorded Electric Range")
    plt.xlabel("Electric range (miles)")
    plt.ylabel("Estimated probability density")
    save_figure("04_electric_range_distribution.png")

    # 7.5 Compare range across EV types
    plot_data = range_data[range_data["ev_type"] != "Unknown"]
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=plot_data, x="ev_type", y="electric_range", color="steelblue")
    plt.title("Electric Range by EV Type")
    plt.xlabel("Electric vehicle type")
    plt.ylabel("Electric range (miles)")
    plt.xticks(rotation=10, ha="right")
    save_figure("05_range_by_ev_type_boxplot.png")

    # 7.6 Model year versus recorded range; sample only for readability.
    scatter_data = range_data
    if len(scatter_data) > 5_000:
        scatter_data = scatter_data.sample(5_000, random_state=42)
    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        data=scatter_data,
        x="model_year",
        y="electric_range",
        alpha=0.45,
        s=25,
        color="steelblue",
    )
    plt.title("Model Year vs Recorded Electric Range")
    plt.xlabel("Model year")
    plt.ylabel("Electric range (miles)")
    save_figure("06_model_year_vs_range.png")

    # 7.7 Correlation heatmap
    correlation = range_data[["model_year", "electric_range"]].corr()
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
    )
    plt.title("Correlation Heatmap of Numerical EV Attributes")
    save_figure("07_correlation_heatmap.png")

    print(f"\nVisualizations saved in: {OUTPUT_DIR}")


# -----------------------------------------------------------------------------
# 8. Save cleaned data and run the complete analysis
# -----------------------------------------------------------------------------

def main() -> int:
    print("\n" + "=" * 76)
    print("ELECTRIC VEHICLE DATA ANALYSIS PROJECT")
    print("=" * 76)
    print("Dataset: Washington State Electric Vehicle Population Data")
    print(f"Download sample limit: {DOWNLOAD_LIMIT:,} records")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        raw_data = load_dataset()
        inspect_dataset(raw_data)

        clean_data, range_data = preprocess_data(raw_data)
        clean_data.to_csv(OUTPUT_DIR / "cleaned_ev_data.csv", index=False)

        if len(range_data) < 2:
            raise ValueError(
                "Fewer than two records have a known positive electric range. "
                "Download a larger or updated dataset before continuing."
            )

        perform_statistical_analysis(clean_data, range_data)
        perform_probability_analysis(clean_data, range_data)
        create_visualizations(clean_data, range_data)

        print("\n" + "=" * 76)
        print("PROJECT EXECUTED SUCCESSFULLY")
        print("=" * 76)
        print(f"Cleaned data and analysis results: {OUTPUT_DIR}")
        print("Open the PNG files in the outputs folder to view the graphs.")
        return 0

    except Exception as exc:
        print(f"\nPROJECT ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
