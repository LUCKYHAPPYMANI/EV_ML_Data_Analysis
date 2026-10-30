"""Interactive EV Data Analysis Dashboard.

Run with:
    streamlit run app.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "electric_vehicle_population.csv"

st.set_page_config(page_title="EV Data Analysis", page_icon="⚡", layout="wide")
sns.set_theme(style="whitegrid")

@st.cache_data

def load_data():
    if not DATA_FILE.exists():
        st.error("Dataset not found. Run `python main.py` once to download it.")
        st.stop()
    df = pd.read_csv(DATA_FILE, low_memory=False)
    for col in ["model_year", "electric_range"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset=["dol_vehicle_id"] if "dol_vehicle_id" in df else None)
    df.loc[~df["model_year"].between(1990, 2035), "model_year"] = np.nan
    df.loc[df["electric_range"] <= 0, "electric_range"] = np.nan
    df = df.dropna(subset=["model_year"]).copy()
    df["model_year"] = df["model_year"].astype(int)
    for col in ["make", "model", "ev_type", "cafv_type"]:
        if col in df:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            df.loc[df[col] == "", col] = "Unknown"
    return df

st.title("⚡ Electric Vehicle Data Analysis Dashboard")
st.caption("Interactive statistical, probability and visualization analysis of EV population data")

df = load_data()
range_df = df.dropna(subset=["electric_range"]).copy()

with st.sidebar:
    st.header("Filters")
    types = st.multiselect("EV Type", sorted(df.ev_type.unique()), default=sorted(df.ev_type.unique()))
    makes = st.multiselect("Manufacturer", sorted(df.make.unique()), default=[])
    year_min, year_max = int(df.model_year.min()), int(df.model_year.max())
    years = st.slider("Model Year", year_min, year_max, (year_min, year_max))

filtered = df[df.ev_type.isin(types) & df.model_year.between(years[0], years[1])].copy()
if makes:
    filtered = filtered[filtered.make.isin(makes)]
filtered_range = filtered.dropna(subset=["electric_range"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Vehicle Records", f"{len(filtered):,}")
c2.metric("Manufacturers", f"{filtered.make.nunique():,}")
c3.metric("Known Range Records", f"{len(filtered_range):,}")
c4.metric("Average Range", f"{filtered_range.electric_range.mean():.1f} mi" if len(filtered_range) else "N/A")

st.divider()

# Distribution and categorical visualization
col1, col2 = st.columns(2)
with col1:
    st.subheader("EV Type Distribution")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts = filtered.ev_type.value_counts()
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_xlabel("Number of vehicle records")
    ax.set_ylabel("EV type")
    st.pyplot(fig, clear_figure=True)
with col2:
    st.subheader("Top 10 Manufacturers")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts = filtered.make.value_counts().head(10).sort_values()
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_xlabel("Number of vehicle records")
    ax.set_ylabel("Manufacturer")
    st.pyplot(fig, clear_figure=True)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Electric Range Probability Distribution")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(filtered_range.electric_range, bins=30, stat="density", kde=True, ax=ax)
    ax.set_xlabel("Electric range (miles)")
    ax.set_ylabel("Estimated probability density")
    st.pyplot(fig, clear_figure=True)
with col4:
    st.subheader("Range by EV Type")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=filtered_range, x="ev_type", y="electric_range", ax=ax)
    ax.set_xlabel("EV type")
    ax.set_ylabel("Electric range (miles)")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig, clear_figure=True)

st.subheader("Model Year vs Electric Range")
fig, ax = plt.subplots(figsize=(12, 5))
plot_df = filtered_range if len(filtered_range) <= 7000 else filtered_range.sample(7000, random_state=42)
sns.scatterplot(data=plot_df, x="model_year", y="electric_range", alpha=.45, s=25, ax=ax)
ax.set_xlabel("Model year")
ax.set_ylabel("Electric range (miles)")
st.pyplot(fig, clear_figure=True)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(6, 4))
corr = filtered_range[["model_year", "electric_range"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
st.pyplot(fig, clear_figure=True)

st.divider()
st.header("Statistical Analysis")
summary = filtered_range[["model_year", "electric_range"]].agg(["count", "mean", "median", "var", "std", "min", "max"]).T
st.dataframe(summary.round(3), use_container_width=True)

st.subheader("Covariance Matrix")
st.dataframe(filtered_range[["model_year", "electric_range"]].cov().round(3), use_container_width=True)

st.subheader("Empirical Probability Distribution")
prob = filtered.ev_type.value_counts(normalize=True).rename("Probability").reset_index()
prob.columns = ["EV Type", "Probability"]
prob["Probability"] = prob["Probability"].round(4)
st.dataframe(prob, use_container_width=True)

st.info("Interpretation: electric-range statistics use only records with a known positive range. In the source data, a zero range can mean that the range was not researched, so zero values are treated as unavailable rather than genuine zero-mile ranges.")
