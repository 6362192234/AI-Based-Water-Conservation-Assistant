"""
eda.py

Comprehensive Exploratory Data Analysis (EDA) Pipeline for HSB Living Lab Water Consumption Dataset.
Generates publication-grade diagnostic plots and statistical summaries for ML modeling.
"""

from pathlib import Path
import argparse
import logging
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Configure seaborn & matplotlib styles
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# Palette definitions
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
PALETTE = sns.color_palette("mako")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_preprocess(input_path: Path) -> pd.DataFrame:
    """Load raw dataset with proper types."""
    logger.info("Loading dataset from: %s", input_path)
    df = pd.read_csv(input_path, sep=";")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def plot_target_distributions(df: pd.DataFrame, output_dir: Path):
    """Plot distribution of value and aggregated_value."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Target Numerical Distributions: Consumption Value & Aggregated Meter", fontsize=15, fontweight="bold", y=0.98)

    # 1A: Value linear histogram
    ax = axes[0, 0]
    sns.histplot(df["value"], bins=50, kde=True, color="#1f77b4", ax=ax)
    ax.set_title("Consumption Value (m³) - Linear Scale (Heavy Right Skew)")
    ax.set_xlabel("Value (m³ per 10-min event)")
    ax.set_ylabel("Frequency Count")
    p99 = df["value"].quantile(0.99)
    ax.axvline(p99, color="red", linestyle="--", label=f"99th percentile: {p99:.3f} m³")
    ax.axvline(df["value"].median(), color="green", linestyle="-", label=f"Median: {df['value'].median():.3f} m³")
    ax.legend(frameon=True)

    # 1B: Value log10 scale
    ax = axes[0, 1]
    log_vals = np.log10(df["value"])
    sns.histplot(log_vals, bins=50, kde=True, color="#2ca02c", ax=ax)
    ax.set_title("Log10 Consumption Value (log10(m³))")
    ax.set_xlabel("log10(Value in m³)")
    ax.set_ylabel("Frequency Count")
    ax.axvline(np.log10(0.001), color="purple", linestyle=":", label="-3.0 (1 Liter)")
    ax.axvline(np.log10(0.010), color="orange", linestyle=":", label="-2.0 (10 Liters)")
    ax.axvline(np.log10(0.100), color="red", linestyle=":", label="-1.0 (100 Liters)")
    ax.legend(frameon=True)

    # 1C: Boxplot of Value by Fixture (Log scale)
    ax = axes[1, 0]
    sns.boxplot(data=df, x="attached_to", y="value", palette="Set2", ax=ax, showfliers=False)
    ax.set_yscale("log")
    ax.set_title("Volume Distribution by Fixture (Log Scale, Fliers Hidden)")
    ax.set_xlabel("Fixture (attached_to)")
    ax.set_ylabel("Value (m³, log scale)")
    ax.tick_params(axis="x", rotation=15)

    # 1D: Aggregated Value Histogram + ECDF
    ax = axes[1, 1]
    sns.histplot(df["aggregated_value"], bins=50, kde=True, color="#9467bd", ax=ax)
    ax.set_title("Aggregated Cumulative Meter Readings (m³)")
    ax.set_xlabel("Aggregated Value (m³)")
    ax.set_ylabel("Frequency Count")
    ax.axvline(df["aggregated_value"].mean(), color="red", linestyle="--", label=f"Mean: {df['aggregated_value'].mean():.1f} m³")
    ax.axvline(df["aggregated_value"].median(), color="green", linestyle="-", label=f"Median: {df['aggregated_value'].median():.1f} m³")
    ax.legend(frameon=True)

    plt.tight_layout()
    save_path = output_dir / "01_target_distributions.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved %s", save_path)


def plot_diurnal_hourly(df: pd.DataFrame, output_dir: Path):
    """Plot diurnal and hourly consumption dynamics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Diurnal & Hourly Water Consumption Dynamics", fontsize=15, fontweight="bold", y=0.98)

    # 2A: Hourly Total Volume and Count
    ax = axes[0, 0]
    hourly = df.groupby("hour").agg(total_vol=("value", "sum"), event_count=("value", "count")).reset_index()
    ax2 = ax.twinx()
    b1 = ax.bar(hourly["hour"] - 0.2, hourly["total_vol"], width=0.4, color="#1f77b4", label="Total Volume (m³)")
    b2 = ax2.bar(hourly["hour"] + 0.2, hourly["event_count"], width=0.4, color="#ff7f0e", alpha=0.7, label="Event Count")
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Total Water Volume (m³)", color="#1f77b4")
    ax2.set_ylabel("Event Count", color="#ff7f0e")
    ax.set_title("Total Consumption Volume & Event Frequency by Hour")
    ax.set_xticks(range(24))
    ax.grid(True, linestyle="--", alpha=0.5)

    # 2B: Mean Consumption Volume per event by Hour with error bands
    ax = axes[0, 1]
    hourly_stats = df.groupby("hour")["value"].agg(["mean", "std", "count"]).reset_index()
    hourly_stats["sem"] = hourly_stats["std"] / np.sqrt(hourly_stats["count"])
    ax.plot(hourly_stats["hour"], hourly_stats["mean"] * 1000, marker="o", color="#d62728", lw=2, label="Mean Event Volume (Liters)")
    ax.fill_between(
        hourly_stats["hour"],
        (hourly_stats["mean"] - 1.96 * hourly_stats["sem"]) * 1000,
        (hourly_stats["mean"] + 1.96 * hourly_stats["sem"]) * 1000,
        color="#d62728",
        alpha=0.2,
        label="95% Confidence Interval",
    )
    ax.set_title("Average Event Volume by Hour (Liters / Event)")
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Mean Volume (Liters)")
    ax.set_xticks(range(24))
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 2C: Fixture Breakdown by Hour of Day
    ax = axes[1, 0]
    fix_hourly = df.groupby(["hour", "attached_to"])["value"].sum().unstack()
    fix_hourly.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", width=0.8)
    ax.set_title("Hourly Total Water Volume by Fixture (Stacked m³)")
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Cumulative Volume (m³)")
    ax.set_xticklabels(range(24), rotation=0)
    ax.legend(title="Fixture", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    # 2D: Night-time water consumption (01:00 - 05:00) by fixture
    ax = axes[1, 1]
    night_df = df[df["hour"].isin([1, 2, 3, 4])]
    night_vol = night_df.groupby("attached_to")["value"].sum().sort_values(ascending=False)
    sns.barplot(x=night_vol.index, y=night_vol.values, palette="Blues_r", ax=ax)
    ax.set_title("Night-Time Consumption (01:00 - 05:00 UTC) by Fixture\n(Critical Baseline for Leak Detection)")
    ax.set_xlabel("Fixture")
    ax.set_ylabel("Total Night Volume (m³)")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f} m³", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    save_path = output_dir / "02_diurnal_hourly_patterns.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved %s", save_path)


def plot_temporal_seasonality(df: pd.DataFrame, output_dir: Path):
    """Plot Day of Week, Heatmap, and Multi-Year Timeline."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Temporal Trends, Weekly Profiles & Seasonality", fontsize=15, fontweight="bold", y=0.98)

    # 3A: Day of Week Total Volume and Mean
    ax = axes[0, 0]
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = df.groupby("day_name").agg(total_vol=("value", "sum"), event_count=("value", "count")).reindex(days_order)
    sns.barplot(x=dow.index, y=dow["total_vol"], palette="Blues_d", ax=ax)
    ax.set_title("Total Consumption Volume by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Total Volume (m³)")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.0f} m³", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)
    ax.tick_params(axis="x", rotation=20)

    # 3B: Heatmap of Consumption by Hour vs Day of Week
    ax = axes[0, 1]
    pivot_hd = df.pivot_table(index="day_name", columns="hour", values="value", aggfunc="mean").reindex(days_order)
    sns.heatmap(pivot_hd * 1000, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Mean Volume (L/event)"})
    ax.set_title("Hour vs Day-of-Week Interaction Heatmap (Liters/event)")
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Day of Week")

    # 3C: Monthly Volume Timeline across years (showing outages)
    ax = axes[1, 0]
    df["year_month"] = df["timestamp"].dt.to_period("M")
    monthly_series = df.groupby("year_month")["value"].sum()
    monthly_series.index = monthly_series.index.to_timestamp()
    ax.plot(monthly_series.index, monthly_series.values, marker="o", color="#2b5c8f", lw=2)
    ax.set_title("Monthly Total Water Volume Over Time (2019-2023)")
    ax.set_xlabel("Timeline")
    ax.set_ylabel("Total Volume (m³ / month)")
    # Annotate outages
    ax.axvspan(pd.Timestamp("2020-07-01"), pd.Timestamp("2020-07-31"), color="red", alpha=0.15, label="Data Gap (Jul 2020)")
    ax.axvspan(pd.Timestamp("2021-12-01"), pd.Timestamp("2022-01-31"), color="orange", alpha=0.15, label="Data Gap (Dec 21 - Jan 22)")
    ax.axvspan(pd.Timestamp("2022-09-01"), pd.Timestamp("2022-09-30"), color="purple", alpha=0.15, label="Data Gap (Sep 2022)")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 3D: Monthly Event Counts
    ax = axes[1, 1]
    monthly_counts = df.groupby("year_month")["value"].count()
    monthly_counts.index = monthly_counts.index.to_timestamp()
    ax.bar(monthly_counts.index, monthly_counts.values, width=20, color="#41b6c4", edgecolor="black", alpha=0.8)
    ax.set_title("Monthly Event Count (Sampling Uniformity Check)")
    ax.set_xlabel("Timeline")
    ax.set_ylabel("Number of Meter Readings")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = output_dir / "03_day_of_week_and_seasonality.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved %s", save_path)


def plot_fixtures_and_water_type(df: pd.DataFrame, output_dir: Path):
    """Plot fixture end-use distribution and hot vs cold water dynamics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Fixture End-Uses & Water Type (Hot vs Cold) Diagnostics", fontsize=15, fontweight="bold", y=0.98)

    # 4A: Fixture Total Volume & Percentage of Total
    ax = axes[0, 0]
    fix_vols = df.groupby("attached_to")["value"].sum().sort_values(ascending=False)
    fix_pcts = (fix_vols / fix_vols.sum()) * 100
    bars = sns.barplot(x=fix_vols.index, y=fix_vols.values, palette="mako", ax=ax)
    ax.set_title("Total Water Volume by Fixture (End-Use Distribution)")
    ax.set_xlabel("Fixture Type")
    ax.set_ylabel("Total Volume (m³)")
    for bar, pct in zip(ax.patches, fix_pcts):
        ax.annotate(f"{bar.get_height():.1f} m³\n({pct:.1f}%)",
                    (bar.get_x() + bar.get_width() / 2., bar.get_height() / 2),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax.tick_params(axis="x", rotation=15)

    # 4B: Hot vs Cold by Fixture (Proportion stacked bar)
    ax = axes[0, 1]
    ct = pd.crosstab(df["attached_to"], df["type"], values=df["value"], aggfunc="sum", normalize="index") * 100
    ct = ct[["cold water consumption", "hot water consumption"]]
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#1f77b4", "#d62728"], width=0.6)
    ax.set_title("Water Type Split (% Cold vs Hot Volume per Fixture)")
    ax.set_xlabel("Fixture Type")
    ax.set_ylabel("Volume Percentage (%)")
    ax.legend(["Cold Water (Mains)", "Hot Water (Heated)"], frameon=True)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=15)

    # 4C: Violin Plot of Log Volume per Fixture
    ax = axes[1, 0]
    df_sample = df.sample(n=min(50000, len(df)), random_state=42).copy()
    df_sample["log_val"] = np.log10(df_sample["value"] * 1000)  # log10(liters)
    sns.violinplot(data=df_sample, x="attached_to", y="log_val", palette="Set2", ax=ax, cut=0)
    ax.set_title("Volume Distribution Density by Fixture (log10 Liters)")
    ax.set_xlabel("Fixture Type")
    ax.set_ylabel("Log10 Volume in Liters (0 = 1L, 1 = 10L, 2 = 100L)")
    ax.axhline(0, color="gray", linestyle=":", label="1 Liter")
    ax.axhline(1, color="gray", linestyle="--", label="10 Liters")
    ax.axhline(2, color="gray", linestyle="-.", label="100 Liters")
    ax.legend(frameon=True, loc="upper right")
    ax.tick_params(axis="x", rotation=15)

    # 4D: Room Type Consumption Breakdown
    ax = axes[1, 1]
    room_vol = df.groupby("room_type")["value"].sum().sort_values(ascending=False)
    room_pct = (room_vol / room_vol.sum()) * 100
    sns.barplot(x=room_vol.index, y=room_vol.values, palette="viridis", ax=ax)
    ax.set_title("Total Water Consumption by Room Type")
    ax.set_xlabel("Room Type")
    ax.set_ylabel("Total Volume (m³)")
    for p, pct in zip(ax.patches, room_pct):
        ax.annotate(f"{pct:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)
    ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    save_path = output_dir / "04_fixtures_and_water_type.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved %s", save_path)


def plot_spatial_apartments(df: pd.DataFrame, output_dir: Path):
    """Plot apartment and cluster level distributions (Pareto analysis)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Spatial Consumption Analysis: Units, Clusters & Disparities", fontsize=15, fontweight="bold", y=0.98)

    # 5A: Pareto Chart of Apartments
    ax = axes[0, 0]
    apt_vol = df.groupby(df["apartment"].fillna("Shared/Unassigned"))["value"].sum().sort_values(ascending=False)
    apt_pct = (apt_vol / apt_vol.sum()) * 100
    cum_pct = apt_pct.cumsum()
    x_positions = range(len(apt_vol))
    bars = ax.bar(x_positions, apt_vol.values, color="#3470a3", width=0.8, label="Volume (m³)")
    ax2 = ax.twinx()
    ax2.plot(x_positions, cum_pct.values, color="#e41a1c", marker="o", lw=2, markersize=4, label="Cumulative %")
    ax2.axhline(80, color="gray", linestyle="--", label="80% Pareto Threshold")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(apt_vol.index, rotation=90, fontsize=8)
    ax.set_title("Apartment Consumption Pareto Distribution")
    ax.set_ylabel("Total Volume (m³)", color="#3470a3")
    ax2.set_ylabel("Cumulative Percentage (%)", color="#e41a1c")
    ax2.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.4)

    # 5B: Cluster Total Volume
    ax = axes[0, 1]
    cl_vol = df.groupby(df["cluster_name"].fillna("Unassigned"))["value"].sum().sort_values(ascending=False)
    sns.barplot(x=cl_vol.index, y=cl_vol.values, palette="Spectral", ax=ax)
    ax.set_title("Water Consumption by Building Cluster (CL_A - CL_I + Unassigned)")
    ax.set_xlabel("Cluster Identifier")
    ax.set_ylabel("Total Volume (m³)")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.0f} m³", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)
    ax.tick_params(axis="x", rotation=20)

    # 5C: Top 15 Sensors by Event Count
    ax = axes[1, 0]
    top_sensors = df["sensor_id"].value_counts().head(15)
    sns.barplot(x=top_sensors.index, y=top_sensors.values, palette="crest", ax=ax)
    ax.set_title("Top 15 Most Active Smart Sensors (Event Frequency)")
    ax.set_xlabel("Sensor ID")
    ax.set_ylabel("Total Event Count")
    ax.tick_params(axis="x", rotation=30)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=8)

    # 5D: Sensor Event Distribution (Histogram of events per sensor)
    ax = axes[1, 1]
    sensor_counts = df["sensor_id"].value_counts()
    sns.histplot(sensor_counts, bins=30, kde=True, color="#8856a7", ax=ax)
    ax.set_title("Sensor Activity Disparity Distribution (Events per Sensor)")
    ax.set_xlabel("Number of Events Recorded by Sensor")
    ax.set_ylabel("Sensor Count")
    ax.axvline(sensor_counts.median(), color="red", linestyle="--", label=f"Median: {sensor_counts.median():.0f}")
    ax.legend(frameon=True)

    plt.tight_layout()
    save_path = output_dir / "05_spatial_apartments_and_clusters.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved %s", save_path)


def plot_anomaly_diagnostics(df: pd.DataFrame, output_dir: Path):
    """Plot anomaly diagnostics, outlier cutoffs, and sensor deltas."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Anomaly & Outlier Diagnostics for Machine Learning", fontsize=15, fontweight="bold", y=0.98)

    # 6A: Scatter plot value vs aggregated_value
    ax = axes[0, 0]
    sample_df = df.sample(n=min(30000, len(df)), random_state=42)
    sns.scatterplot(data=sample_df, x="aggregated_value", y="value", hue="attached_to", alpha=0.5, s=20, ax=ax, palette="tab10")
    ax.set_title("Event Value vs Aggregated Meter Reading (Sample 30k)")
    ax.set_xlabel("Aggregated Meter Value (m³)")
    ax.set_ylabel("Event Value (m³)")
    p99 = df["value"].quantile(0.99)
    p999 = df["value"].quantile(0.999)
    ax.axhline(p99, color="orange", linestyle="--", label=f"99th Pct ({p99:.3f} m³)")
    ax.axhline(p999, color="red", linestyle="-.", label=f"99.9th Pct ({p999:.3f} m³)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=8)

    # 6B: Outlier Flag Proportions under Alternative Heuristics
    ax = axes[0, 1]
    q25 = df["value"].quantile(0.25)
    q75 = df["value"].quantile(0.75)
    iqr = q75 - q25
    iqr_15 = q75 + 1.5 * iqr
    iqr_30 = q75 + 3.0 * iqr
    z_score_3 = df["value"].mean() + 3.0 * df["value"].std()

    heuristics = {
        "IQR 1.5x (>0.011)": (df["value"] > iqr_15).mean() * 100,
        "IQR 3.0x (>0.017)": (df["value"] > iqr_30).mean() * 100,
        "Z-Score > 3 (>0.033)": (df["value"] > z_score_3).mean() * 100,
        "99th Pct (>0.045)": (df["value"] > p99).mean() * 100,
        "99.9th Pct (>0.104)": (df["value"] > p999).mean() * 100,
        "> 0.200 m³ (>200L)": (df["value"] > 0.200).mean() * 100,
    }
    h_df = pd.Series(heuristics)
    sns.barplot(x=h_df.values, y=h_df.index, palette="Reds_r", ax=ax)
    ax.set_title("Outlier Candidate Percentage by Statistical Heuristic")
    ax.set_xlabel("Dataset Percentage Flagged (%)")
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.2f}%", (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha="left", va="center", xytext=(5, 0), textcoords="offset points", fontsize=9)

    # 6C: Cumulative time series for top active sensor
    ax = axes[1, 0]
    top_sensor = df["sensor_id"].value_counts().index[0]
    s_df = df[df["sensor_id"] == top_sensor].sort_values("timestamp")
    ax.plot(s_df["timestamp"], s_df["aggregated_value"], color="#1f77b4", lw=1.2)
    ax.set_title(f"Cumulative Meter Progression for Top Sensor ({top_sensor})")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Aggregated Value (m³)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 6D: Extreme Events (>0.25 m³) Breakdown by Fixture
    ax = axes[1, 1]
    extreme_df = df[df["value"] >= 0.25]
    sns.countplot(data=extreme_df, x="attached_to", palette="magma", ax=ax)
    ax.set_title(f"Extreme Consumption Events (>= 250 Liters, N={len(extreme_df)}) by Fixture")
    ax.set_xlabel("Fixture Type")
    ax.set_ylabel("Extreme Event Count")
    for p in ax.patches:
        ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    save_path = output_dir / "06_anomalies_outliers_diagnostics.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved %s", save_path)


def plot_correlations(df: pd.DataFrame, output_dir: Path):
    """Plot correlation matrix heatmap."""
    plt.figure(figsize=(9, 7))
    num_cols = ["hour", "day_of_week", "month", "year", "value", "aggregated_value"]
    corr = df[num_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=0.3,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix (Numerical & Temporal Features)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    save_path = output_dir / "07_correlation_matrix.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved %s", save_path)


def plot_data_quality_and_missingness(df: pd.DataFrame, output_dir: Path):
    """Plot missingness breakdown and data quality diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Data Quality & Missingness Diagnostics", fontsize=15, fontweight="bold", y=0.98)

    # 8A: Missing values per column
    ax = axes[0]
    null_pct = (df.isnull().sum() / len(df)) * 100
    null_cols = null_pct[null_pct > 0].sort_values(ascending=False)
    if not null_cols.empty:
        sns.barplot(x=null_cols.index, y=null_cols.values, palette="Reds_r", ax=ax)
        ax.set_title("Missing Values Percentage by Column")
        ax.set_ylabel("Missing Percentage (%)")
        ax.set_ylim(0, 30)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=10)
    else:
        ax.text(0.5, 0.5, "No Missing Values", ha="center", va="center")

    # 8B: Missingness breakdown by room type
    ax = axes[1]
    missing_apt = df[df["apartment"].isnull()]["room_type"].value_counts(normalize=True) * 100
    sns.barplot(x=missing_apt.index, y=missing_apt.values, palette="Blues_r", ax=ax)
    ax.set_title("Room Type Distribution for Rows with Missing Apartment (NaN)\n(Validates Missingness is Shared Facilities, Not Corrupt Data)")
    ax.set_xlabel("Room Type")
    ax.set_ylabel("Percentage of Missing Rows (%)")
    ax.tick_params(axis="x", rotation=25)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)

    plt.tight_layout()
    save_path = output_dir / "08_missingness_and_data_quality.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved %s", save_path)


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive EDA and generate ML figures.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/HSB_Living_Lab_Water_Consumption_Anonymized.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eda/figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_preprocess(args.input)

    logger.info("Generating target distribution plots...")
    plot_target_distributions(df, args.output_dir)

    logger.info("Generating diurnal and hourly dynamics plots...")
    plot_diurnal_hourly(df, args.output_dir)

    logger.info("Generating temporal seasonality plots...")
    plot_temporal_seasonality(df, args.output_dir)

    logger.info("Generating fixture and water type plots...")
    plot_fixtures_and_water_type(df, args.output_dir)

    logger.info("Generating spatial apartment plots...")
    plot_spatial_apartments(df, args.output_dir)

    logger.info("Generating anomaly diagnostics plots...")
    plot_anomaly_diagnostics(df, args.output_dir)

    logger.info("Generating correlation matrix plot...")
    plot_correlations(df, args.output_dir)

    logger.info("Generating data quality and missingness plots...")
    plot_data_quality_and_missingness(df, args.output_dir)

    logger.info("All 8 EDA figures successfully generated in: %s", args.output_dir)


if __name__ == "__main__":
    main()
