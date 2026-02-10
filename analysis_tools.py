"""
Native function-calling analysis tools for VayuChat.

Naming convention signals return type:
  get_*  → returns a text string   (rendered via st.markdown)
  df_*   → returns a pd.DataFrame  (rendered via st.dataframe)
  plot_* → returns a .png filename  (rendered via st.image)

Architecture: shared _data helpers → thin get_/df_/plot_ wrappers.
"""

import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from google import genai

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_POLLUTANT_COLS = [
    "PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO (µg/m³)", "NO2 (µg/m³)",
    "NOx (ppb)", "NH3 (µg/m³)", "SO2 (µg/m³)", "CO (mg/m³)", "Ozone (µg/m³)",
]
_MET_COLS = [
    "AT (°C)", "RH (%)", "WS (m/s)", "WD (deg)", "RF (mm)",
    "TOT-RF (mm)", "SR (W/mt2)", "BP (mmHg)", "VWS (m/s)",
]
_ALL_NUMERIC_COLS = _POLLUTANT_COLS + _MET_COLS

_GUIDELINES = {
    "PM2.5 (µg/m³)": {"India": 60, "WHO": 15},
    "PM10 (µg/m³)":  {"India": 100, "WHO": 50},
}

_SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
}

_MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

_NORTH_STATES = {
    "Delhi", "Haryana", "Punjab", "Uttar Pradesh", "Uttarakhand",
    "Himachal Pradesh", "Jammu and Kashmir", "Rajasthan",
    "Chandigarh", "Ladakh",
}
_SOUTH_STATES = {
    "Tamil Nadu", "Kerala", "Karnataka", "Andhra Pradesh",
    "Telangana", "Puducherry",
}


# ---------------------------------------------------------------------------
# City coordinates lookup (built from Data.csv at import time)
# ---------------------------------------------------------------------------

def _build_city_coords():
    """Read Data.csv once and return {city_lower: (lat, lon)} for cities with valid coords."""
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data.csv")
    if not os.path.exists(path):
        return {}
    try:
        raw = pd.read_csv(path, usecols=["city", "latitude", "longitude"])
        raw = raw.dropna(subset=["latitude", "longitude"])
        raw["city_lower"] = raw["city"].str.strip().str.lower()
        coords = raw.groupby("city_lower")[["latitude", "longitude"]].median()
        return {city: (row["latitude"], row["longitude"]) for city, row in coords.iterrows()}
    except Exception:
        return {}

_CITY_COORDS = _build_city_coords()


def _load_india_boundary():
    """Load simplified India boundary GeoJSON (official, includes J&K + Ladakh)."""
    import os, json as _json
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "india_boundary.geojson")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return _json.load(f)
    except Exception:
        return None

_INDIA_GEOJSON = _load_india_boundary()


def _merge_coords(agg_df, city_col="City"):
    """Join lat/lon from _CITY_COORDS onto an aggregated DataFrame with a city column."""
    agg_df = agg_df.copy()
    agg_df["latitude"] = agg_df[city_col].str.strip().str.lower().map(
        lambda c: _CITY_COORDS.get(c, (None, None))[0]
    )
    agg_df["longitude"] = agg_df[city_col].str.strip().str.lower().map(
        lambda c: _CITY_COORDS.get(c, (None, None))[1]
    )
    return agg_df.dropna(subset=["latitude", "longitude"])


# ---------------------------------------------------------------------------
# Shared data helpers  (_data_*)
# ---------------------------------------------------------------------------

def _resolve_col(name):
    """Resolve a loose column name to the exact column name in the data."""
    if not name:
        return "PM2.5 (µg/m³)"
    name_lower = name.lower().strip()
    for col in _ALL_NUMERIC_COLS:
        if col.lower() == name_lower:
            return col
    for col in _ALL_NUMERIC_COLS:
        short = col.split(" (")[0].lower()
        if name_lower == short or name_lower == short.replace(".", ""):
            return col
    keyword_map = {
        "pm25": "PM2.5 (µg/m³)", "pm2.5": "PM2.5 (µg/m³)", "pm 2.5": "PM2.5 (µg/m³)",
        "pm10": "PM10 (µg/m³)", "pm 10": "PM10 (µg/m³)",
        "no2": "NO2 (µg/m³)", "nitrogen dioxide": "NO2 (µg/m³)",
        "no": "NO (µg/m³)", "nitric oxide": "NO (µg/m³)",
        "nox": "NOx (ppb)", "nitrogen oxides": "NOx (ppb)",
        "nh3": "NH3 (µg/m³)", "ammonia": "NH3 (µg/m³)",
        "so2": "SO2 (µg/m³)", "sulfur dioxide": "SO2 (µg/m³)", "sulphur dioxide": "SO2 (µg/m³)",
        "co": "CO (mg/m³)", "carbon monoxide": "CO (mg/m³)",
        "ozone": "Ozone (µg/m³)", "o3": "Ozone (µg/m³)",
        "temperature": "AT (°C)", "temp": "AT (°C)",
        "humidity": "RH (%)", "rh": "RH (%)",
        "wind speed": "WS (m/s)", "wind": "WS (m/s)", "ws": "WS (m/s)",
        "wind direction": "WD (deg)", "wd": "WD (deg)",
        "rainfall": "RF (mm)", "rain": "RF (mm)", "rf": "RF (mm)",
        "solar radiation": "SR (W/mt2)", "sr": "SR (W/mt2)",
        "barometric pressure": "BP (mmHg)", "bp": "BP (mmHg)", "pressure": "BP (mmHg)",
        "pollution": "PM2.5 (µg/m³)",
    }
    if name_lower in keyword_map:
        return keyword_map[name_lower]
    return name


def _col_unit(col):
    return col.split("(")[-1].rstrip(")") if "(" in col else ""


def _filter_df(df, year=None, month=None, cities=None, states=None):
    """Apply common filters.  Returns a (possibly smaller) DataFrame."""
    d = df
    if year:
        d = d[d["Year"] == int(year)]
    if month:
        d = d[d["Timestamp"].dt.month == int(month)]
    if cities:
        city_list = [c.strip() for c in cities] if isinstance(cities, list) else [cities.strip()]
        d = d[d["City"].str.lower().isin([c.lower() for c in city_list])]
    if states:
        state_list = [s.strip() for s in states] if isinstance(states, list) else [states.strip()]
        d = d[d["State"].str.lower().isin([s.lower() for s in state_list])]
    return d


def _add_derived_group(d, group_by):
    """Add Month / Season columns if needed.  Returns (df_copy, group_col)."""
    if group_by == "Month":
        d = d.copy()
        d["Month"] = d["Timestamp"].dt.month
        return d, "Month"
    if group_by == "Season":
        d = d.copy()
        d["Season"] = d["Timestamp"].dt.month.map(_SEASON_MAP)
        return d, "Season"
    return d, group_by


def _data_grouped_stats(df, col, group_by, agg_func="mean",
                        year=None, month=None, cities=None, states=None,
                        top_n=None, sort_order="desc"):
    """
    Core groupby+agg computation shared by get_statistics, df_ranking, and
    plot functions that need ranked data.

    Returns a DataFrame with columns: [group_by, 'mean', 'std', 'count']
    (plus an extra agg column if agg_func not in the base set),
    sorted and optionally truncated.  Returns None if no data.
    """
    d = _filter_df(df, year=year, month=month, cities=cities, states=states)
    d = d.dropna(subset=[col])
    if d.empty:
        return None

    d, gcol = _add_derived_group(d, group_by)

    agg_map = {"mean": "mean", "max": "max", "min": "min", "median": "median", "std": "std"}
    func = agg_map.get(agg_func, "mean")

    base_aggs = ["mean", "std", "count"]
    if func not in base_aggs:
        base_aggs.append(func)
    grouped = d.groupby(gcol)[col].agg(base_aggs).reset_index()

    sort_col = func if func in grouped.columns else "mean"
    ascending = sort_order == "asc"
    grouped = grouped.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

    if top_n:
        grouped = grouped.head(int(top_n))

    return grouped


def _data_trend(df, col, freq="monthly", cities=None, states=None,
                year_start=None, year_end=None):
    """
    Core trend computation.  Returns a dict of {label: Series} where Series
    index is the time period and values are the mean of `col`.
    Returns None if no data.
    """
    d = df.dropna(subset=[col])
    if cities:
        d = _filter_df(d, cities=cities)
    if states:
        d = _filter_df(d, states=states)
    if year_start:
        d = d[d["Year"] >= int(year_start)]
    if year_end:
        d = d[d["Year"] <= int(year_end)]
    if d.empty:
        return None

    group_col = "City" if cities else ("State" if states else None)

    series_dict = {}
    if freq == "yearly":
        if group_col:
            for name, grp in d.groupby(group_col):
                series_dict[name] = grp.groupby("Year")[col].mean()
        else:
            series_dict["All"] = d.groupby("Year")[col].mean()
    else:
        d = d.copy()
        d["YearMonth"] = d["Timestamp"].dt.to_period("M")
        if group_col:
            for name, grp in d.groupby(group_col):
                s = grp.groupby("YearMonth")[col].mean()
                s.index = s.index.astype(str)
                series_dict[name] = s
        else:
            s = d.groupby("YearMonth")[col].mean()
            s.index = s.index.astype(str)
            series_dict["All"] = s

    return series_dict


def _data_exceedances(df, col, threshold, direction="above",
                      group_by="City", year=None):
    """
    Core exceedance computation.
    Returns a DataFrame with [group_by, 'mean', 'std', 'count'] filtered by
    threshold, or None if no data / no matches.
    """
    d = _filter_df(df, year=year).dropna(subset=[col])
    if d.empty:
        return None

    agg = d.groupby(group_by)[col].agg(["mean", "std", "count"]).reset_index()

    if direction == "above":
        result = agg[agg["mean"] > float(threshold)].sort_values("mean", ascending=False)
    else:
        result = agg[agg["mean"] < float(threshold)].sort_values("mean", ascending=True)

    return result.reset_index(drop=True) if not result.empty else None


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _apply_style():
    try:
        plt.style.use("vayuchat.mplstyle")
    except OSError:
        pass


def _save_fig(fig):
    fname = f"plot_{uuid.uuid4().hex[:8]}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return fname


def _add_guidelines(ax, col):
    if col in _GUIDELINES:
        g = _GUIDELINES[col]
        ax.axhline(g["India"], color="red", linestyle="--", linewidth=1, label=f"India Std ({g['India']})")
        ax.axhline(g["WHO"], color="green", linestyle="--", linewidth=1, label=f"WHO Std ({g['WHO']})")


# ===================================================================
# get_* → text string
# ===================================================================

def get_statistics(df, pollutant="PM2.5 (µg/m³)", group_by="City",
                   agg_func="mean", year=None, month=None,
                   top_n=None, sort_order="desc",
                   cities=None, states=None, **_kw):
    col = _resolve_col(pollutant)
    grouped = _data_grouped_stats(df, col, group_by, agg_func,
                                  year=year, month=month, cities=cities,
                                  states=states, top_n=top_n, sort_order=sort_order)
    if grouped is None:
        return "No data available for the specified filters."

    agg_map = {"mean": "mean", "max": "max", "min": "min", "median": "median", "std": "std"}
    func = agg_map.get(agg_func, "mean")
    sort_col = func if func in grouped.columns else "mean"

    # Build header
    filters = []
    if year:  filters.append(f"Year: {year}")
    if month: filters.append(f"Month: {month}")
    if cities: filters.append(f"Cities: {', '.join(cities) if isinstance(cities, list) else cities}")
    if states: filters.append(f"States: {', '.join(states) if isinstance(states, list) else states}")
    filter_str = f" ({', '.join(filters)})" if filters else ""

    unit = _col_unit(col)
    lines = [f"**{agg_func.capitalize()} {col} by {group_by}**{filter_str}\n"]
    for _, row in grouped.iterrows():
        val = round(row[sort_col], 2)
        std_val = round(row["std"], 2) if pd.notna(row["std"]) else "N/A"
        n = int(row["count"])
        label = _MONTH_NAMES.get(row[group_by], row[group_by]) if group_by == "Month" else row[group_by]
        lines.append(f"- **{label}**: {val} {unit} (SD: {std_val}, n={n})")

    return "\n".join(lines)


def get_exceedances(df, pollutant="PM2.5 (µg/m³)", threshold=60,
                    direction="above", group_by="City", year=None, **_kw):
    col = _resolve_col(pollutant)
    result = _data_exceedances(df, col, threshold, direction, group_by, year)
    if result is None:
        dir_label = "above" if direction == "above" else "below"
        return f"No {group_by.lower()}s have average {col} {dir_label} {threshold}."

    unit = _col_unit(col)
    dir_label = "above" if direction == "above" else "below"
    lines = [f"**{group_by}s with average {col} {dir_label} {threshold} {unit}**\n"]
    for _, row in result.iterrows():
        lines.append(f"- **{row[group_by]}**: {round(row['mean'], 2)} {unit}")
    lines.append(f"\nTotal: {len(result)} {group_by.lower()}(s)")
    return "\n".join(lines)


def get_ncap_analysis(df, ncap_df, pollutant="PM2.5 (µg/m³)",
                      analysis_type="exceed_guidelines", cities=None, **_kw):
    col = _resolve_col(pollutant)

    # Filter ncap_df to specific cities if requested
    ndf = ncap_df
    if cities:
        city_list = [c.strip().lower() for c in (cities if isinstance(cities, list) else [cities])]
        ndf = ncap_df[ncap_df["city"].str.lower().isin(city_list)]
        if ndf.empty:
            return f"No NCAP funding data found for: {', '.join(cities) if isinstance(cities, list) else cities}"

    if analysis_type == "exceed_guidelines":
        ncap_cities = ndf["city"].str.lower().unique()
        d = df[df["City"].str.lower().isin(ncap_cities)]
        result = _data_exceedances(d, col, _GUIDELINES.get(col, {}).get("India", 60),
                                   "above", "City")
        if result is None:
            guideline = _GUIDELINES.get(col, {}).get("India", 60)
            return f"No NCAP cities exceed the India guideline ({guideline}) for {col}."
        unit = _col_unit(col)
        guideline = _GUIDELINES.get(col, {}).get("India", 60)
        lines = [f"**NCAP cities exceeding India guideline ({guideline} {unit}) for {col}:**\n"]
        for _, row in result.iterrows():
            lines.append(f"- **{row['City']}**: {round(row['mean'], 2)} {unit}")
        return "\n".join(lines)

    elif analysis_type == "funding_summary":
        if cities:
            rows = ndf.sort_values("Total fund released", ascending=False)
        else:
            rows = ndf.nlargest(15, "Total fund released")
        header = f"**NCAP funding for {', '.join(cities) if isinstance(cities, list) else cities}:**\n" if cities else "**Top 15 NCAP-funded cities:**\n"
        lines = [header]
        for _, row in rows.iterrows():
            lines.append(f"- **{row['city']}** ({row['state']}): {round(row['Total fund released'], 2)} ₹ Cr")
        total = ndf["Total fund released"].sum()
        lines.append(f"\nTotal: {round(total, 2)} ₹ Cr")
        return "\n".join(lines)

    elif analysis_type == "reduction":
        ncap_cities = ndf["city"].str.lower().unique()
        d = df[df["City"].str.lower().isin(ncap_cities)].dropna(subset=[col])
        if d.empty:
            return f"No {col} data found for NCAP cities."

        # For each city: compare mean of earliest 2 years vs latest 2 years
        unit = _col_unit(col)
        city_yearly = d.groupby(["City", "Year"])[col].mean().reset_index()
        reductions = []
        for city, grp in city_yearly.groupby("City"):
            grp = grp.sort_values("Year")
            years = grp["Year"].unique()
            if len(years) < 2:
                continue
            early = grp.head(2)[col].mean()
            late = grp.tail(2)[col].mean()
            change = late - early
            pct = (change / early * 100) if early != 0 else 0
            reductions.append({
                "City": city, "early": early, "late": late,
                "change": change, "pct": pct,
                "yr_early": int(years[0]), "yr_late": int(years[-1]),
            })

        if not reductions:
            return "Not enough yearly data to compute reductions."

        rdf = pd.DataFrame(reductions).sort_values("change")  # most negative = best reduction
        improved = sum(1 for r in reductions if r["change"] < 0)
        worsened = sum(1 for r in reductions if r["change"] > 0)

        # Show top 15 improvers + top 5 worseners
        top_improved = rdf.head(15)
        top_worsened = rdf.tail(5).iloc[::-1] if worsened > 0 else pd.DataFrame()

        lines = [f"**{col} reduction in NCAP cities (earliest 2 yrs → latest 2 yrs):**\n"]
        lines.append(f"**Top improvers:**")
        for _, row in top_improved.iterrows():
            lines.append(
                f"- **{row['City']}**: {round(row['early'], 1)} → {round(row['late'], 1)} {unit} "
                f"(↓ {abs(round(row['change'], 1))} {unit}, {abs(round(row['pct'], 1))}%)"
            )
        if not top_worsened.empty:
            lines.append(f"\n**Worsened:**")
            for _, row in top_worsened.iterrows():
                if row["change"] > 0:
                    lines.append(
                        f"- **{row['City']}**: {round(row['early'], 1)} → {round(row['late'], 1)} {unit} "
                        f"(↑ {round(row['change'], 1)} {unit}, +{round(row['pct'], 1)}%)"
                    )
        lines.append(f"\nOverall: {improved}/{len(reductions)} NCAP cities showed improvement, {worsened} worsened.")
        return "\n".join(lines)

    return "Unknown NCAP analysis type."


# ===================================================================
# df_* → pd.DataFrame
# ===================================================================

def df_ranking(df, pollutant="PM2.5 (µg/m³)", group_by="City",
               agg_func="mean", year=None, month=None,
               top_n=None, sort_order="desc",
               cities=None, states=None, **_kw):
    col = _resolve_col(pollutant)
    grouped = _data_grouped_stats(df, col, group_by, agg_func,
                                  year=year, month=month, cities=cities,
                                  states=states, top_n=top_n, sort_order=sort_order)
    if grouped is None:
        return "No data available for the specified filters."

    # Pretty-print Month numbers
    if group_by == "Month":
        grouped["Month"] = grouped["Month"].map(_MONTH_NAMES)

    unit = _col_unit(col)
    rename = {group_by: group_by, "mean": f"Mean ({unit})", "std": f"Std Dev ({unit})", "count": "Data Points"}
    agg_map = {"mean": "mean", "max": "max", "min": "min", "median": "median", "std": "std"}
    func = agg_map.get(agg_func, "mean")
    if func not in ("mean", "std", "count"):
        rename[func] = f"{agg_func.capitalize()} ({unit})"
    grouped = grouped.rename(columns=rename)

    for c in grouped.select_dtypes(include="number").columns:
        grouped[c] = grouped[c].round(2)

    grouped.index = range(1, len(grouped) + 1)
    grouped.index.name = "Rank"
    return grouped.reset_index()


def df_exceedances(df, pollutant="PM2.5 (µg/m³)", threshold=60,
                   direction="above", group_by="City", year=None, **_kw):
    col = _resolve_col(pollutant)
    result = _data_exceedances(df, col, threshold, direction, group_by, year)
    if result is None:
        dir_label = "above" if direction == "above" else "below"
        return f"No {group_by.lower()}s have average {col} {dir_label} {threshold}."

    unit = _col_unit(col)
    result = result.rename(columns={
        "mean": f"Mean ({unit})", "std": f"Std Dev ({unit})", "count": "Data Points",
    })
    for c in result.select_dtypes(include="number").columns:
        result[c] = result[c].round(2)
    return result.reset_index(drop=True)


# ===================================================================
# plot_* → .png filename
# ===================================================================

def plot_trend(df, pollutant="PM2.5 (µg/m³)", cities=None, states=None,
               freq="monthly", year_start=None, year_end=None, **_kw):
    _apply_style()
    col = _resolve_col(pollutant)
    series = _data_trend(df, col, freq, cities, states, year_start, year_end)
    if series is None:
        return "No data available for the specified filters."

    fig, ax = plt.subplots(figsize=(9, 6))
    single = len(series) == 1 and "All" in series
    for label, s in series.items():
        kw = {"color": "tab:red"} if single else {}
        ax.plot(s.index, s.values, marker="o" if freq == "yearly" else None,
                label=None if single else label, **kw)

    if freq == "monthly":
        ticks = ax.get_xticks()
        if len(ticks) > 24:
            ax.set_xticks(ticks[::max(len(ticks) // 12, 1)])

    _add_guidelines(ax, col)
    ax.set_xlabel("Year" if freq == "yearly" else "Month")
    ax.set_ylabel(col)
    title_parts = [f"{freq.capitalize()} Trend of {col}"]
    if cities:
        title_parts.append(f"for {', '.join(cities) if isinstance(cities, list) else cities}")
    if year_start or year_end:
        title_parts.append(f"({year_start or ''}–{year_end or ''})")
    ax.set_title(" ".join(title_parts))
    plt.xticks(rotation=45)
    if not single:
        ax.legend()
    fig.tight_layout()
    return _save_fig(fig)


def plot_comparison(df, pollutant="PM2.5 (µg/m³)", comparison_type="seasonal",
                    cities=None, states=None, year=None, **_kw):
    _apply_style()
    col = _resolve_col(pollutant)
    d = _filter_df(df, year=year, cities=cities, states=states).dropna(subset=[col])
    if d.empty:
        return "No data available for the specified filters."

    fig, ax = plt.subplots(figsize=(9, 6))

    if comparison_type == "seasonal":
        d = d.copy()
        d["Season"] = d["Timestamp"].dt.month.map(_SEASON_MAP)
        order = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
        sns.boxplot(data=d, x="Season", y=col, hue="Season", order=order, ax=ax, palette="Reds", legend=False)
        ax.set_title(f"Seasonal Comparison of {col}")
    elif comparison_type == "weekday_weekend":
        d = d.copy()
        d["DayType"] = d["Timestamp"].dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")
        sns.boxplot(data=d, x="DayType", y=col, hue="DayType", ax=ax, palette="Reds", legend=False)
        ax.set_title(f"Weekday vs Weekend — {col}")
    elif comparison_type == "cities":
        if not cities:
            return "Please specify cities to compare."
        sns.boxplot(data=d, x="City", y=col, hue="City", ax=ax, palette="Reds", legend=False)
        ax.set_title(f"City Comparison — {col}")
    elif comparison_type == "states":
        if not states:
            return "Please specify states to compare."
        sns.boxplot(data=d, x="State", y=col, hue="State", ax=ax, palette="Reds", legend=False)
        ax.set_title(f"State Comparison — {col}")
    elif comparison_type == "north_south":
        d = d.copy()
        d["Region"] = d["State"].apply(
            lambda s: "North" if s in _NORTH_STATES else ("South" if s in _SOUTH_STATES else "Other")
        )
        d = d[d["Region"] != "Other"]
        if d.empty:
            return "No data for North/South regions."
        sns.boxplot(data=d, x="Region", y=col, hue="Region", ax=ax, palette="Reds", legend=False)
        ax.set_title(f"North vs South — {col}")
    else:
        return f"Unknown comparison type: {comparison_type}"

    _add_guidelines(ax, col)
    plt.xticks(rotation=45)
    fig.tight_layout()
    return _save_fig(fig)


def plot_correlation(df, var_x="PM2.5 (µg/m³)", var_y="PM10 (µg/m³)",
                     city=None, state=None, year=None, **_kw):
    _apply_style()
    cx = _resolve_col(var_x)
    cy = _resolve_col(var_y)
    d = _filter_df(df, year=year, cities=city, states=state).dropna(subset=[cx, cy])
    if len(d) < 20:
        return "Insufficient data (< 20 points) for a meaningful correlation."

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(d[cx], d[cy], alpha=0.3, s=10, color="indianred")
    z = np.polyfit(d[cx], d[cy], 1)
    p = np.poly1d(z)
    x_line = np.linspace(d[cx].min(), d[cx].max(), 100)
    ax.plot(x_line, p(x_line), color="black", linewidth=1.5, label=f"y = {z[0]:.2f}x + {z[1]:.2f}")
    corr = d[cx].corr(d[cy])
    ax.set_xlabel(cx)
    ax.set_ylabel(cy)
    title_parts = [f"{cx} vs {cy}"]
    if city:  title_parts.append(f"— {city}")
    if year:  title_parts.append(f"({year})")
    ax.set_title(" ".join(title_parts))
    ax.legend(title=f"r = {corr:.2f}")
    fig.tight_layout()
    return _save_fig(fig)


def plot_ncap(df, ncap_df, pollutant="PM2.5 (µg/m³)",
              analysis_type="funding_comparison", cities=None, **_kw):
    _apply_style()
    col = _resolve_col(pollutant)

    # Filter ncap_df to specific cities if requested
    ndf = ncap_df
    if cities:
        city_list = [c.strip().lower() for c in (cities if isinstance(cities, list) else [cities])]
        ndf = ncap_df[ncap_df["city"].str.lower().isin(city_list)]
        if ndf.empty:
            return f"No NCAP funding data found for: {', '.join(cities) if isinstance(cities, list) else cities}"

    if analysis_type == "funding_comparison":
        if cities:
            rows = ndf.sort_values("Total fund released", ascending=False)
        else:
            rows = ndf.nlargest(15, "Total fund released")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(rows["city"], rows["Total fund released"], color="steelblue", alpha=0.8)
        ax.set_ylabel("Total Fund Released (₹ Cr)")
        title = f"NCAP Funding — {', '.join(cities) if isinstance(cities, list) else cities}" if cities else "NCAP Funding by City"
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        return _save_fig(fig)

    elif analysis_type == "reduction_analysis":
        ncap_cities = ncap_df["city"].str.lower().unique()
        # Reuse _data_trend by splitting NCAP / non-NCAP
        d = df.dropna(subset=[col]).copy()
        d["is_ncap"] = d["City"].str.lower().isin(ncap_cities)
        yearly = d.groupby(["Year", "is_ncap"])[col].mean().reset_index()
        fig, ax = plt.subplots(figsize=(9, 6))
        for label, grp in yearly.groupby("is_ncap"):
            name = "NCAP Cities" if label else "Non-NCAP Cities"
            ax.plot(grp["Year"], grp[col], marker="o", label=name)
        _add_guidelines(ax, col)
        ax.set_xlabel("Year")
        ax.set_ylabel(col)
        ax.set_title(f"{col} Trend: NCAP vs Non-NCAP Cities")
        ax.legend()
        fig.tight_layout()
        return _save_fig(fig)

    return "Unknown NCAP plot type."


def plot_met_impact(df, met_factor="WS (m/s)", pollutant="PM2.5 (µg/m³)",
                    city=None, threshold=None, **_kw):
    _apply_style()
    met_col = _resolve_col(met_factor)
    pol_col = _resolve_col(pollutant)
    d = _filter_df(df, cities=city).dropna(subset=[met_col, pol_col])
    if len(d) < 20:
        return "Insufficient data for meteorological impact analysis."

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter
    ax1 = axes[0]
    ax1.scatter(d[met_col], d[pol_col], alpha=0.2, s=8, color="indianred")
    z = np.polyfit(d[met_col], d[pol_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(d[met_col].min(), d[met_col].max(), 100)
    ax1.plot(x_line, p(x_line), color="black", linewidth=1.5)
    corr = d[met_col].corr(d[pol_col])
    ax1.set_xlabel(met_col)
    ax1.set_ylabel(pol_col)
    ax1.set_title(f"Scatter (r = {corr:.2f})")

    # Threshold split
    ax2 = axes[1]
    thr = float(threshold) if threshold is not None else d[met_col].median()
    high = d[d[met_col] > thr][pol_col]
    low = d[d[met_col] <= thr][pol_col]
    means = [low.mean(), high.mean()]
    stds = [low.std(), high.std()]
    labels = [f"{met_col} ≤ {round(thr, 1)}", f"{met_col} > {round(thr, 1)}"]
    bars = ax2.bar(labels, means, yerr=stds, color=["steelblue", "indianred"], capsize=5, alpha=0.8)
    for bar, m, n in zip(bars, means, [len(low), len(high)]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{round(m, 1)}\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel(pol_col)
    ax2.set_title(f"Avg {pol_col} by {met_col} split")

    title = f"Impact of {met_col} on {pol_col}"
    if city:
        title += f" — {city}"
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return _save_fig(fig)


# ===================================================================
# map_* → plotly Figure (interactive map)
# ===================================================================

def _apply_india_boundary(fig):
    """Overlay India's official boundary (incl. J&K + Ladakh) on a mapbox figure."""
    if _INDIA_GEOJSON is None:
        return
    fig.update_layout(
        mapbox_style="white-bg",
        mapbox_layers=[{
            "source": _INDIA_GEOJSON,
            "type": "line",
            "color": "#444444",
            "line": {"width": 1.2},
            "below": "traces",
        }],
    )


def map_pollution(df, pollutant="PM2.5 (µg/m³)", year=None, month=None,
                  top_n=None, states=None, **_kw):
    """Bubble map of city-level average pollution across India."""
    col = _resolve_col(pollutant)
    d = _filter_df(df, year=year, month=month, states=states).dropna(subset=[col])
    if d.empty:
        return "No data available for the specified filters."

    # Compute year range for subtitle
    yr_min, yr_max = int(d["Year"].min()), int(d["Year"].max())

    agg = d.groupby("City")[col].agg(["mean", "count"]).reset_index()
    agg.columns = ["City", "mean", "count"]
    agg = agg.sort_values("mean", ascending=False)
    if top_n:
        agg = agg.head(int(top_n))

    agg = _merge_coords(agg)
    if agg.empty:
        return "No coordinates available for the matching cities."

    agg["mean"] = agg["mean"].round(2)
    unit = _col_unit(col)

    # Build title
    title_parts = [f"Average {col} by City"]
    if top_n:
        title_parts.insert(0, f"Top {top_n}")
    if states:
        state_label = ", ".join(states) if isinstance(states, list) else states
        title_parts.append(f"in {state_label}")

    # Time period
    if year and month:
        time_str = f"{_MONTH_NAMES.get(int(month), month)} {year}"
    elif year:
        time_str = str(year)
    elif month:
        time_str = f"{_MONTH_NAMES.get(int(month), month)}, {yr_min}–{yr_max}"
    else:
        time_str = f"{yr_min}–{yr_max}"
    title_parts.append(f"({time_str})")

    fig = px.scatter_mapbox(
        agg,
        lat="latitude",
        lon="longitude",
        size="mean",
        color="mean",
        hover_name="City",
        hover_data={"mean": True, "count": True, "latitude": False, "longitude": False},
        color_continuous_scale="YlOrRd",
        size_max=25,
        zoom=3.8,
        center={"lat": 22.5, "lon": 79.0},
        mapbox_style="open-street-map",
        title=" ".join(title_parts),
        labels={"mean": f"Avg {col}", "count": "Data Points"},
    )
    _apply_india_boundary(fig)
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=600)
    return fig


def map_change(df, pollutant="PM2.5 (µg/m³)", year_start=None, year_end=None,
               states=None, **_kw):
    """Map showing pollution change between two years per city (green=improved, red=worsened)."""
    col = _resolve_col(pollutant)
    d = _filter_df(df, states=states).dropna(subset=[col])
    if d.empty:
        return "No data available for the specified filters."

    years = sorted(d["Year"].unique())
    if len(years) < 2:
        return "Need at least 2 years of data to compute change."

    y1 = int(year_start) if year_start else years[0]
    y2 = int(year_end) if year_end else years[-1]

    d1 = d[d["Year"] == y1].groupby("City")[col].mean().rename("start")
    d2 = d[d["Year"] == y2].groupby("City")[col].mean().rename("end")
    change = pd.concat([d1, d2], axis=1).dropna()
    if change.empty:
        return f"No cities with data in both {y1} and {y2}."

    change["change"] = (change["end"] - change["start"]).round(2)
    change["pct_change"] = ((change["change"] / change["start"]) * 100).round(1)
    change["start"] = change["start"].round(2)
    change["end"] = change["end"].round(2)
    change = change.reset_index()

    change = _merge_coords(change)
    if change.empty:
        return "No coordinates available for the matching cities."

    unit = _col_unit(col)

    fig = px.scatter_mapbox(
        change,
        lat="latitude",
        lon="longitude",
        size=change["change"].abs(),
        color="change",
        hover_name="City",
        hover_data={
            "start": True, "end": True, "change": True,
            "pct_change": True, "latitude": False, "longitude": False,
        },
        color_continuous_scale="RdYlGn_r",
        color_continuous_midpoint=0,
        size_max=22,
        zoom=3.8,
        center={"lat": 22.5, "lon": 79.0},
        mapbox_style="open-street-map",
        title=f"{col} Change: {y1} → {y2}",
        labels={
            "change": f"Δ {unit}", "start": f"{y1} Avg",
            "end": f"{y2} Avg", "pct_change": "% Change",
        },
    )
    _apply_india_boundary(fig)
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=600)
    return fig


# ---------------------------------------------------------------------------
# Gemini Tool Declarations
# ---------------------------------------------------------------------------

_tool_declarations = [
    # --- get_* → TEXT ---
    genai.types.FunctionDeclaration(
        name="get_statistics",
        description="Returns a TEXT answer with statistical summaries. Use for 'which city/month/year has highest X', averages, single-value answers. Use whenever the user asks 'which', 'what', 'how much' and wants a short text answer (not a table or chart). 'pollution' or 'air quality' = PM2.5.",
        parameters=genai.types.Schema(
            type="OBJECT",
            properties={
                "pollutant": genai.types.Schema(type="STRING", description="Variable: PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, Ozone, temperature, humidity, wind speed, rainfall, solar radiation, pressure. Use 'PM2.5' for 'pollution' or 'air quality'."),
                "group_by": genai.types.Schema(type="STRING", description="Group by dimension", enum=["City", "State", "Station", "Year", "Month", "Season"]),
                "agg_func": genai.types.Schema(type="STRING", description="Aggregation function", enum=["mean", "max", "min", "median", "std"]),
                "year": genai.types.Schema(type="INTEGER", description="Optional year filter (e.g. 2023)"),
                "month": genai.types.Schema(type="INTEGER", description="Optional month filter (1-12)"),
                "cities": genai.types.Schema(type="ARRAY", items=genai.types.Schema(type="STRING"), description="Optional city filter (e.g. ['Mumbai'])"),
                "states": genai.types.Schema(type="ARRAY", items=genai.types.Schema(type="STRING"), description="Optional state filter (e.g. ['Maharashtra'])"),
                "top_n": genai.types.Schema(type="INTEGER", description="Return only top N results (e.g. 1 for 'which month has highest', 10 for 'top 10 cities')"),
                "sort_order": genai.types.Schema(type="STRING", description="Sort order", enum=["desc", "asc"]),
            },
            required=["pollutant"],
        ),
    ),
    genai.types.FunctionDeclaration(
        name="get_exceedances",
        description="Returns a TEXT list of cities/states where average pollutant exceeds or is below a threshold. Use for 'which cities exceed WHO guideline', 'cities above 60', 'states exceeding standard'. WHO PM2.5=15, India PM2.5=60, WHO PM10=50, India PM10=100. NEVER produces a plot.",
        parameters=genai.types.Schema(
            type="OBJECT",
            properties={
                "pollutant": genai.types.Schema(type="STRING", description="Variable: PM2.5, PM10, NO2, SO2, CO, Ozone, NH3, etc."),
                "threshold": genai.types.Schema(type="NUMBER", description="Threshold value. Use 15 for WHO PM2.5, 60 for India PM2.5, 50 for WHO PM10, 100 for India PM10."),
                "direction": genai.types.Schema(type="STRING", description="'above' or 'below'", enum=["above", "below"]),
                "group_by": genai.types.Schema(type="STRING", description="Group by", enum=["City", "State", "Station"]),
                "year": genai.types.Schema(type="INTEGER", description="Optional year filter"),
            },
            required=["pollutant", "threshold", "direction"],
        ),
    ),
    genai.types.FunctionDeclaration(
        name="get_ncap_analysis",
        description="Returns a TEXT answer about NCAP (National Clean Air Programme) funding and compliance. Use for any NCAP question that wants text (not a chart): 'which NCAP cities exceed guidelines', 'top funded cities', 'NCAP funding summary', 'how much funding did X receive', 'which NCAP cities reduced pollution', 'best PM2.5 reduction'. NEVER produces a plot.",
        parameters=genai.types.Schema(
            type="OBJECT",
            properties={
                "analysis_type": genai.types.Schema(type="STRING", description="'exceed_guidelines' for cities exceeding standards, 'funding_summary' for funding amounts, 'reduction' for change/reduction in pollution over time", enum=["exceed_guidelines", "funding_summary", "reduction"]),
                "pollutant": genai.types.Schema(type="STRING", description="Pollutant, e.g. 'PM2.5', 'PM10'"),
                "cities": genai.types.Schema(type="ARRAY", items=genai.types.Schema(type="STRING"), description="Optional city filter to show funding for specific cities (e.g. ['Delhi', 'Mumbai'])"),
            },
            required=["analysis_type"],
        ),
    ),
    # --- df_* → TABLE ---
    genai.types.FunctionDeclaration(
        name="df_ranking",
        description="Returns an interactive TABLE ranking cities/states/months/seasons by a metric. Use when user wants multiple rows of data: 'list', 'show table', 'rank all', 'top 10/20', 'ranking'. Also use for multi-row summaries like 'average X by state' or 'minimum X by year' when a table is appropriate. 'pollution' = PM2.5.",
        parameters=genai.types.Schema(
            type="OBJECT",
            properties={
                "pollutant": genai.types.Schema(type="STRING", description="Variable: PM2.5, PM10, NO2, SO2, CO, Ozone, NH3, temperature, humidity, wind speed, etc. Use 'PM2.5' for 'pollution'."),
                "group_by": genai.types.Schema(type="STRING", description="Group by", enum=["City", "State", "Station", "Year", "Month", "Season"]),
                "agg_func": genai.types.Schema(type="STRING", description="Aggregation function", enum=["mean", "max", "min", "median", "std"]),
                "year": genai.types.Schema(type="INTEGER", description="Optional year filter"),
                "month": genai.types.Schema(type="INTEGER", description="Optional month filter (1-12)"),
                "cities": genai.types.Schema(type="ARRAY", items=genai.types.Schema(type="STRING"), description="Optional city filter"),
                "states": genai.types.Schema(type="ARRAY", items=genai.types.Schema(type="STRING"), description="Optional state filter"),
                "top_n": genai.types.Schema(type="INTEGER", description="Number of rows (e.g. 10 for top 10)"),
                "sort_order": genai.types.Schema(type="STRING", description="Sort order", enum=["desc", "asc"]),
            },
            required=["pollutant", "group_by"],
        ),
    ),
    genai.types.FunctionDeclaration(
        name="df_exceedances",
        description="Returns an interactive TABLE of cities/states where pollutant exceeds or is below a threshold. Use for 'list cities exceeding', 'show table of areas above', 'table of cities exceeding WHO guideline'. WHO PM2.5=15, India PM2.5=60, WHO PM10=50, India PM10=100.",
        parameters=genai.types.Schema(
            type="OBJECT",
            properties={
                "pollutant": genai.types.Schema(type="STRING", description="Variable: PM2.5, PM10, NO2, SO2, CO, Ozone, etc."),
                "threshold": genai.types.Schema(type="NUMBER", description="Threshold value. Use 15 for WHO PM2.5, 60 for India PM2.5, 50 for WHO PM10, 100 for India PM10."),
                "direction": genai.types.Schema(type="STRING", description="'above' or 'below'", enum=["above", "below"]),
                "group_by": genai.types.Schema(type="STRING", description="Group by", enum=["City", "State", "Station"]),
                "year": genai.types.Schema(type="INTEGER", description="Optional year filter"),
            },
            required=["pollutant", "threshold", "direction"],
        ),
    ),
    # --- plot_* → CHART ---
    genai.types.FunctionDeclaration(
        name="plot_trend",
        description="Generates a LINE CHART showing how a variable changes over time (monthly or yearly). Use for 'plot trend', 'show trend', 'chart over the years', 'monthly trend', 'visualize trend'. Works with ALL pollutants (PM2.5, PM10, NO2, SO2, CO, NH3, Ozone) and met variables (temperature, wind speed, humidity, rainfall).",
        parameters=genai.types.Schema(
            type="OBJECT",
            properties={
                "pollutant": genai.types.Schema(type="STRING", description="Variable: PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, Ozone, temperature, humidity, wind speed, rainfall. Use 'PM2.5' for 'pollution'."),
                "cities": genai.types.Schema(type="ARRAY", items=genai.types.Schema(type="STRING"), description="Optional city names"),
                "states": genai.types.Schema(type="ARRAY", items=genai.types.Schema(type="STRING"), description="Optional state names"),
                "freq": genai.types.Schema(type="STRING", description="'monthly' or 'yearly'", enum=["monthly", "yearly"]),
                "year_start": genai.types.Schema(type="INTEGER", description="Start year"),
                "year_end": genai.types.Schema(type="INTEGER", description="End year"),
            },
            required=["pollutant"],
        ),
    ),
    genai.types.FunctionDeclaration(
        name="plot_comparison",
        description="Generates a BOX PLOT comparing a variable across groups. Use for 'compare across seasons', 'seasonal comparison', 'weekday vs weekend', 'compare cities', 'north vs south'. 'pollution' = PM2.5.",
        parameters=genai.types.Schema(
            type="OBJECT",
            properties={
                "pollutant": genai.types.Schema(type="STRING", description="Variable: PM2.5, PM10, NO2, SO2, CO, Ozone, NH3, temperature, etc. Use 'PM2.5' for 'pollution'."),
                "comparison_type": genai.types.Schema(type="STRING", description="Comparison type", enum=["seasonal", "weekday_weekend", "cities", "states", "north_south"]),
                "cities": genai.types.Schema(type="ARRAY", items=genai.types.Schema(type="STRING"), description="City names (for 'cities' comparison)"),
                "states": genai.types.Schema(type="ARRAY", items=genai.types.Schema(type="STRING"), description="State names (for 'states' comparison)"),
                "year": genai.types.Schema(type="INTEGER", description="Optional year filter"),
            },
            required=["pollutant", "comparison_type"],
        ),
    ),
    genai.types.FunctionDeclaration(
        name="plot_correlation",
        description="Generates a SCATTER PLOT with regression line showing relationship between two variables. Use when user asks for 'X vs Y', 'correlation between X and Y', 'scatter plot', 'relationship between', 'PM2.5 vs PM10', 'temperature vs pollution'. Works with any pair of pollutants or met variables.",
        parameters=genai.types.Schema(
            type="OBJECT",
            properties={
                "var_x": genai.types.Schema(type="STRING", description="X-axis variable: PM2.5, PM10, NO2, SO2, CO, Ozone, temperature, humidity, wind speed, rainfall, etc."),
                "var_y": genai.types.Schema(type="STRING", description="Y-axis variable: PM2.5, PM10, NO2, SO2, CO, Ozone, temperature, humidity, wind speed, rainfall, etc."),
                "city": genai.types.Schema(type="STRING", description="Optional city filter"),
                "state": genai.types.Schema(type="STRING", description="Optional state filter"),
                "year": genai.types.Schema(type="INTEGER", description="Optional year filter"),
            },
            required=["var_x", "var_y"],
        ),
    ),
    genai.types.FunctionDeclaration(
        name="plot_ncap",
        description="Generates a CHART about NCAP (National Clean Air Programme). Use for 'plot NCAP funding', 'chart NCAP vs non-NCAP', 'visualize NCAP', 'graph NCAP funding distribution'. Use funding_comparison for funding amounts, reduction_analysis for NCAP vs non-NCAP pollution trend over years.",
        parameters=genai.types.Schema(
            type="OBJECT",
            properties={
                "analysis_type": genai.types.Schema(type="STRING", description="'funding_comparison' for funding bar chart, 'reduction_analysis' for NCAP vs non-NCAP yearly trend", enum=["funding_comparison", "reduction_analysis"]),
                "pollutant": genai.types.Schema(type="STRING", description="Pollutant, e.g. 'PM2.5'"),
                "cities": genai.types.Schema(type="ARRAY", items=genai.types.Schema(type="STRING"), description="Optional city filter to show funding for specific cities (e.g. ['Delhi', 'Mumbai'])"),
            },
            required=["analysis_type"],
        ),
    ),
    genai.types.FunctionDeclaration(
        name="plot_met_impact",
        description="Generates a dual-panel CHART showing how a meteorological factor affects a pollutant. Use when user asks 'does wind affect PM2.5', 'impact of temperature on pollution', 'how does humidity affect', 'effect of rainfall on', 'does wind reduce pollution'. Shows scatter + threshold split.",
        parameters=genai.types.Schema(
            type="OBJECT",
            properties={
                "met_factor": genai.types.Schema(type="STRING", description="Met variable: temperature, humidity, wind speed, rainfall, wind direction, solar radiation, pressure"),
                "pollutant": genai.types.Schema(type="STRING", description="Pollutant: PM2.5, PM10, NO2, SO2, CO, Ozone, etc. Use 'PM2.5' for 'pollution'."),
                "city": genai.types.Schema(type="STRING", description="Optional city filter"),
                "threshold": genai.types.Schema(type="NUMBER", description="Optional split threshold"),
            },
            required=["met_factor", "pollutant"],
        ),
    ),
    # --- map_* → INTERACTIVE MAP ---
    genai.types.FunctionDeclaration(
        name="map_pollution",
        description="Generates an INTERACTIVE MAP showing city-level pollution as colored bubbles on a map of India. Use when user asks 'show on a map', 'map of pollution', 'pollution map', 'geographic distribution', 'spatial distribution', 'where is pollution highest on map'. Works with any pollutant: PM2.5, PM10, NO2, SO2, CO, Ozone, NH3.",
        parameters=genai.types.Schema(
            type="OBJECT",
            properties={
                "pollutant": genai.types.Schema(type="STRING", description="Variable: PM2.5, PM10, NO2, SO2, CO, Ozone, NH3, etc. Use 'PM2.5' for 'pollution' or 'air quality'."),
                "year": genai.types.Schema(type="INTEGER", description="Optional year filter (e.g. 2023)"),
                "month": genai.types.Schema(type="INTEGER", description="Optional month filter (1-12)"),
                "top_n": genai.types.Schema(type="INTEGER", description="Show only top N most polluted cities (e.g. 20)"),
                "states": genai.types.Schema(type="ARRAY", items=genai.types.Schema(type="STRING"), description="Optional state filter"),
            },
            required=["pollutant"],
        ),
    ),
    genai.types.FunctionDeclaration(
        name="map_change",
        description="Generates an INTERACTIVE MAP showing pollution CHANGE between two years per city. Green bubbles = improved, red = worsened. Use when user asks 'map pollution change', 'map improvement', 'which cities improved on map', 'geographic change in pollution'.",
        parameters=genai.types.Schema(
            type="OBJECT",
            properties={
                "pollutant": genai.types.Schema(type="STRING", description="Variable: PM2.5, PM10, NO2, SO2, CO, Ozone, etc. Use 'PM2.5' for 'pollution'."),
                "year_start": genai.types.Schema(type="INTEGER", description="Start year for comparison (defaults to earliest year)"),
                "year_end": genai.types.Schema(type="INTEGER", description="End year for comparison (defaults to latest year)"),
                "states": genai.types.Schema(type="ARRAY", items=genai.types.Schema(type="STRING"), description="Optional state filter"),
            },
            required=["pollutant"],
        ),
    ),
]

analysis_tools = genai.types.Tool(function_declarations=_tool_declarations)

FC_SYSTEM_PROMPT = (
    "You are an air quality data analyst. ALWAYS use one of the provided tools to answer "
    "questions about Indian air quality data. Data covers daily measurements from 2017–2024 "
    "across Indian cities.\n\n"
    "Available pollutants: PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, Ozone.\n"
    "Available met variables: temperature, humidity, wind speed, wind direction, rainfall, "
    "solar radiation, barometric pressure.\n"
    "The word 'pollution' or 'air quality' without a specific pollutant means PM2.5.\n\n"
    "RULES:\n"
    "1. ALWAYS call a tool. Only respond with plain text if the question is completely "
    "unrelated to air quality, pollution, or weather data.\n"
    "2. Pick the tool based on desired OUTPUT TYPE:\n"
    "   - get_*  → short TEXT answer (single values, summaries, 'which city has highest')\n"
    "   - df_*   → interactive TABLE (multiple rows: 'list', 'rank', 'show table', 'top N')\n"
    "   - plot_* → CHART image (any plot/chart/graph/visualization/trend/comparison request)\n"
    "   - map_*  → INTERACTIVE MAP (any request mentioning 'map', 'geographic', 'spatial', 'show on map')\n"
    "3. For WHO PM2.5 guideline use threshold=15. For India PM2.5 standard use threshold=60.\n"
    "   For WHO PM10 guideline use threshold=50. For India PM10 standard use threshold=100.\n"
    "4. When user says 'X vs Y' or 'correlation' or 'scatter' or 'relationship between', "
    "use plot_correlation.\n"
    "5. When user asks how a weather factor 'affects' or 'impacts' pollution, use plot_met_impact.\n"
)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_FUNCTION_MAP = {
    "get_statistics": get_statistics,
    "get_exceedances": get_exceedances,
    "get_ncap_analysis": get_ncap_analysis,
    "df_ranking": df_ranking,
    "df_exceedances": df_exceedances,
    "plot_trend": plot_trend,
    "plot_comparison": plot_comparison,
    "plot_correlation": plot_correlation,
    "plot_ncap": plot_ncap,
    "plot_met_impact": plot_met_impact,
    "map_pollution": map_pollution,
    "map_change": map_change,
}

_NEEDS_NCAP = {"get_ncap_analysis", "plot_ncap"}


def dispatch(function_call, df, ncap_df):
    """Dispatch a Gemini function_call to the matching local function."""
    name = function_call.name
    args = dict(function_call.args) if function_call.args else {}
    func = _FUNCTION_MAP.get(name)
    if func is None:
        return f"Unknown function: {name}"

    if name in _NEEDS_NCAP:
        return func(df=df, ncap_df=ncap_df, **args)
    return func(df=df, **args)
