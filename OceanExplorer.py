"""
ocean_explorer.py
─────────────────
CS-MACH1 — Ocean Temperature Climate Explorer

Layout
──────
┌─────────────────────┬─────────────────────┐
│ CORA monthly        │ CORA DOY             │
│ mean ± std          │ interannual scatter  │
├─────────────────────┼─────────────────────┤
│ WOD T–depth scatter │ CORA T–depth profile │
│  (reactive to depth)│  (reactive to depth) │
└─────────────────────┴─────────────────────┘

Reactivity
──────────
• "Run Analysis" fetches surface CORA + WOD raw profiles (cached by lat/lon).
• Changing the depth slider re-clips the cached WOD data and re-fetches the
  CORA depth profile (cached by lat/lon/depth) — no full re-run needed.

Dependencies:
    streamlit folium streamlit-folium requests pandas matplotlib numpy beacon-api
"""

from __future__ import annotations

import io
import warnings
from datetime import datetime

import folium
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


# ── Page config & branding ────────────────────────────────────────────────────

st.set_page_config(
    page_title="CS-MACH1 Ocean Temperature Climate Explorer",
    page_icon="🌊",
    layout="wide",
)

st.markdown("""
<style>
.main-title  { font-size:2rem; font-weight:800; color:#00A6D6; letter-spacing:-0.5px; }
.sub-title   { font-size:1rem; color:#555; margin-bottom:1rem; }
.section-hdr { font-size:1.2rem; font-weight:700; color:#00A6D6;
               border-bottom:2px solid #00A6D6; padding-bottom:4px;
               margin-top:1.4rem; margin-bottom:.6rem; }
.stButton>button { background-color:#00A6D6; color:white;
                   border-radius:8px; border:none; font-weight:600; }
.stButton>button:hover { background-color:#007EA3; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🌊 CS-MACH1 — Ocean Temperature Climate Explorer</div>",
            unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>"
    "Click a point on the map (or type coordinates) → set max depth → Run Analysis"
    "</div>",
    unsafe_allow_html=True,
)


# ── Constants ─────────────────────────────────────────────────────────────────

CORA_SURFACE_URL = (
    "https://erddap.emodnet-physics.eu/erddap/griddap/"
    "INSITU_GLO_PHY_TS_OA_MY_013_052_TEMP.csv"
    "?TEMP%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D"
    "%5B(1.0):1:(1)%5D"
    "%5B({lat}):1:({lat})%5D"
    "%5B({lon}):1:({lon})%5D"
)

CORA_DEPTH_URL = (
    "https://erddap.emodnet-physics.eu/erddap/griddap/"
    "INSITU_GLO_PHY_TS_OA_MY_013_052_TEMP.csv"
    "?TEMP%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D"
    "%5B(1.0):1:({depth})%5D"
    "%5B({lat}):1:({lat})%5D"
    "%5B({lon}):1:({lon})%5D"
)

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

DEFAULT_LAT, DEFAULT_LON = 44.38, 9.07


# ── Data fetchers ─────────────────────────────────────────────────────────────

def _wod_client():
    try:
        from beacon_api import Client
        return Client("https://beacon-wod.maris.nl",
                        proxy_headers={"User-Agent": "my-app/1.0 (antonio.novellino@dedagroup.it)"}
                       )
        #return Client("https://beacon-wod.maris.nl")
      
    except ImportError as exc:
        raise ImportError("Run: pip install beacon-api") from exc


# FIX: normalise ERDDAP column names (strips spaces, finds TEMP regardless of case)
def _normalize_cora_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]

    temp_col = next(
        (c for c in df.columns
         if c.strip().upper() in ("TEMP", "TEMPERATURE", "SEA_WATER_TEMPERATURE",
                                  "THETAO", "POTENTIAL_TEMPERATURE")),
        None,
    )
    if temp_col and temp_col != "TEMP":
        df = df.rename(columns={temp_col: "TEMP"})

    time_col = next(
        (c for c in df.columns if c.strip().lower() in ("time", "datetime", "date_time")),
        None,
    )
    if time_col and time_col != "time":
        df = df.rename(columns={time_col: "time"})

    depth_col = next(
        (c for c in df.columns if c.strip().lower() in ("depth", "z", "depth_m")),
        None,
    )
    if depth_col and depth_col != "depth":
        df = df.rename(columns={depth_col: "depth"})

    return df


@st.cache_data(show_spinner="Querying World Ocean Database…", ttl=3600)
def fetch_wod_all(latitude: float, longitude: float) -> pd.DataFrame | None:
    try:
        client  = _wod_client()
        lat_min = round(latitude,  1) - 0.1
        lat_max = round(latitude,  1) + 0.1
        lon_min = round(longitude, 1) - 0.1
        lon_max = round(longitude, 1) + 0.1

        qb = client.query()
        qb.add_select_column("wod_unique_cast")
        qb.add_select_column("Temperature",         alias="TEMPERATURE")
        qb.add_select_column("Temperature_WODflag", alias="TEMPERATURE_QC")
        qb.add_select_column("z",                   alias="DEPTH")
        qb.add_select_column("time",                alias="TIME")
        qb.add_select_column("lon",                 alias="LONGITUDE")
        qb.add_select_column("lat",                 alias="LATITUDE")

        qb.add_range_filter("TIME",        "1970-01-01T00:00:00", "2023-01-01T00:00:00")
        qb.add_is_not_null_filter("TEMPERATURE")
        qb.add_not_equals_filter("TEMPERATURE", -1e10)
        qb.add_equals_filter("TEMPERATURE_QC", 0.0)
        qb.add_range_filter("DEPTH",       0, 10_000)
        qb.add_range_filter("LONGITUDE",   lon_min, lon_max)
        qb.add_range_filter("LATITUDE",    lat_min, lat_max)

        raw = qb.to_pandas_dataframe()
        raw["TEMPERATURE"] = pd.to_numeric(raw["TEMPERATURE"], errors="coerce").round(3)
        raw["DEPTH"]       = pd.to_numeric(raw["DEPTH"],       errors="coerce")
        raw["TIME"]        = pd.to_datetime(raw["TIME"],        errors="coerce")
        return raw.dropna(subset=["DEPTH", "TEMPERATURE"])
    except Exception as exc:
        st.warning(f"WOD query failed: {exc}")
        return None


@st.cache_data(show_spinner="Downloading CORA surface climatology…", ttl=86400)
def fetch_cora_surface(latitude: float, longitude: float) -> pd.DataFrame | None:
    url = CORA_SURFACE_URL.format(lat=round(latitude, 4), lon=round(longitude, 4))
    try:
        r = requests.get(url, verify=False, timeout=60)
        r.raise_for_status()
        if "<html" in r.text.lower():
            raise ValueError("CORA returned an HTML error page.")
        df = pd.read_csv(io.StringIO(r.text), skiprows=[1])
        df = _normalize_cora_df(df)
        if "TEMP" not in df.columns:
            raise KeyError(f"No temperature column found. Columns: {list(df.columns)}")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["TEMP"] = pd.to_numeric(df["TEMP"], errors="coerce").round(3)
        return df.dropna(subset=["time", "TEMP"])
    except Exception as exc:
        st.warning(f"CORA surface fetch failed: {exc}")
        return None


@st.cache_data(show_spinner="Downloading CORA depth profile…", ttl=86400)
def fetch_cora_depth_profile(latitude: float, longitude: float,
                              max_depth: float) -> pd.DataFrame | None:
    url = CORA_DEPTH_URL.format(
        lat=round(latitude, 4),
        lon=round(longitude, 4),
        depth=float(max_depth),
    )
    try:
        r = requests.get(url, verify=False, timeout=90)
        r.raise_for_status()
        if "<html" in r.text.lower():
            raise ValueError("CORA returned an HTML error page.")
        df = pd.read_csv(io.StringIO(r.text), skiprows=[1])
        df = _normalize_cora_df(df)
        if "TEMP" not in df.columns:
            raise KeyError(f"No temperature column found. Columns: {list(df.columns)}")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["TEMP"] = pd.to_numeric(df["TEMP"], errors="coerce").round(3)
        if "depth" in df.columns:
            df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
        return df.dropna(subset=["time", "TEMP"])
    except Exception as exc:
        st.warning(f"CORA depth profile fetch failed: {exc}")
        return None


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_cora_monthly(cora: pd.DataFrame,
                      latitude: float, longitude: float) -> plt.Figure:
    cora      = cora.copy()
    cora["m"] = cora["time"].dt.month
    monthly   = cora.groupby("m")["TEMP"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(monthly["m"],
                    monthly["mean"] - monthly["std"],
                    monthly["mean"] + monthly["std"],
                    alpha=0.2, color="steelblue", label="± 1 std")
    ax.plot(monthly["m"], monthly["mean"], "o-",
            color="steelblue", lw=2, ms=6, label="Monthly mean")
    ax.plot(monthly["m"],
            monthly["mean"].rolling(3, center=True).mean(),
            "--", color="navy", lw=1.2, alpha=0.6, label="3-month smooth")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        f"CORA Monthly Mean ± Std (surface)\n"
        f"({latitude:.4f}°N, {longitude:.4f}°E) · 1990–2023", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cora_doy(cora: pd.DataFrame,
                  latitude: float, longitude: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    years   = sorted(cora["time"].dt.year.unique())
    colours = cm.viridis(np.linspace(0, 1, len(years)))

    for colour, (_, ydata) in zip(colours, cora.groupby(cora["time"].dt.year)):
        ax.scatter(ydata["time"].dt.dayofyear, ydata["TEMP"],
                   s=8, color=colour, alpha=0.55)

    cora2        = cora.copy()
    cora2["doy"] = cora2["time"].dt.dayofyear
    doy_med      = cora2.groupby("doy")["TEMP"].median()
    ax.plot(doy_med.index, doy_med.values,
            color="crimson", lw=2, zorder=5, label="Daily median")

    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(vmin=min(years), vmax=max(years)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, label="Year")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        f"CORA Interannual Temperature Variability (surface)\n"
        f"({latitude:.4f}°N, {longitude:.4f}°E)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_wod_monthly(wod: pd.DataFrame,
                     latitude: float, longitude: float) -> plt.Figure:
    surf = wod[wod["DEPTH"] <= 10].copy()
    surf["m"] = pd.to_datetime(surf["TIME"], errors="coerce").dt.month
    monthly   = surf.groupby("m")["TEMPERATURE"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    if monthly.empty:
        ax.text(0.5, 0.5, "No surface WOD data (depth ≤ 10 m)",
                ha="center", va="center", transform=ax.transAxes, color="grey")
    else:
        ax.fill_between(monthly["m"],
                        monthly["mean"] - monthly["std"],
                        monthly["mean"] + monthly["std"],
                        alpha=0.2, color="seagreen", label="± 1 std")
        ax.plot(monthly["m"], monthly["mean"], "o-",
                color="seagreen", lw=2, ms=6, label="Monthly mean")
        ax.plot(monthly["m"],
                monthly["mean"].rolling(3, center=True).mean(),
                "--", color="darkgreen", lw=1.2, alpha=0.6, label="3-month smooth")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        f"WOD Monthly Mean ± Std (depth ≤ 10 m)\n"
        f"({latitude:.4f}°N, {longitude:.4f}°E) · 1970–2023", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_wod_doy(wod: pd.DataFrame,
                 latitude: float, longitude: float) -> plt.Figure:
    surf         = wod[wod["DEPTH"] <= 10].copy()
    surf["time"] = pd.to_datetime(surf["TIME"], errors="coerce")
    surf         = surf.dropna(subset=["time"])

    fig, ax = plt.subplots(figsize=(8, 5))
    if surf.empty:
        ax.text(0.5, 0.5, "No surface WOD data (depth ≤ 10 m)",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title("WOD Interannual Temperature Variability (surface)", fontsize=10)
        fig.tight_layout()
        return fig

    years   = sorted(surf["time"].dt.year.unique())
    colours = cm.plasma(np.linspace(0, 1, len(years)))
    for colour, (_, ydata) in zip(colours, surf.groupby(surf["time"].dt.year)):
        ax.scatter(ydata["time"].dt.dayofyear, ydata["TEMPERATURE"],
                   s=8, color=colour, alpha=0.55)

    surf["doy"] = surf["time"].dt.dayofyear
    doy_med     = surf.groupby("doy")["TEMPERATURE"].median()
    ax.plot(doy_med.index, doy_med.values,
            color="crimson", lw=2, zorder=5, label="Daily median")

    sm = plt.cm.ScalarMappable(cmap="plasma",
                                norm=plt.Normalize(vmin=min(years), vmax=max(years)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, label="Year")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        f"WOD Interannual Temperature Variability (depth ≤ 10 m)\n"
        f"({latitude:.4f}°N, {longitude:.4f}°E)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_wod_scatter(raw_full: pd.DataFrame, max_depth: float,
                     latitude: float, longitude: float) -> plt.Figure:
    raw = raw_full[raw_full["DEPTH"] <= max_depth].copy()
    fig, ax = plt.subplots(figsize=(6, 8))
    MAX_PTS = 8_000
    plot_df = raw.sample(min(MAX_PTS, len(raw)), random_state=42) if len(raw) > 0 else raw

    if not plot_df.empty:
        sc = ax.scatter(plot_df["TEMPERATURE"], plot_df["DEPTH"],
                        c=plot_df["DEPTH"], cmap="Blues_r",
                        s=5, alpha=0.4, vmin=0, vmax=max_depth)
        fig.colorbar(sc, ax=ax, label="Depth (m)", pad=0.02)
    else:
        ax.text(0.5, 0.5, "No data in range", ha="center", va="center",
                transform=ax.transAxes, color="grey")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_ylim(bottom=max_depth, top=0)
    ax.set_title(
        f"WOD T–Depth Observations\n({latitude:.4f}°N, {longitude:.4f}°E)\n"
        f"n = {len(raw):,} · 0 – {max_depth:.0f} m", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cora_depth_profile(cora_dp: pd.DataFrame, max_depth: float,
                             latitude: float, longitude: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 8))
    depth_col = "depth" if "depth" in cora_dp.columns else None

    if depth_col is None or cora_dp.empty:
        ax.text(0.5, 0.5, "CORA depth data not available",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title("CORA T–Depth Profile", fontsize=10)
        fig.tight_layout()
        return fig

    profile = (cora_dp.groupby(depth_col)["TEMP"]
               .agg(["mean", "std", "median"]).reset_index().sort_values(depth_col))

    ax.fill_betweenx(profile[depth_col],
                     profile["mean"] - profile["std"],
                     profile["mean"] + profile["std"],
                     alpha=0.18, color="steelblue", label="± 1 std")
    ax.plot(profile["mean"] - profile["std"], profile[depth_col],
            "--", color="royalblue", lw=1.2, alpha=0.7, label="Mean − std")
    ax.plot(profile["mean"] + profile["std"], profile[depth_col],
            "--", color="tomato", lw=1.2, alpha=0.7, label="Mean + std")
    ax.plot(profile["mean"],   profile[depth_col],
            "-",  color="steelblue", lw=2.5, label="Mean")
    ax.plot(profile["median"], profile[depth_col],
            "-",  color="darkorange", lw=1.8, ls=":", label="Median")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_ylim(bottom=max_depth, top=0)
    ax.set_title(
        f"CORA T–Depth Profile\n({latitude:.4f}°N, {longitude:.4f}°E)\n"
        f"0 – {max_depth:.0f} m · 1990–2023", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📍 Location")

    # FIX: dynamic key forces widget recreation when map is clicked,
    # so the number_input always shows the updated coordinates.
    _lat_key = f"lat_input_{st.session_state.get('sel_lat', DEFAULT_LAT)}"
    _lon_key = f"lon_input_{st.session_state.get('sel_lon', DEFAULT_LON)}"

    lat_in = st.number_input(
        "Latitude (°N)",  min_value=-90.0, max_value=90.0,
        value=st.session_state.get("sel_lat", DEFAULT_LAT),
        step=0.01, format="%.4f",
        key=_lat_key,
    )
    lon_in = st.number_input(
        "Longitude (°E)", min_value=-180.0, max_value=180.0,
        value=st.session_state.get("sel_lon", DEFAULT_LON),
        step=0.01, format="%.4f",
        key=_lon_key,
    )

    st.divider()
    st.markdown("### ⚙️ Parameters")

    max_depth = st.slider(
        "Max depth (m)", min_value=10, max_value=5000,
        value=st.session_state.get("last_depth", 200),
        step=10, key="depth_slider",
    )

    st.divider()
    run_btn = st.button("▶️ Run Analysis", type="primary", use_container_width=True)

    if st.button("🧹 Reset", use_container_width=True):
        for k in ["sel_lat", "sel_lon", "results", "last_depth"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()
    st.caption(
        "Data sources\n"
        "• **CORA**: EMODnet-Physics ERDDAP (1990–2023)\n"
        "• **WOD**: Beacon API / MARIS (1970–2023)\n"
        "• WOD search box: ±0.1° around selected point\n\n"
        "**Depth slider** reactively updates\n"
        "the WOD scatter and CORA depth profile\n"
        "without re-running the full analysis."
    )


# ── Map ───────────────────────────────────────────────────────────────────────

st.markdown("<div class='section-hdr'>🗺️ Select a Point on the Map</div>",
            unsafe_allow_html=True)
st.caption(
    "Click anywhere on the ocean to set the analysis location, "
    "or type coordinates directly in the sidebar."
)

center_lat = st.session_state.get("sel_lat", DEFAULT_LAT)
center_lon = st.session_state.get("sel_lon", DEFAULT_LON)

m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles=None)

folium.TileLayer(
    tiles="CartoDB positron", name="CartoDB Positron",
    overlay=False, control=True, show=True,
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Esri Ocean", overlay=False, control=True,
).add_to(m)

folium.TileLayer(
    tiles="https://tiles.emodnet-bathymetry.eu/wmts/1.0.0/mean_multicolour/default/web_mercator/{z}/{y}/{x}.png",
    attr='&copy; <a href="https://www.emodnet-bathymetry.eu/" target="_blank">EMODnet Bathymetry</a>',
    name="EMODnet Bathymetry (mean depth)",
    overlay=False, control=True, show=False, opacity=0.85,
).add_to(m)

folium.TileLayer(
    tiles="https://tiles.emodnet-bathymetry.eu/wmts/1.0.0/mean_rainbowcolour/default/web_mercator/{z}/{y}/{x}.png",
    attr='&copy; <a href="https://www.emodnet-bathymetry.eu/" target="_blank">EMODnet Bathymetry</a>',
    name="EMODnet Bathymetry (rainbow)",
    overlay=False, control=True, show=False, opacity=0.85,
).add_to(m)

folium.WmsTileLayer(
    url="https://ows.emodnet-bathymetry.eu/wms",
    layers="emodnet:contours", fmt="image/png", transparent=True, version="1.3.0",
    attr='&copy; <a href="https://www.emodnet-bathymetry.eu/" target="_blank">EMODnet Bathymetry contours</a>',
    name="Bathymetric contours (EMODnet)",
    overlay=True, control=True, show=True, opacity=0.7,
).add_to(m)

folium.WmsTileLayer(
    url="https://ows.emodnet-bathymetry.eu/wms",
    layers="emodnet:mean_multicolour", fmt="image/png", transparent=True, version="1.3.0",
    attr='&copy; <a href="https://www.emodnet-bathymetry.eu/" target="_blank">EMODnet Bathymetry DTM</a>',
    name="Mean depth DTM (EMODnet WMS)",
    overlay=True, control=True, show=False, opacity=0.6,
).add_to(m)

folium.Marker(
    location=[center_lat, center_lon],
    tooltip=f"Selected: {center_lat:.4f}°N, {center_lon:.4f}°E",
    icon=folium.Icon(color="blue", icon="tint", prefix="fa"),
).add_to(m)

folium.Rectangle(
    bounds=[[center_lat - 0.1, center_lon - 0.1],
            [center_lat + 0.1, center_lon + 0.1]],
    color="#00A6D6", weight=1.5, fill=True, fill_opacity=0.08,
    tooltip="WOD search box (±0.1°)",
).add_to(m)

folium.LayerControl().add_to(m)

# FIX: use_container_width=True instead of width="100%" (string not accepted)
map_result = st_folium(m, use_container_width=True, height=420,
                       returned_objects=["last_clicked"])

if map_result and map_result.get("last_clicked"):
    clicked = map_result["last_clicked"]
    new_lat = round(clicked["lat"], 4)
    new_lon = round(clicked["lng"], 4)
    if (new_lat != st.session_state.get("sel_lat")
            or new_lon != st.session_state.get("sel_lon")):
        st.session_state["sel_lat"] = new_lat
        st.session_state["sel_lon"] = new_lon
        st.rerun()

# FIX: always read from session_state so map clicks propagate to analysis
latitude  = st.session_state.get("sel_lat", DEFAULT_LAT)
longitude = st.session_state.get("sel_lon", DEFAULT_LON)

st.info(
    f"📍 **Analysis point:** {latitude:.4f}°N, {longitude:.4f}°E  "
    f"· Max depth: **{max_depth} m**"
)


# ── Initial run ───────────────────────────────────────────────────────────────

if run_btn:
    st.session_state.pop("results", None)

    pbar = st.progress(0, text="Fetching CORA surface data…")
    cora_surf = fetch_cora_surface(latitude, longitude)

    pbar.progress(35, text="Querying WOD (full water column)…")
    wod_raw = fetch_wod_all(latitude, longitude)

    pbar.progress(70, text="Fetching CORA depth profile…")
    cora_dp = fetch_cora_depth_profile(latitude, longitude, float(max_depth))

    pbar.progress(100, text="✅ Done!")

    st.session_state["results"] = {
        "cora_surf": cora_surf,
        "wod_raw":   wod_raw,
        "cora_dp":   cora_dp,
        "lat":  latitude,
        "lon":  longitude,
        "ts":   datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state["last_depth"] = max_depth


# ── Reactive depth update ─────────────────────────────────────────────────────

if (
    "results" in st.session_state
    and st.session_state.get("last_depth") != max_depth
):
    res = st.session_state["results"]
    with st.spinner(f"Updating depth profiles to {max_depth} m…"):
        res["cora_dp"] = fetch_cora_depth_profile(
            res["lat"], res["lon"], float(max_depth)
        )
    st.session_state["results"]    = res
    st.session_state["last_depth"] = max_depth


# ── Display ───────────────────────────────────────────────────────────────────

if "results" in st.session_state:
    res       = st.session_state["results"]
    cora_surf = res["cora_surf"]
    wod_raw   = res["wod_raw"]
    cora_dp   = res["cora_dp"]
    rlat      = res["lat"]
    rlon      = res["lon"]

    st.markdown(
        f"<div class='section-hdr'>📊 Results — "
        f"{rlat:.4f}°N, {rlon:.4f}°E · max {max_depth} m · {res['ts']}</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    if cora_surf is not None:
        c1.metric("CORA records", f"{len(cora_surf):,}")
        c2.metric("CORA period",
                  f"{cora_surf['time'].dt.year.min()}–{cora_surf['time'].dt.year.max()}")
    if wod_raw is not None and not wod_raw.empty:
        wod_clipped = wod_raw[wod_raw["DEPTH"] <= max_depth]
        c3.metric("WOD obs (clipped)", f"{len(wod_clipped):,}")
        c4.metric("WOD depth range",
                  f"{wod_clipped['DEPTH'].min():.0f}–{wod_clipped['DEPTH'].max():.0f} m")

    st.caption(
        "ℹ️ Move the **Max depth** slider in the sidebar to update the bottom row "
        "without re-running the full analysis."
    )

    fig, axes = plt.subplots(3, 2, figsize=(18, 18),
                              gridspec_kw={"hspace": 0.42, "wspace": 0.30})
    ax_cm, ax_cd = axes[0, 0], axes[0, 1]
    ax_wm, ax_wd = axes[1, 0], axes[1, 1]
    ax_ws, ax_cp = axes[2, 0], axes[2, 1]

    def _blank(ax: plt.Axes, msg: str) -> None:
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, color="grey", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # ── Row 1 — CORA surface ──────────────────────────────────────────────────
    if cora_surf is not None and not cora_surf.empty:
        cs      = cora_surf.copy()
        cs["m"] = cs["time"].dt.month
        cmon    = cs.groupby("m")["TEMP"].agg(["mean", "std"]).reset_index()

        ax_cm.fill_between(cmon["m"],
                           cmon["mean"] - cmon["std"], cmon["mean"] + cmon["std"],
                           alpha=0.2, color="steelblue", label="± 1 std")
        ax_cm.plot(cmon["m"], cmon["mean"], "o-",
                   color="steelblue", lw=2, ms=5, label="Monthly mean")
        ax_cm.set_xticks(range(1, 13))
        ax_cm.set_xticklabels(MONTH_LABELS, fontsize=7)
        ax_cm.set_xlabel("Month")
        ax_cm.set_ylabel("Temperature (°C)")
        ax_cm.set_title(
            f"CORA Monthly Mean ± Std (surface)\n"
            f"({rlat:.4f}°N, {rlon:.4f}°E) · 1990–2023", fontsize=9)
        ax_cm.legend(fontsize=7)
        ax_cm.grid(True, alpha=0.3)

        years_c   = sorted(cora_surf["time"].dt.year.unique())
        colours_c = cm.viridis(np.linspace(0, 1, len(years_c)))
        for col_c, (_, ydata) in zip(colours_c, cora_surf.groupby(cora_surf["time"].dt.year)):
            ax_cd.scatter(ydata["time"].dt.dayofyear, ydata["TEMP"],
                          s=6, color=col_c, alpha=0.5)
        cs2        = cora_surf.copy()
        cs2["doy"] = cs2["time"].dt.dayofyear
        doy_med_c  = cs2.groupby("doy")["TEMP"].median()
        ax_cd.plot(doy_med_c.index, doy_med_c.values,
                   color="crimson", lw=2, zorder=5, label="Daily median")
        sm_c = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(vmin=min(years_c), vmax=max(years_c)))
        sm_c.set_array([])
        fig.colorbar(sm_c, ax=ax_cd, pad=0.02, label="Year")
        ax_cd.set_xlabel("Day of Year")
        ax_cd.set_ylabel("Temperature (°C)")
        ax_cd.set_title(
            f"CORA Interannual Variability (surface)\n"
            f"({rlat:.4f}°N, {rlon:.4f}°E)", fontsize=9)
        ax_cd.legend(fontsize=7)
        ax_cd.grid(True, alpha=0.3)
    else:
        _blank(ax_cm, "CORA surface data not available")
        _blank(ax_cd, "CORA surface data not available")

    # ── Row 2 — WOD surface (depth ≤ 10 m) ───────────────────────────────────
    if wod_raw is not None and not wod_raw.empty:
        surf_wod         = wod_raw[wod_raw["DEPTH"] <= 10].copy()
        surf_wod["time"] = pd.to_datetime(surf_wod["TIME"], errors="coerce")
        surf_wod         = surf_wod.dropna(subset=["time", "TEMPERATURE"])
        surf_wod["m"]    = surf_wod["time"].dt.month
        wmon = surf_wod.groupby("m")["TEMPERATURE"].agg(["mean", "std"]).reset_index()

        if not wmon.empty:
            ax_wm.fill_between(wmon["m"],
                               wmon["mean"] - wmon["std"], wmon["mean"] + wmon["std"],
                               alpha=0.2, color="seagreen", label="± 1 std")
            ax_wm.plot(wmon["m"], wmon["mean"], "o-",
                       color="seagreen", lw=2, ms=5, label="Monthly mean")
        else:
            _blank(ax_wm, "No surface WOD data (depth ≤ 10 m)")

        ax_wm.set_xticks(range(1, 13))
        ax_wm.set_xticklabels(MONTH_LABELS, fontsize=7)
        ax_wm.set_xlabel("Month")
        ax_wm.set_ylabel("Temperature (°C)")
        ax_wm.set_title(
            f"WOD Monthly Mean ± Std (depth ≤ 10 m)\n"
            f"({rlat:.4f}°N, {rlon:.4f}°E) · 1970–2023", fontsize=9)
        ax_wm.legend(fontsize=7)
        ax_wm.grid(True, alpha=0.3)

        if not surf_wod.empty:
            years_w   = sorted(surf_wod["time"].dt.year.unique())
            colours_w = cm.plasma(np.linspace(0, 1, len(years_w)))
            for col_w, (_, ydata) in zip(colours_w,
                                         surf_wod.groupby(surf_wod["time"].dt.year)):
                ax_wd.scatter(ydata["time"].dt.dayofyear, ydata["TEMPERATURE"],
                              s=6, color=col_w, alpha=0.5)
            surf_wod["doy"] = surf_wod["time"].dt.dayofyear
            doy_med_w = surf_wod.groupby("doy")["TEMPERATURE"].median()
            ax_wd.plot(doy_med_w.index, doy_med_w.values,
                       color="crimson", lw=2, zorder=5, label="Daily median")
            sm_w = plt.cm.ScalarMappable(
                cmap="plasma",
                norm=plt.Normalize(vmin=min(years_w), vmax=max(years_w)))
            sm_w.set_array([])
            fig.colorbar(sm_w, ax=ax_wd, pad=0.02, label="Year")
        else:
            _blank(ax_wd, "No surface WOD data (depth ≤ 10 m)")

        ax_wd.set_xlabel("Day of Year")
        ax_wd.set_ylabel("Temperature (°C)")
        ax_wd.set_title(
            f"WOD Interannual Variability (depth ≤ 10 m)\n"
            f"({rlat:.4f}°N, {rlon:.4f}°E)", fontsize=9)
        ax_wd.legend(fontsize=7)
        ax_wd.grid(True, alpha=0.3)
    else:
        _blank(ax_wm, "WOD data not available")
        _blank(ax_wd, "WOD data not available")

    # ── Row 3 — Depth profiles ────────────────────────────────────────────────
    if wod_raw is not None and not wod_raw.empty:
        raw_clip = wod_raw[wod_raw["DEPTH"] <= max_depth].copy()
        MAX_PTS  = 8_000
        plot_df  = (raw_clip.sample(min(MAX_PTS, len(raw_clip)), random_state=42)
                    if len(raw_clip) > 0 else raw_clip)
        if not plot_df.empty:
            sc = ax_ws.scatter(
                plot_df["TEMPERATURE"], plot_df["DEPTH"],
                c=plot_df["DEPTH"], cmap="Blues_r",
                s=4, alpha=0.4, vmin=0, vmax=max_depth)
            fig.colorbar(sc, ax=ax_ws, label="Depth (m)", pad=0.02)
        else:
            _blank(ax_ws, "No WOD data in depth range")
        ax_ws.set_xlabel("Temperature (°C)")
        ax_ws.set_ylabel("Depth (m)")
        ax_ws.invert_yaxis()
        ax_ws.set_ylim(bottom=max_depth, top=0)
        ax_ws.set_title(
            f"WOD T–Depth Observations\n({rlat:.4f}°N, {rlon:.4f}°E)\n"
            f"n = {len(raw_clip):,} · 0 – {max_depth:.0f} m", fontsize=9)
        ax_ws.grid(True, alpha=0.3)
    else:
        _blank(ax_ws, "WOD data not available")

    if cora_dp is not None and not cora_dp.empty and "depth" in cora_dp.columns:
        profile = (cora_dp.groupby("depth")["TEMP"]
                   .agg(["mean", "std", "median"]).reset_index().sort_values("depth"))
        ax_cp.fill_betweenx(profile["depth"],
                             profile["mean"] - profile["std"],
                             profile["mean"] + profile["std"],
                             alpha=0.18, color="steelblue", label="± 1 std")
        ax_cp.plot(profile["mean"] - profile["std"], profile["depth"],
                   "--", color="royalblue", lw=1.2, alpha=0.7, label="Mean − std")
        ax_cp.plot(profile["mean"] + profile["std"], profile["depth"],
                   "--", color="tomato", lw=1.2, alpha=0.7, label="Mean + std")
        ax_cp.plot(profile["mean"],   profile["depth"],
                   "-",  color="steelblue", lw=2.5, label="Mean")
        ax_cp.plot(profile["median"], profile["depth"],
                   ":",  color="darkorange", lw=1.8, label="Median")
        ax_cp.set_xlabel("Temperature (°C)")
        ax_cp.set_ylabel("Depth (m)")
        ax_cp.invert_yaxis()
        ax_cp.set_ylim(bottom=max_depth, top=0)
        ax_cp.set_title(
            f"CORA T–Depth Profile\n({rlat:.4f}°N, {rlon:.4f}°E)\n"
            f"0 – {max_depth:.0f} m · 1990–2023", fontsize=9)
        ax_cp.legend(fontsize=7)
        ax_cp.grid(True, alpha=0.3)
    else:
        _blank(ax_cp, f"CORA depth profile not available\ndown to {max_depth} m")

    for ax, label in [
        (ax_cm, "CORA surface"),
        (ax_wm, "WOD surface\n(depth ≤ 10 m)"),
        (ax_ws, f"Depth profiles\n(0 – {max_depth} m)"),
    ]:
        ax.annotate(label, xy=(-0.18, 0.5), xycoords="axes fraction",
                    fontsize=8, fontweight="bold", color="#00A6D6",
                    ha="center", va="center", rotation=90)

    st.pyplot(fig)
    plt.close(fig)

    # ── Hovmöller diagrams ────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div class='section-hdr'>🌡️ Monthly Climatological Hovmöller Diagrams</div>",
        unsafe_allow_html=True,
    )

    fig3, axes3 = plt.subplots(1, 2, figsize=(18, 7),
                                gridspec_kw={"wspace": 0.28})
    ax_ch, ax_wh = axes3[0], axes3[1]

    if has_cora_dp:
        cora_plot = cora_dp.dropna(subset=["time", "depth", "TEMP"]).copy()
        cora_plot["month"]     = cora_plot["time"].dt.month
        depth_bin              = 10
        cora_plot["DEPTH_BIN"] = np.round(cora_plot["depth"] / depth_bin) * depth_bin
        cora_monthly = (cora_plot.groupby(["month", "DEPTH_BIN"])["TEMP"]
                        .mean().reset_index())
        if not cora_monthly.empty:
            hov_c = cora_monthly.pivot(
                index="DEPTH_BIN", columns="month", values="TEMP").sort_index()
            cf_c = ax_ch.contourf(hov_c.columns, hov_c.index, hov_c.values,
                                  levels=30, cmap="rainbow", extend="both")
            fig3.colorbar(cf_c, ax=ax_ch, pad=0.02).set_label("Temperature (°C)", fontsize=8)
            ax_ch.contour(hov_c.columns, hov_c.index, hov_c.values,
                          levels=15, colors="k", linewidths=0.25, alpha=0.35)
        else:
            _blank(ax_ch, "No CORA climatology available")
        ax_ch.set_xticks(range(1, 13))
        ax_ch.set_xticklabels(MONTH_LABELS, fontsize=8)
        ax_ch.set_xlabel("Month")
        ax_ch.set_ylabel("Depth (m)")
        ax_ch.invert_yaxis()
        ax_ch.set_ylim(bottom=max_depth, top=0)
        ax_ch.set_title(
            f"CORA Monthly Climatological Hovmöller\n"
            f"({rlat:.4f}°N, {rlon:.4f}°E)", fontsize=10)
        ax_ch.grid(False)
    else:
        _blank(ax_ch, "CORA depth data not available")

    if has_wod_dp:
        wod_plot         = wod_raw[wod_raw["DEPTH"] <= max_depth].copy()
        wod_plot["time"] = pd.to_datetime(wod_plot["TIME"], errors="coerce")
        wod_plot         = wod_plot.dropna(subset=["time", "DEPTH", "TEMPERATURE"])
        wod_plot["month"]     = wod_plot["time"].dt.month
        depth_bin             = 10
        wod_plot["DEPTH_BIN"] = np.round(wod_plot["DEPTH"] / depth_bin) * depth_bin
        wod_monthly = (wod_plot.groupby(["month", "DEPTH_BIN"])["TEMPERATURE"]
                       .mean().reset_index())
        if not wod_monthly.empty:
            hov_w = wod_monthly.pivot(
                index="DEPTH_BIN", columns="month", values="TEMPERATURE").sort_index()
            cf_w = ax_wh.contourf(hov_w.columns, hov_w.index, hov_w.values,
                                  levels=30, cmap="rainbow", extend="both")
            fig3.colorbar(cf_w, ax=ax_wh, pad=0.02).set_label("Temperature (°C)", fontsize=8)
            ax_wh.contour(hov_w.columns, hov_w.index, hov_w.values,
                          levels=15, colors="k", linewidths=0.25, alpha=0.35)
        else:
            _blank(ax_wh, "No WOD climatology available")
        ax_wh.set_xticks(range(1, 13))
        ax_wh.set_xticklabels(MONTH_LABELS, fontsize=8)
        ax_wh.set_xlabel("Month")
        ax_wh.set_ylabel("Depth (m)")
        ax_wh.invert_yaxis()
        ax_wh.set_ylim(bottom=max_depth, top=0)
        ax_wh.set_title(
            f"WOD Monthly Climatological Hovmöller\n"
            f"({rlat:.4f}°N, {rlon:.4f}°E)", fontsize=10)
        ax_wh.grid(False)
    else:
        _blank(ax_wh, "WOD data not available")

    st.pyplot(fig3)
    plt.close(fig3)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div style='text-align:center;color:grey;font-size:13px;'>"
        "CS-MACH1 Project · Ocean Climate Explorer · "
        "CORA (EMODnet-Physics ERDDAP) + WOD (Beacon API / MARIS) · 1970–2023"
        "</div>",
        unsafe_allow_html=True,
    )
