"""
ocean_explorer.py
─────────────────
CS-MACH1 — Ocean Climate Explorer

Workflow
────────
1. User clicks a point on an interactive Leaflet map (st_folium)
   OR types lat / lon manually in the sidebar.
2. User sets max depth and presses "Run Analysis".
3. App fetches:
   • CORA surface climatology (ERDDAP / EMODnet-Physics)
   • WOD full water-column profiles (Beacon API / novantonio/wod)
4. Four plots are rendered:
   ┌─────────────────────┬─────────────────────┐
   │ CORA monthly        │ CORA DOY             │
   │ mean ± std          │ interannual scatter  │
   ├─────────────────────┼─────────────────────┤
   │ WOD T-profile       │ WOD min/max          │
   │ scatter (0–maxdep)  │ envelope (0–maxdep)  │
   └─────────────────────┴─────────────────────┘

Dependencies (pip install):
    streamlit folium streamlit-folium requests pandas
    matplotlib numpy beacon-api
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
    page_title="CS-MACH1 Ocean Climate Explorer",
    page_icon="🌊",
    layout="wide",
)

st.markdown("""
<style>
.main-title   { font-size:2rem; font-weight:800; color:#00A6D6; letter-spacing:-0.5px; }
.sub-title    { font-size:1rem; color:#555; margin-bottom:1rem; }
.section-hdr  { font-size:1.2rem; font-weight:700; color:#00A6D6;
                border-bottom:2px solid #00A6D6; padding-bottom:4px;
                margin-top:1.4rem; margin-bottom:.6rem; }
.stButton>button { background-color:#00A6D6; color:white;
                   border-radius:8px; border:none; font-weight:600; }
.stButton>button:hover { background-color:#007EA3; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🌊 CS-MACH1 — Ocean Climate Explorer</div>",
            unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>"
    "Click a point on the map (or type coordinates) → set max depth → Run Analysis"
    "</div>",
    unsafe_allow_html=True,
)


# ── Constants ─────────────────────────────────────────────────────────────────

CORA_URL = (
    "https://erddap.emodnet-physics.eu/erddap/griddap/"
    "INSITU_GLO_PHY_TS_OA_MY_013_052_TEMP.csv"
    "?TEMP%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D"
    "%5B(1.0):1:(1)%5D"
    "%5B({lat}):1:({lat})%5D"
    "%5B({lon}):1:({lon})%5D"
)

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

DEFAULT_LAT, DEFAULT_LON = 44.38, 9.07


# ── WOD client (inline — mirrors novantonio/wod/wod_client.py) ─────────────────

def _wod_client():
    try:
        from beacon_api import Client          # noqa: PLC0415
        return Client("https://beacon-wod.maris.nl")
    except ImportError as exc:
        raise ImportError("Run: pip install beacon-api") from exc


@st.cache_data(show_spinner="Querying World Ocean Database…", ttl=3600)
def fetch_wod(latitude: float, longitude: float,
              max_depth: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (temp_range_df, raw_df).

    temp_range_df : DEPTH | min_temperature | max_temperature   (clipped to max_depth)
    raw_df        : DEPTH | TEMPERATURE | TIME | LATITUDE | LONGITUDE | …
    """
    client  = _wod_client()
    lat_min = round(latitude,  1) - 0.5
    lat_max = round(latitude,  1) + 0.5
    lon_min = round(longitude, 1) - 0.5
    lon_max = round(longitude, 1) + 0.5

    qb = client.query()
    qb.add_select_column("wod_unique_cast")
    qb.add_select_column("Temperature",         alias="TEMPERATURE")
    qb.add_select_column("Temperature_WODflag", alias="TEMPERATURE_QC")
    qb.add_select_column("z",                   alias="DEPTH")
    qb.add_select_column("time",                alias="TIME")
    qb.add_select_column("lon",                 alias="LONGITUDE")
    qb.add_select_column("lat",                 alias="LATITUDE")

    qb.add_range_filter("TIME",      "1970-01-01T00:00:00", "2023-01-01T00:00:00")
    qb.add_is_not_null_filter("TEMPERATURE")
    qb.add_not_equals_filter("TEMPERATURE", -1e10)
    qb.add_equals_filter("TEMPERATURE_QC",  0.0)
    qb.add_range_filter("DEPTH",     0, max_depth)
    qb.add_range_filter("LONGITUDE", lon_min, lon_max)
    qb.add_range_filter("LATITUDE",  lat_min, lat_max)

    raw = qb.to_pandas_dataframe()
    raw["TEMPERATURE"] = pd.to_numeric(raw["TEMPERATURE"], errors="coerce")
    raw["DEPTH"]       = pd.to_numeric(raw["DEPTH"],       errors="coerce")
    raw["TIME"]        = pd.to_datetime(raw["TIME"],        errors="coerce")
    raw = raw.dropna(subset=["DEPTH", "TEMPERATURE"])

    # Depth-binned min/max envelope
    mn = raw.groupby("DEPTH")["TEMPERATURE"].min().reset_index()
    mx = raw.groupby("DEPTH")["TEMPERATURE"].max().reset_index()
    env = pd.merge(
        mn.rename(columns={"TEMPERATURE": "min_temperature"}),
        mx.rename(columns={"TEMPERATURE": "max_temperature"}),
        on="DEPTH",
    ).sort_values("DEPTH").reset_index(drop=True)
    for col in ("min_temperature", "max_temperature"):
        env[col] = env[col].interpolate(method="linear", limit_direction="both")

    return env, raw


@st.cache_data(show_spinner="Downloading CORA climatology…", ttl=86400)
def fetch_cora(latitude: float, longitude: float) -> pd.DataFrame | None:
    url = CORA_URL.format(lat=round(latitude, 4), lon=round(longitude, 4))
    try:
        r = requests.get(url, verify=False, timeout=60)
        r.raise_for_status()
        if "<html" in r.text.lower():
            raise ValueError("CORA returned an HTML error page.")
        df = pd.read_csv(io.StringIO(r.text), skiprows=[1])
        df["time"] = pd.to_datetime(df["time"])
        df["TEMP"] = pd.to_numeric(df["TEMP"], errors="coerce")
        return df.dropna()
    except Exception as exc:
        st.warning(f"CORA fetch failed: {exc}")
        return None


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_cora_monthly(cora: pd.DataFrame,
                      latitude: float, longitude: float) -> plt.Figure:
    """CORA monthly mean ± std bar chart."""
    cora       = cora.copy()
    cora["m"]  = cora["time"].dt.month
    monthly    = cora.groupby("m")["TEMP"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(monthly["m"],
                    monthly["mean"] - monthly["std"],
                    monthly["mean"] + monthly["std"],
                    alpha=0.2, color="steelblue", label="± 1 std")
    ax.plot(monthly["m"], monthly["mean"], "o-",
            color="steelblue", lw=2, ms=6, label="Monthly mean")
    ax.plot(monthly["m"], monthly["mean"].rolling(3, center=True).mean(),
            "--", color="navy", lw=1.2, alpha=0.6, label="3-month smooth")

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        f"CORA Monthly Mean ± Std\n({latitude:.4f}°N, {longitude:.4f}°E) "
        f"· 1990–2023 surface",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cora_doy(cora: pd.DataFrame,
                  latitude: float, longitude: float) -> plt.Figure:
    """CORA interannual scatter: temperature vs day-of-year, one colour per year."""
    fig, ax = plt.subplots(figsize=(8, 5))

    years   = sorted(cora["time"].dt.year.unique())
    colours = cm.viridis(np.linspace(0, 1, len(years)))

    for colour, (yr, ydata) in zip(colours, cora.groupby(cora["time"].dt.year)):
        doy = ydata["time"].dt.dayofyear
        ax.scatter(doy, ydata["TEMP"], s=8, color=colour, alpha=0.55)

    # Monthly median overlay
    cora2       = cora.copy()
    cora2["doy"] = cora2["time"].dt.dayofyear
    doy_med     = cora2.groupby("doy")["TEMP"].median()
    ax.plot(doy_med.index, doy_med.values,
            color="crimson", lw=2, zorder=5, label="Daily median")

    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=min(years), vmax=max(years)),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, label="Year")

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        f"CORA Interannual Temperature Variability\n"
        f"({latitude:.4f}°N, {longitude:.4f}°E) · surface",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_wod_scatter(raw: pd.DataFrame, max_depth: float,
                     latitude: float, longitude: float) -> plt.Figure:
    """WOD individual observations: temperature vs depth scatter."""
    fig, ax = plt.subplots(figsize=(6, 8))

    MAX_PTS = 8_000
    plot_df = raw.sample(min(MAX_PTS, len(raw)), random_state=42)

    sc = ax.scatter(
        plot_df["TEMPERATURE"], plot_df["DEPTH"],
        c=plot_df["DEPTH"], cmap="Blues_r",
        s=5, alpha=0.4,
    )
    fig.colorbar(sc, ax=ax, label="Depth (m)", pad=0.02)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_ylim(bottom=max_depth, top=0)
    ax.set_title(
        f"WOD T–Depth Observations\n({latitude:.4f}°N, {longitude:.4f}°E)\n"
        f"n = {len(raw):,} · max depth {max_depth:.0f} m",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_wod_envelope(env: pd.DataFrame, raw: pd.DataFrame,
                      max_depth: float,
                      latitude: float, longitude: float) -> plt.Figure:
    """WOD climatological min/max envelope + monthly mean profile."""
    fig, ax = plt.subplots(figsize=(6, 8))

    # Min/max envelope
    ax.fill_betweenx(
        env["DEPTH"],
        env["min_temperature"],
        env["max_temperature"],
        alpha=0.15, color="steelblue", label="Min–Max envelope",
    )
    ax.plot(env["min_temperature"], env["DEPTH"],
            color="royalblue", lw=1.5, ls="--", label="Min")
    ax.plot(env["max_temperature"], env["DEPTH"],
            color="tomato",    lw=1.5, ls="--", label="Max")

    # Depth-binned mean profile
    mean_profile = raw.groupby("DEPTH")["TEMPERATURE"].mean()
    ax.plot(mean_profile.values, mean_profile.index,
            color="crimson", lw=2.5, zorder=5, label="Mean profile")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_ylim(bottom=max_depth, top=0)
    ax.set_title(
        f"WOD Temperature Envelope\n({latitude:.4f}°N, {longitude:.4f}°E)\n"
        f"max depth {max_depth:.0f} m · 1970–2023",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📍 Location")

    # Manual input (also updated when map is clicked)
    lat_in = st.number_input(
        "Latitude (°N)",  min_value=-90.0, max_value=90.0,
        value=st.session_state.get("sel_lat", DEFAULT_LAT),
        step=0.01, format="%.4f",
        key="lat_input",
    )
    lon_in = st.number_input(
        "Longitude (°E)", min_value=-180.0, max_value=180.0,
        value=st.session_state.get("sel_lon", DEFAULT_LON),
        step=0.01, format="%.4f",
        key="lon_input",
    )

    st.divider()
    st.markdown("### ⚙️ Parameters")

    max_depth = st.slider(
        "Max depth (m)", min_value=10, max_value=5000,
        value=200, step=10,
    )

    st.divider()
    run_btn = st.button("▶️ Run Analysis", type="primary", use_container_width=True)

    if st.button("🧹 Reset", use_container_width=True):
        for k in ["sel_lat", "sel_lon", "results"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()
    st.caption(
        "Data sources\n"
        "• **CORA**: EMODnet-Physics ERDDAP (1990–2023)\n"
        "• **WOD**: Beacon API / MARIS (1970–2023)\n"
        "• Search box: ±0.5° around selected point"
    )


# ── Map ───────────────────────────────────────────────────────────────────────

st.markdown("<div class='section-hdr'>🗺️ Select a Point on the Map</div>",
            unsafe_allow_html=True)
st.caption(
    "Click anywhere on the ocean to set the analysis location. "
    "You can also type coordinates directly in the sidebar."
)

center_lat = st.session_state.get("sel_lat", DEFAULT_LAT)
center_lon = st.session_state.get("sel_lon", DEFAULT_LON)

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles="CartoDB positron",
)

# Add ocean / bathymetry tile layer for context
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Esri Ocean",
    overlay=False,
    control=True,
).add_to(m)

# Selected point marker
folium.Marker(
    location=[center_lat, center_lon],
    tooltip=f"Selected: {center_lat:.4f}°N, {center_lon:.4f}°E",
    icon=folium.Icon(color="blue", icon="tint", prefix="fa"),
).add_to(m)

# Search ±0.5° bounding box
folium.Rectangle(
    bounds=[
        [center_lat - 0.5, center_lon - 0.5],
        [center_lat + 0.5, center_lon + 0.5],
    ],
    color="#00A6D6", weight=1.5, fill=True, fill_opacity=0.08,
    tooltip="WOD search box (±0.5°)",
).add_to(m)

folium.LayerControl().add_to(m)

map_result = st_folium(m, width="100%", height=420, returned_objects=["last_clicked"])

# Capture click → update session state + sidebar inputs
if map_result and map_result.get("last_clicked"):
    clicked = map_result["last_clicked"]
    st.session_state["sel_lat"] = round(clicked["lat"], 4)
    st.session_state["sel_lon"] = round(clicked["lng"], 4)
    # Force sidebar inputs to follow by triggering a rerun
    st.rerun()

# Use sidebar values as the definitive source of truth
latitude  = lat_in
longitude = lon_in

st.info(
    f"📍 **Analysis point:** {latitude:.4f}°N, {longitude:.4f}°E  "
    f"· Max depth: **{max_depth} m**"
)


# ── Run analysis ──────────────────────────────────────────────────────────────

if run_btn:
    st.session_state.pop("results", None)

    col_prog, _ = st.columns([2, 1])
    with col_prog:
        pbar = st.progress(0, text="Fetching CORA data…")

    cora_df = fetch_cora(latitude, longitude)
    pbar.progress(40, text="Querying WOD…")

    try:
        env_df, raw_df = fetch_wod(latitude, longitude, float(max_depth))
        wod_ok = True
    except Exception as exc:
        st.warning(f"WOD query failed: {exc}")
        env_df = raw_df = None
        wod_ok = False

    pbar.progress(100, text="✅ Done!")

    if cora_df is not None or wod_ok:
        st.session_state["results"] = {
            "cora": cora_df,
            "env":  env_df,
            "raw":  raw_df,
            "lat":  latitude,
            "lon":  longitude,
            "dep":  max_depth,
            "ts":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        }


# ── Display results ───────────────────────────────────────────────────────────

if "results" in st.session_state:
    res   = st.session_state["results"]
    cora  = res["cora"]
    env   = res["env"]
    raw   = res["raw"]
    rlat  = res["lat"]
    rlon  = res["lon"]
    rdep  = res["dep"]

    st.markdown(
        f"<div class='section-hdr'>📊 Results — "
        f"{rlat:.4f}°N, {rlon:.4f}°E · max {rdep} m · {res['ts']}</div>",
        unsafe_allow_html=True,
    )

    # Quick metrics
    c1, c2, c3, c4 = st.columns(4)
    if cora is not None:
        c1.metric("CORA records",  f"{len(cora):,}")
        c2.metric("CORA period",
                  f"{cora['time'].dt.year.min()}–{cora['time'].dt.year.max()}")
    if raw is not None:
        c3.metric("WOD observations", f"{len(raw):,}")
        c4.metric("WOD depth range",
                  f"{raw['DEPTH'].min():.0f}–{raw['DEPTH'].max():.0f} m")

    st.divider()

    # ── Row 1: CORA monthly | CORA DOY ───────────────────────────────────────
    st.markdown("<div class='section-hdr'>🌡️ CORA Surface Climatology</div>",
                unsafe_allow_html=True)

    if cora is not None:
        col_l, col_r = st.columns(2)
        with col_l:
            fig_mon = plot_cora_monthly(cora, rlat, rlon)
            st.pyplot(fig_mon)
            plt.close(fig_mon)
        with col_r:
            fig_doy = plot_cora_doy(cora, rlat, rlon)
            st.pyplot(fig_doy)
            plt.close(fig_doy)
    else:
        st.warning("CORA data not available for this location.")

    st.divider()

    # ── Row 2: WOD scatter | WOD envelope ────────────────────────────────────
    st.markdown("<div class='section-hdr'>🔵 WOD Temperature Profiles (0 – {:.0f} m)</div>".format(rdep),
                unsafe_allow_html=True)

    if raw is not None and not raw.empty:
        col_l2, col_r2 = st.columns(2)
        with col_l2:
            fig_sc = plot_wod_scatter(raw, rdep, rlat, rlon)
            st.pyplot(fig_sc)
            plt.close(fig_sc)
        with col_r2:
            fig_env = plot_wod_envelope(env, raw, rdep, rlat, rlon)
            st.pyplot(fig_env)
            plt.close(fig_env)
    else:
        st.warning(
            "No WOD data found within ±0.5° of this point down to "
            f"{rdep} m. Try a different location or increase the search area."
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div style='text-align:center;color:grey;font-size:13px;'>"
        "CS-MACH1 Project · Ocean Climate Explorer · "
        "CORA (EMODnet-Physics) + WOD (Beacon API / MARIS) · 1970–2023"
        "</div>",
        unsafe_allow_html=True,
    )
