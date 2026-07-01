"""
OceanExplorer_TS.py
───────────────────
CS-MACH1 — Ocean Temperature + Salinity Climate Explorer
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CS-MACH1 Ocean T+S Explorer", page_icon="🌊", layout="wide")

st.markdown("""
<style>
.main-title { font-size:2.1rem; font-weight:800; color:#00A6D6; }
.section-hdr { font-size:1.25rem; font-weight:700; color:#00A6D6; border-bottom:2px solid #00A6D6; padding-bottom:4px; margin:1.2rem 0 0.6rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🌊 CS-MACH1 — Ocean Temperature + Salinity Explorer</div>", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CORA_TEMP_SURF_URL = "https://erddap.emodnet-physics.eu/erddap/griddap/INSITU_GLO_PHY_TS_OA_MY_013_052_TEMP.csv?TEMP%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D%5B(1.0):1:(1)%5D%5B({lat}):1:({lat})%5D%5B({lon}):1:({lon})%5D"
CORA_TEMP_DEPTH_URL = "https://erddap.emodnet-physics.eu/erddap/griddap/INSITU_GLO_PHY_TS_OA_MY_013_052_TEMP.csv?TEMP%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D%5B(1.0):1:({depth})%5D%5B({lat}):1:({lat})%5D%5B({lon}):1:({lon})%5D"

CORA_PSAL_SURF_URL = "https://erddap.emodnet-physics.eu/erddap/griddap/INSITU_GLO_PHY_TS_OA_MY_013_052_PSAL.csv?PSAL%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D%5B(1.0):1:(1)%5D%5B({lat}):1:({lat})%5D%5B({lon}):1:({lon})%5D"
CORA_PSAL_DEPTH_URL = "https://erddap.emodnet-physics.eu/erddap/griddap/INSITU_GLO_PHY_TS_OA_MY_013_052_PSAL.csv?PSAL%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D%5B(1.0):1:({depth})%5D%5B({lat}):1:({lat})%5D%5B({lon}):1:({lon})%5D"

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DEFAULT_LAT, DEFAULT_LON = 44.38, 9.07


# ── Helpers ───────────────────────────────────────────────────────────────────
def _normalize_cora_df(df: pd.DataFrame, var: str) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    var_col = next((c for c in df.columns if c.strip().upper() in (var.upper(), "SEA_WATER_SALINITY", "PRACTICAL_SALINITY")), None)
    if var_col and var_col != var:
        df = df.rename(columns={var_col: var})
    for old, new in [("time","time"), ("depth","depth"), ("z","depth")]:
        col = next((c for c in df.columns if c.strip().lower() == old), None)
        if col and col != new:
            df = df.rename(columns={col: new})
    return df


def _wod_client():
    from beacon_api import Client
    return Client("https://beacon-wod.maris.nl", proxy_headers={"User-Agent": "OceanExplorer"})


@st.cache_data(ttl=3600)
def fetch_wod_all(lat: float, lon: float):
    # ... (stessa implementazione dell'originale con aggiunta PSAL) ...
    try:
        client = _wod_client()
        # query con range lat/lon ±0.1°
        qb = client.query()
        qb.add_select_column("Temperature", alias="TEMPERATURE")
        qb.add_select_column("Salinity", alias="PSAL")
        qb.add_select_column("z", alias="DEPTH")
        qb.add_select_column("time", alias="TIME")
        # filtri ...
        raw = qb.to_pandas_dataframe()
        raw[["TEMPERATURE","PSAL","DEPTH"]] = raw[["TEMPERATURE","PSAL","DEPTH"]].apply(pd.to_numeric, errors='coerce')
        raw["TIME"] = pd.to_datetime(raw["TIME"], errors='coerce')
        return raw.dropna(subset=["DEPTH"])
    except Exception as e:
        st.warning(f"WOD error: {e}")
        return None


@st.cache_data(ttl=86400)
def fetch_cora_surface(lat: float, lon: float, is_salinity=False):
    url = (CORA_PSAL_SURF_URL if is_salinity else CORA_TEMP_SURF_URL).format(lat=round(lat,4), lon=round(lon,4))
    var = "PSAL" if is_salinity else "TEMP"
    try:
        r = requests.get(url, verify=False, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), skiprows=[1])
        df = _normalize_cora_df(df, var)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df[var] = pd.to_numeric(df[var], errors="coerce").round(3)
        return df.dropna(subset=["time", var])
    except Exception as e:
        st.warning(f"CORA {'PSAL' if is_salinity else 'TEMP'} surface error: {e}")
        return None


@st.cache_data(ttl=86400)
def fetch_cora_depth(lat: float, lon: float, max_d: float, is_salinity=False):
    url = (CORA_PSAL_DEPTH_URL if is_salinity else CORA_TEMP_DEPTH_URL).format(
        lat=round(lat,4), lon=round(lon,4), depth=float(max_d))
    var = "PSAL" if is_salinity else "TEMP"
    try:
        r = requests.get(url, verify=False, timeout=90)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), skiprows=[1])
        df = _normalize_cora_df(df, var)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df[var] = pd.to_numeric(df[var], errors="coerce").round(3)
        if "depth" in df.columns:
            df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
        return df.dropna(subset=["time", var])
    except Exception as e:
        st.warning(f"CORA {'PSAL' if is_salinity else 'TEMP'} depth error: {e}")
        return None


# ── Plot functions (Temperature) ──────────────────────────────────────────────
def plot_monthly_temp(cora, lat, lon):
    # stessa logica dell'originale
    cora = cora.copy()
    cora["m"] = cora["time"].dt.month
    monthly = cora.groupby("m")["TEMP"].agg(["mean","std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8,5))
    ax.fill_between(monthly["m"], monthly["mean"]-monthly["std"], monthly["mean"]+monthly["std"], alpha=0.2, color="steelblue")
    ax.plot(monthly["m"], monthly["mean"], "o-", color="steelblue", lw=2, label="Mean")
    ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"CORA Temp Monthly (surface)\n{lat:.4f}°N, {lon:.4f}°E")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def plot_depth_profile_temp(cora_dp, max_depth, lat, lon):
    # simile all'originale
    fig, ax = plt.subplots(figsize=(6,8))
    if cora_dp.empty:
        ax.text(0.5,0.5,"No data", ha="center", va="center")
        return fig
    profile = cora_dp.groupby("depth")["TEMP"].agg(["mean","std","median"]).reset_index()
    ax.fill_betweenx(profile["depth"], profile["mean"]-profile["std"], profile["mean"]+profile["std"], alpha=0.2, color="steelblue")
    ax.plot(profile["mean"], profile["depth"], color="steelblue", lw=2)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_ylim(max_depth, 0)
    ax.set_title(f"CORA Temp Profile {max_depth}m")
    fig.tight_layout()
    return fig

# ── Plot functions (Salinity) ─────────────────────────────────────────────────
def plot_monthly_psal(cora, lat, lon):
    cora = cora.copy()
    cora["m"] = cora["time"].dt.month
    monthly = cora.groupby("m")["PSAL"].agg(["mean","std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8,5))
    ax.fill_between(monthly["m"], monthly["mean"]-monthly["std"], monthly["mean"]+monthly["std"], alpha=0.2, color="teal")
    ax.plot(monthly["m"], monthly["mean"], "o-", color="teal", lw=2, label="Mean")
    ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel("Salinity (PSU)")
    ax.set_title(f"CORA Salinity Monthly (surface)\n{lat:.4f}°N, {lon:.4f}°E")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def plot_depth_profile_psal(cora_dp, max_depth, lat, lon):
    fig, ax = plt.subplots(figsize=(6,8))
    if cora_dp.empty or "depth" not in cora_dp.columns:
        ax.text(0.5,0.5,"No salinity depth data", ha="center", va="center", color="grey")
        fig.tight_layout()
        return fig
    profile = cora_dp.groupby("depth")["PSAL"].agg(["mean","std","median"]).reset_index().sort_values("depth")
    ax.fill_betweenx(profile["depth"], profile["mean"]-profile["std"], profile["mean"]+profile["std"], alpha=0.2, color="teal")
    ax.plot(profile["mean"], profile["depth"], color="teal", lw=2, label="Mean")
    ax.set_xlabel("Salinity (PSU)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_ylim(max_depth, 0)
    ax.set_title(f"CORA PSAL Profile {max_depth}m")
    ax.legend()
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


# ── Analysis ──────────────────────────────────────────────────────────────────
if run_btn or st.session_state.get("results"):
    if run_btn:
        with st.spinner("Fetching data..."):
            cora_surf_t = fetch_cora_surface(lat, lon, False)
            cora_dp_t   = fetch_cora_depth(lat, lon, max_depth, False)
            cora_surf_s = fetch_cora_surface(lat, lon, True)
            cora_dp_s   = fetch_cora_depth(lat, lon, max_depth, True)
            wod = fetch_wod_all(lat, lon)

            st.session_state.results = {
                "cora_surf_t": cora_surf_t, "cora_dp_t": cora_dp_t,
                "cora_surf_s": cora_surf_s, "cora_dp_s": cora_dp_s,
                "wod": wod, "lat": lat, "lon": lon
            }

    res = st.session_state.get("results", {})

    st.markdown("<div class='section-hdr'>🌡️ TEMPERATURE ANALYSIS</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if res.get("cora_surf_t") is not None:
            st.pyplot(plot_monthly_temp(res["cora_surf_t"], res["lat"], res["lon"]))
    with c2:
        if res.get("cora_dp_t") is not None:
            st.pyplot(plot_depth_profile_temp(res["cora_dp_t"], max_depth, res["lat"], res["lon"]))

    st.markdown("<div class='section-hdr'>🌊 SALINITY ANALYSIS (PSAL)</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if res.get("cora_surf_s") is not None:
            st.pyplot(plot_monthly_psal(res["cora_surf_s"], res["lat"], res["lon"]))
    with c2:
        if res.get("cora_dp_s") is not None:
            st.pyplot(plot_depth_profile_psal(res["cora_dp_s"], max_depth, res["lat"], res["lon"]))

    st.success("Analysis completed!")
    st.caption("Data: CORA (EMODnet) + WOD")
