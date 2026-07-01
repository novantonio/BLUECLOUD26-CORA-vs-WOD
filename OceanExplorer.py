"""
OceanExplorer_TS.py
Versione corretta con mappa funzionante + Salinity + Hovmöller
"""

from __future__ import annotations

import io
import warnings
from datetime import datetime

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CS-MACH1 Ocean Explorer", page_icon="🌊", layout="wide")

st.markdown("""
<style>
.main-title {font-size:2.2rem; font-weight:800; color:#00A6D6; letter-spacing:-0.5px;}
.section-hdr {font-size:1.3rem; font-weight:700; color:#00A6D6; border-bottom:2px solid #00A6D6; padding-bottom:6px; margin:1.5rem 0 0.8rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🌊 CS-MACH1 — Ocean T+S Explorer</div>", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DEFAULT_LAT, DEFAULT_LON = 44.38, 9.07

CORA_TEMP_SURF = "https://erddap.emodnet-physics.eu/erddap/griddap/INSITU_GLO_PHY_TS_OA_MY_013_052_TEMP.csv?TEMP%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D%5B(1.0):1:(1)%5D%5B({lat}):1:({lat})%5D%5B({lon}):1:({lon})%5D"
CORA_TEMP_DEPTH = "https://erddap.emodnet-physics.eu/erddap/griddap/INSITU_GLO_PHY_TS_OA_MY_013_052_TEMP.csv?TEMP%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D%5B(1.0):1:({depth})%5D%5B({lat}):1:({lat})%5D%5B({lon}):1:({lon})%5D"

CORA_PSAL_SURF = "https://erddap.emodnet-physics.eu/erddap/griddap/INSITU_GLO_PHY_TS_OA_MY_013_052_PSAL.csv?PSAL%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D%5B(1.0):1:(1)%5D%5B({lat}):1:({lat})%5D%5B({lon}):1:({lon})%5D"
CORA_PSAL_DEPTH = "https://erddap.emodnet-physics.eu/erddap/griddap/INSITU_GLO_PHY_TS_OA_MY_013_052_PSAL.csv?PSAL%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D%5B(1.0):1:({depth})%5D%5B({lat}):1:({lat})%5D%5B({lon}):1:({lon})%5D"

# ── Data fetchers ─────────────────────────────────────────────────────────────
def _normalize_cora(df: pd.DataFrame, var: str) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    for possible in [var, var.upper(), "SEA_WATER_TEMPERATURE", "SEA_WATER_SALINITY", "PRACTICAL_SALINITY"]:
        if possible in df.columns:
            if possible != var:
                df = df.rename(columns={possible: var})
            break
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if "depth" not in df.columns and "z" in df.columns:
        df = df.rename(columns={"z": "depth"})
    return df


@st.cache_data(ttl=3600)
def fetch_wod_all(lat: float, lon: float):
    try:
        from beacon_api import Client
        client = Client("https://beacon-wod.maris.nl")
        qb = client.query()
        qb.add_select_column("Temperature", alias="TEMPERATURE")
        qb.add_select_column("Salinity", alias="PSAL")
        qb.add_select_column("z", alias="DEPTH")
        qb.add_select_column("time", alias="TIME")
        raw = qb.to_pandas_dataframe()
        for c in ["TEMPERATURE", "PSAL", "DEPTH"]:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw["TIME"] = pd.to_datetime(raw["TIME"], errors="coerce")
        return raw.dropna(subset=["DEPTH"])
    except Exception as e:
        st.warning(f"WOD: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def fetch_cora_surface(lat: float, lon: float, is_salinity=False):
    url = (CORA_PSAL_SURF if is_salinity else CORA_TEMP_SURF).format(lat=round(lat,4), lon=round(lon,4))
    var = "PSAL" if is_salinity else "TEMP"
    try:
        r = requests.get(url, verify=False, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), skiprows=[1])
        df = _normalize_cora(df, var)
        df[var] = pd.to_numeric(df[var], errors="coerce").round(3)
        return df.dropna(subset=["time", var])
    except Exception as e:
        st.warning(f"CORA surface: {e}")
        return None


@st.cache_data(ttl=86400)
def fetch_cora_depth(lat: float, lon: float, max_depth: float, is_salinity=False):
    url = (CORA_PSAL_DEPTH if is_salinity else CORA_TEMP_DEPTH).format(lat=round(lat,4), lon=round(lon,4), depth=float(max_depth))
    var = "PSAL" if is_salinity else "TEMP"
    try:
        r = requests.get(url, verify=False, timeout=90)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), skiprows=[1])
        df = _normalize_cora(df, var)
        df[var] = pd.to_numeric(df[var], errors="coerce").round(3)
        if "depth" in df.columns:
            df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
        return df.dropna(subset=["time", var])
    except Exception as e:
        st.warning(f"CORA depth: {e}")
        return None


# ── Plot functions ────────────────────────────────────────────────────────────
def plot_monthly(df, var, title, color="steelblue"):
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.text(0.5,0.5,"No data", ha="center", va="center")
        return fig
    df = df.copy()
    df["m"] = df["time"].dt.month
    monthly = df.groupby("m")[var].agg(["mean","std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8,5))
    ax.fill_between(monthly["m"], monthly["mean"]-monthly["std"], monthly["mean"]+monthly["std"], alpha=0.25, color=color)
    ax.plot(monthly["m"], monthly["mean"], "o-", color=color, lw=2)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel("Temperature (°C)" if var=="TEMP" else "Salinity (PSU)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_depth_profile(df, var, max_depth, title, color="steelblue"):
    fig, ax = plt.subplots(figsize=(6,8))
    if df is None or df.empty or "depth" not in df.columns:
        ax.text(0.5,0.5,"No depth data", ha="center", va="center")
        fig.tight_layout()
        return fig
    profile = df.groupby("depth")[var].agg(["mean","std","median"]).reset_index()
    ax.fill_betweenx(profile["depth"], profile["mean"]-profile["std"], profile["mean"]+profile["std"], alpha=0.2, color=color)
    ax.plot(profile["mean"], profile["depth"], color=color, lw=2.5)
    ax.set_xlabel(var)
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_ylim(max_depth, 0)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_hovmoller(df, var, max_depth, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    dfp = df.copy()
    if "DEPTH" in dfp.columns:
        dfp = dfp.rename(columns={"DEPTH": "depth"})
    if "TIME" in dfp.columns:
        dfp["time"] = pd.to_datetime(dfp["TIME"])
    dfp = dfp[dfp["depth"] <= max_depth].dropna(subset=["time", "depth", var])
    dfp["month"] = dfp["time"].dt.month
    dfp["DEPTH_BIN"] = np.round(dfp["depth"] / 10) * 10
    monthly = dfp.groupby(["month", "DEPTH_BIN"])[var].mean().reset_index()
    if monthly.empty:
        ax.text(0.5,0.5,"Insufficient data", ha="center")
        return fig
    hov = monthly.pivot(index="DEPTH_BIN", columns="month", values=var).sort_index()
    cf = ax.contourf(hov.columns, hov.index, hov.values, levels=30, cmap="RdYlBu_r", extend="both")
    ax.contour(hov.columns, hov.index, hov.values, levels=10, colors="k", linewidths=0.3)
    fig.colorbar(cf, ax=ax, label=var)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel("Month")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ── Sidebar + Mappa (presa dall'originale) ────────────────────────────────────
with st.sidebar:
    st.markdown("### 📍 Location")
    _lat_key = f"lat_{st.session_state.get('sel_lat', DEFAULT_LAT)}"
    _lon_key = f"lon_{st.session_state.get('sel_lon', DEFAULT_LON)}"
    lat_in = st.number_input("Latitude (°N)", -90.0, 90.0,
                             value=st.session_state.get("sel_lat", DEFAULT_LAT),
                             step=0.01, format="%.4f", key=_lat_key)
    lon_in = st.number_input("Longitude (°E)", -180.0, 180.0,
                             value=st.session_state.get("sel_lon", DEFAULT_LON),
                             step=0.01, format="%.4f", key=_lon_key)
    st.divider()
    max_depth = st.slider("Max depth (m)", 10, 5000, 300, step=10, key="depth_slider")
    run_btn = st.button("▶️ Run Analysis", type="primary", use_container_width=True)

# ── Mappa interattiva (blocco originale corretto) ─────────────────────────────
st.markdown("<div class='section-hdr'>🗺️ Select Point on Map</div>", unsafe_allow_html=True)

center_lat = st.session_state.get("sel_lat", DEFAULT_LAT)
center_lon = st.session_state.get("sel_lon", DEFAULT_LON)

m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles=None)

folium.TileLayer("CartoDB positron", name="CartoDB Positron").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Esri Ocean"
).add_to(m)

folium.Marker(
    location=[center_lat, center_lon],
    tooltip=f"Selected: {center_lat:.4f}°N, {center_lon:.4f}°E",
    icon=folium.Icon(color="blue", icon="tint")
).add_to(m)

map_result = st_folium(m, use_container_width=True, height=420, returned_objects=["last_clicked"])

if map_result and map_result.get("last_clicked"):
    clicked = map_result["last_clicked"]
    new_lat = round(clicked["lat"], 4)
    new_lon = round(clicked["lng"], 4)
    if (new_lat != st.session_state.get("sel_lat") or new_lon != st.session_state.get("sel_lon")):
        st.session_state["sel_lat"] = new_lat
        st.session_state["sel_lon"] = new_lon
        st.rerun()

latitude = st.session_state.get("sel_lat", lat_in)
longitude = st.session_state.get("sel_lon", lon_in)

st.info(f"📍 **Point:** {latitude:.4f}°N, {longitude:.4f}°E  · Max depth: **{max_depth} m**")

# ── Analysis ──────────────────────────────────────────────────────────────────
if run_btn or st.session_state.get("results"):
    if run_btn:
        with st.spinner("Fetching data from CORA..."):
            res = {
                "cora_t_surf": fetch_cora_surface(latitude, longitude, False),
                "cora_s_surf": fetch_cora_surface(latitude, longitude, True),
                "cora_t_dep": fetch_cora_depth(latitude, longitude, max_depth, False),
                "cora_s_dep": fetch_cora_depth(latitude, longitude, max_depth, True),
                "lat": latitude,
                "lon": longitude
            }
            st.session_state.results = res

    res = st.session_state.results

    # Temperature
    st.markdown("<div class='section-hdr'>🌡️ TEMPERATURE</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_monthly(res["cora_t_surf"], "TEMP", "CORA Temperature Monthly"))
        st.pyplot(plot_hovmoller(res["cora_t_dep"], "TEMP", max_depth, "CORA Temperature Hovmöller"))
    with c2:
        st.pyplot(plot_depth_profile(res["cora_t_dep"], "TEMP", max_depth, "CORA Temp Depth Profile"))

    # Salinity
    st.markdown("<div class='section-hdr'>🌊 SALINITY</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_monthly(res["cora_s_surf"], "PSAL", "CORA Salinity Monthly", "teal"))
        st.pyplot(plot_hovmoller(res["cora_s_dep"], "PSAL", max_depth, "CORA Salinity Hovmöller"))
    with c2:
        st.pyplot(plot_depth_profile(res["cora_s_dep"], "PSAL", max_depth, "CORA Salinity Depth Profile", "teal"))

    st.success("Analysis completed!")

st.caption("CORA (EMODnet Physics) • WOD")
