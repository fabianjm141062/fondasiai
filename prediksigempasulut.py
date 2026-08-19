
import math
from datetime import date

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ============================================================
# FJM AI EARTHQUAKE SULUT - RANDOM FOREST LITE
# Direct Streamlit deployment, no .pkl / no joblib
# ============================================================

st.set_page_config(
    page_title="FJM AI Earthquake SULUT",
    page_icon="🌏",
    layout="wide"
)

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Candidate area around North Sulawesi, then filtered <=300 km
MIN_LAT, MAX_LAT = -2.5, 6.0
MIN_LON, MAX_LON = 121.0, 129.5
MAX_DISTANCE_SULUT_KM = 300.0

SULUT_REFERENCE_POINTS = [
    {"name":"Manado","lat":1.4748,"lon":124.8421},
    {"name":"Bitung","lat":1.4451,"lon":125.1824},
    {"name":"Tondano","lat":1.3054,"lon":124.9126},
    {"name":"Kotamobagu","lat":0.7337,"lon":124.3124},
    {"name":"Bolaang Mongondow","lat":0.7300,"lon":123.9000},
    {"name":"Sangihe","lat":3.6090,"lon":125.4990},
    {"name":"Talaud","lat":4.1500,"lon":126.7000},
]

TECTONIC_ZONES = [
    {"name":"Molucca Sea Collision Zone","lat":1.50,"lon":126.00},
    {"name":"North Sulawesi Trench","lat":2.25,"lon":124.75},
    {"name":"Sangihe Arc / Eastern SULUT Seismic Zone","lat":2.20,"lon":125.90},
    {"name":"Minahasa–North Sulawesi Crustal Zone","lat":1.15,"lon":124.85},
    {"name":"Southern Molucca Sea Seismic Zone","lat":0.30,"lon":126.10},
]

FEATURES = [
    "event_ordinal","year","month","day","dayofyear",
    "prev_lat","prev_lon","prev_mag","prev_depth","delta_days",
    "rolling7_count","rolling30_count","rolling30_mag_mean",
    "rolling30_mag_max","rolling30_depth_mean"
]

TARGETS = ["target_lat","target_lon","target_mag","target_depth"]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(math.sqrt(a))


def distance_to_sulut(lat, lon):
    vals = [
        (haversine_km(lat,lon,p["lat"],p["lon"]), p["name"])
        for p in SULUT_REFERENCE_POINTS
    ]
    vals.sort(key=lambda x:x[0])
    return vals[0]


def nearest_source(lat, lon):
    vals = [
        (haversine_km(lat,lon,z["lat"],z["lon"]), z["name"])
        for z in TECTONIC_ZONES
    ]
    vals.sort(key=lambda x:x[0])
    return vals[0]


@st.cache_data(ttl=21600, show_spinner=False)
def get_usgs_data(starttime, endtime, min_magnitude):
    params = {
        "format":"geojson",
        "starttime":starttime,
        "endtime":endtime,
        "minlatitude":MIN_LAT,
        "maxlatitude":MAX_LAT,
        "minlongitude":MIN_LON,
        "maxlongitude":MAX_LON,
        "minmagnitude":min_magnitude,
        "orderby":"time-asc",
        "limit":8000,
    }

    r = requests.get(USGS_URL, params=params, timeout=45)
    r.raise_for_status()

    rows = []
    for f in r.json().get("features", []):
        p = f.get("properties", {})
        g = f.get("geometry", {}).get("coordinates", [])

        if len(g) < 3 or p.get("mag") is None:
            continue

        lat = float(g[1])
        lon = float(g[0])
        depth = float(g[2])

        dist, ref = distance_to_sulut(lat, lon)

        if dist <= MAX_DISTANCE_SULUT_KM:
            rows.append({
                "time":pd.to_datetime(p["time"], unit="ms", utc=True).tz_convert(None),
                "place":p.get("place") or "Unknown",
                "mag":float(p["mag"]),
                "depth":depth,
                "lat":lat,
                "lon":lon,
                "distance_sulut_km":dist,
                "nearest_sulut_ref":ref,
            })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["time","lat","lon","mag"])
        .sort_values("time")
        .reset_index(drop=True)
    )


def build_supervised(df):
    d = df.copy().sort_values("time").reset_index(drop=True)

    d["event_ordinal"] = np.arange(len(d), dtype=float)
    d["year"] = d["time"].dt.year
    d["month"] = d["time"].dt.month
    d["day"] = d["time"].dt.day
    d["dayofyear"] = d["time"].dt.dayofyear

    d["prev_lat"] = d["lat"].shift(1)
    d["prev_lon"] = d["lon"].shift(1)
    d["prev_mag"] = d["mag"].shift(1)
    d["prev_depth"] = d["depth"].shift(1)
    d["delta_days"] = d["time"].diff().dt.total_seconds().div(86400).clip(lower=0)

    idx = d.set_index("time")
    d["rolling7_count"] = idx["mag"].rolling("7D", closed="left").count().to_numpy()
    d["rolling30_count"] = idx["mag"].rolling("30D", closed="left").count().to_numpy()
    d["rolling30_mag_mean"] = idx["mag"].rolling("30D", closed="left").mean().to_numpy()
    d["rolling30_mag_max"] = idx["mag"].rolling("30D", closed="left").max().to_numpy()
    d["rolling30_depth_mean"] = idx["depth"].rolling("30D", closed="left").mean().to_numpy()

    d["target_lat"] = d["lat"].shift(-1)
    d["target_lon"] = d["lon"].shift(-1)
    d["target_mag"] = d["mag"].shift(-1)
    d["target_depth"] = d["depth"].shift(-1)

    return d.dropna(subset=FEATURES+TARGETS).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def train_rf_models(supervised):
    if len(supervised) < 100:
        raise ValueError("Data terlalu sedikit untuk training Random Forest.")

    split = int(len(supervised)*0.80)
    split = min(max(split, 50), len(supervised)-20)

    train = supervised.iloc[:split]
    test = supervised.iloc[split:]

    models = {}
    metrics = {}

    for target in TARGETS:
        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )
        model.fit(train[FEATURES], train[target])

        pred = model.predict(test[FEATURES])
        models[target] = model
        metrics[target] = float(mean_absolute_error(test[target], pred))

    return models, metrics


def make_prediction_features(df, selected_date):
    ts = pd.Timestamp(selected_date)
    d = df.sort_values("time").reset_index(drop=True)
    prior = d[d["time"] <= ts]

    if prior.empty:
        prior = d.iloc[[0]]

    last = prior.iloc[-1]
    recent7 = prior[prior["time"] > ts - pd.Timedelta(days=7)]
    recent30 = prior[prior["time"] > ts - pd.Timedelta(days=30)]

    if len(prior) >= 2:
        prev = prior.iloc[-2]
        delta_days = max((last["time"] - prev["time"]).total_seconds()/86400.0, 0.0)
    else:
        delta_days = 0.0

    row = {
        "event_ordinal":float(len(prior)),
        "year":ts.year,
        "month":ts.month,
        "day":ts.day,
        "dayofyear":ts.dayofyear,
        "prev_lat":float(last["lat"]),
        "prev_lon":float(last["lon"]),
        "prev_mag":float(last["mag"]),
        "prev_depth":float(last["depth"]),
        "delta_days":float(delta_days),
        "rolling7_count":float(len(recent7)),
        "rolling30_count":float(len(recent30)),
        "rolling30_mag_mean":float(recent30["mag"].mean() if len(recent30) else last["mag"]),
        "rolling30_mag_max":float(recent30["mag"].max() if len(recent30) else last["mag"]),
        "rolling30_depth_mean":float(recent30["depth"].mean() if len(recent30) else last["depth"]),
    }

    return pd.DataFrame([row], columns=FEATURES)


st.title("🌏 FJM AI Earthquake SULUT")
st.caption("Random Forest Lite | Radius Analisis ≤ 300 km dari Sulawesi Utara")

with st.expander("⚠️ Catatan ilmiah"):
    st.info(
        "Aplikasi menggunakan Random Forest sebagai model AI. "
        "Output merupakan experimental machine-learning forecast, bukan prediksi deterministik "
        "bahwa gempa pasti terjadi pada tanggal/lokasi/magnitude tertentu."
    )

with st.sidebar:
    st.header("Parameter")
    start_year = st.number_input("Tahun awal data", 2005, 2025, 2010, 1)
    min_mag = st.slider("Minimum magnitude katalog", 3.0, 5.0, 4.0, 0.1)

try:
    with st.spinner("Mengambil katalog USGS..."):
        df = get_usgs_data(f"{int(start_year)}-01-01", "2025-12-31", min_mag)
except Exception as e:
    st.error(f"Gagal mengambil data USGS: {e}")
    st.stop()

if df.empty:
    st.warning("Tidak ada data gempa.")
    st.stop()

supervised = build_supervised(df)

if len(supervised) < 100:
    st.error("Data training terlalu sedikit. Turunkan minimum magnitude atau gunakan tahun awal lebih lama.")
    st.stop()

with st.spinner("Menyiapkan model AI Random Forest..."):
    models, metrics = train_rf_models(supervised)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Jumlah Event", f"{len(df):,}")
c2.metric("Radius", "≤300 km")
c3.metric("Magnitude Maks.", f"M {df['mag'].max():.1f}")
c4.metric("Model AI", "Random Forest")

selected_date = st.date_input("Pilih tanggal analisis/prediksi", value=date.today())

if st.button("PREDIKSI GEMPA DENGAN AI", type="primary", use_container_width=True):
    x = make_prediction_features(df, selected_date)

    lat = float(models["target_lat"].predict(x)[0])
    lon = float(models["target_lon"].predict(x)[0])
    mag = float(np.clip(models["target_mag"].predict(x)[0], 0, 9.5))
    depth = float(np.clip(models["target_depth"].predict(x)[0], 0, 700))

    sulut_dist, nearest_ref = distance_to_sulut(lat, lon)
    source_dist, source_name = nearest_source(lat, lon)

    st.session_state["rf_result"] = {
        "lat":lat,
        "lon":lon,
        "mag":mag,
        "depth":depth,
        "distance_sulut":sulut_dist,
        "nearest_ref":nearest_ref,
        "source_name":source_name,
        "source_distance":source_dist,
    }

r = st.session_state.get("rf_result")

if r:
    a,b,c,d,e = st.columns(5)
    a.metric("Prediksi Magnitude", f"M {r['mag']:.2f}")
    b.metric("Prediksi Depth", f"{r['depth']:.1f} km")
    c.metric("Latitude", f"{r['lat']:.4f}°")
    d.metric("Longitude", f"{r['lon']:.4f}°")
    e.metric("Jarak dari SULUT", f"{r['distance_sulut']:.1f} km")

    if r["distance_sulut"] <= MAX_DISTANCE_SULUT_KM:
        st.success(
            f"**Dalam radius analisis ≤300 km dari SULUT**  \n"
            f"Sumber tektonik terdekat (aproksimasi): **{r['source_name']}**  \n"
            f"Jarak ke zona sumber: **{r['source_distance']:.1f} km**"
        )
    else:
        st.warning(
            f"Hasil model berada {r['distance_sulut']:.1f} km dari SULUT "
            "dan berada di luar radius analisis 300 km."
        )

    point = pd.DataFrame([{
        "lat":r["lat"],
        "lon":r["lon"],
        "Magnitude":r["mag"],
        "Depth":r["depth"],
        "Source":r["source_name"],
        "DistanceSulut":r["distance_sulut"],
        "Label":(
            f"★ PREDIKSI AI\n"
            f"M {r['mag']:.2f} | {r['depth']:.0f} km\n"
            f"{r['lat']:.4f}, {r['lon']:.4f}\n"
            f"{r['source_name']}"
        )
    }])

    hist = df.tail(500).copy()
    hist["radius"] = np.clip((hist["mag"]**3)*120, 800, 16000)

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=hist,
            get_position="[lon, lat]",
            get_fill_color="[70,110,170,70]",
            get_radius="radius",
            radius_min_pixels=1,
            radius_max_pixels=7,
            pickable=True
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=point,
            get_position="[lon, lat]",
            get_fill_color="[255,190,0,245]",
            get_line_color="[0,0,0,255]",
            get_radius=15000,
            radius_min_pixels=17,
            radius_max_pixels=34,
            stroked=True,
            filled=True,
            line_width_min_pixels=4,
            pickable=True
        ),
        pdk.Layer(
            "TextLayer",
            data=point,
            get_position="[lon, lat]",
            get_text="'★'",
            get_size=38,
            get_color="[0,0,0,255]",
            get_text_anchor="'middle'",
            get_alignment_baseline="'center'",
            billboard=True
        ),
        pdk.Layer(
            "TextLayer",
            data=point,
            get_position="[lon, lat]",
            get_text="Label",
            get_size=15,
            get_color="[0,0,0,255]",
            get_pixel_offset="[0,-80]",
            get_text_anchor="'middle'",
            get_alignment_baseline="'bottom'",
            billboard=True
        )
    ]

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(
                latitude=r["lat"],
                longitude=r["lon"],
                zoom=7.6,
                pitch=20
            ),
            map_style=None
        ),
        use_container_width=True
    )

with st.expander("Validasi Model"):
    st.write("MAE temporal hold-out:")
    st.dataframe(pd.DataFrame({
        "Target":["Latitude","Longitude","Magnitude","Depth"],
        "MAE":[metrics["target_lat"],metrics["target_lon"],metrics["target_mag"],metrics["target_depth"]]
    }), use_container_width=True, hide_index=True)

with st.expander("Data Historis"):
    show = df.sort_values("time", ascending=False).copy()
    show["time"] = show["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(
        show[["time","place","mag","depth","lat","lon","distance_sulut_km","nearest_sulut_ref"]],
        use_container_width=True,
        hide_index=True
    )
