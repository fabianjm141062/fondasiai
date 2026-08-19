import math
from datetime import date
Pld
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="FJM AI Earthquake SULUT", page_icon="🌏", layout="wide")

MIN_LAT, MAX_LAT = -2.5, 6.0
MIN_LON, MAX_LON = 121.0, 129.5
MAX_DISTANCE_SULUT_KM = 300.0
USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

FEATURES = [
    "event_ordinal","year","month","day","dayofyear",
    "prev_lat","prev_lon","prev_mag","prev_depth","delta_days",
    "rolling7_count","rolling30_count","rolling30_mag_mean",
    "rolling30_mag_max","rolling30_depth_mean"
]
TARGETS = ["target_lat","target_lon","target_mag","target_depth"]

TECTONIC_ZONES = [
    {"name":"Molucca Sea Collision Zone","lat":1.50,"lon":126.00},
    {"name":"North Sulawesi Trench","lat":2.25,"lon":124.75},
    {"name":"Sangihe Arc / Eastern SULUT Seismic Zone","lat":2.20,"lon":125.90},
    {"name":"Minahasa–North Sulawesi Crustal Zone","lat":1.15,"lon":124.85},
    {"name":"Southern Molucca Sea Seismic Zone","lat":0.30,"lon":126.10},
]

# Representative points covering North Sulawesi.
# An event is included when its minimum great-circle distance
# to one of these points is <= 300 km.
SULUT_REFERENCE_POINTS = [
    {"name":"Manado", "lat":1.4748, "lon":124.8421},
    {"name":"Bitung", "lat":1.4451, "lon":125.1824},
    {"name":"Tondano", "lat":1.3054, "lon":124.9126},
    {"name":"Kotamobagu", "lat":0.7337, "lon":124.3124},
    {"name":"Bolaang Mongondow", "lat":0.7300, "lon":123.9000},
    {"name":"Sangihe", "lat":3.6090, "lon":125.4990},
    {"name":"Talaud", "lat":4.1500, "lon":126.7000},
]

def distance_to_sulut_km(lat, lon):
    return min(
        haversine_km(lat, lon, p["lat"], p["lon"])
        for p in SULUT_REFERENCE_POINTS
    )


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(math.sqrt(a))

def nearest_tectonic_zone(lat, lon):
    ranked = [(haversine_km(lat,lon,z["lat"],z["lon"]), z["name"]) for z in TECTONIC_ZONES]
    ranked.sort(key=lambda x:x[0])
    return ranked[0][1], ranked[0][0]

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_usgs_data(starttime, endtime, minmag):
    params = {
        "format":"geojson","starttime":starttime,"endtime":endtime,
        "minlatitude":MIN_LAT,"maxlatitude":MAX_LAT,
        "minlongitude":MIN_LON,"maxlongitude":MAX_LON,
        "minmagnitude":minmag,"orderby":"time-asc","limit":20000
    }
    r = requests.get(USGS_URL, params=params, timeout=60)
    r.raise_for_status()
    rows = []
    for f in r.json().get("features", []):
        p = f.get("properties", {})
        g = f.get("geometry", {}).get("coordinates", [None,None,None])
        if p.get("mag") is None or len(g) < 3:
            continue
        rows.append({
            "time":pd.to_datetime(p["time"], unit="ms", utc=True).tz_convert(None),
            "place":p.get("place") or "Unknown",
            "mag":float(p["mag"]),"lat":float(g[1]),"lon":float(g[0]),"depth":float(g[2])
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = (
        df.dropna()
          .drop_duplicates(subset=["time","lat","lon","mag"])
          .sort_values("time")
          .reset_index(drop=True)
    )

    df["distance_sulut_km"] = df.apply(
        lambda r: distance_to_sulut_km(r["lat"], r["lon"]),
        axis=1
    )

    df = (
        df[df["distance_sulut_km"] <= MAX_DISTANCE_SULUT_KM]
        .sort_values("time")
        .reset_index(drop=True)
    )
    return df

def build_supervised_dataset(df):
    d = df.copy().sort_values("time").reset_index(drop=True)
    d["event_ordinal"] = np.arange(len(d), dtype=float)
    d["year"], d["month"], d["day"] = d["time"].dt.year, d["time"].dt.month, d["time"].dt.day
    d["dayofyear"] = d["time"].dt.dayofyear
    d["prev_lat"], d["prev_lon"] = d["lat"].shift(1), d["lon"].shift(1)
    d["prev_mag"], d["prev_depth"] = d["mag"].shift(1), d["depth"].shift(1)
    d["delta_days"] = d["time"].diff().dt.total_seconds().div(86400).clip(lower=0)

    idx = d.set_index("time")
    d["rolling7_count"] = idx["mag"].rolling("7D", closed="left").count().to_numpy()
    d["rolling30_count"] = idx["mag"].rolling("30D", closed="left").count().to_numpy()
    d["rolling30_mag_mean"] = idx["mag"].rolling("30D", closed="left").mean().to_numpy()
    d["rolling30_mag_max"] = idx["mag"].rolling("30D", closed="left").max().to_numpy()
    d["rolling30_depth_mean"] = idx["depth"].rolling("30D", closed="left").mean().to_numpy()

    d["target_lat"], d["target_lon"] = d["lat"].shift(-1), d["lon"].shift(-1)
    d["target_mag"], d["target_depth"] = d["mag"].shift(-1), d["depth"].shift(-1)
    return d.dropna(subset=FEATURES+TARGETS).reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def train_models(supervised):
    if len(supervised) < 100:
        raise ValueError("Data terlalu sedikit untuk training.")
    split = int(len(supervised)*0.80)
    split = min(max(split,50), len(supervised)-20)
    train, test = supervised.iloc[:split], supervised.iloc[split:]
    models, metrics = {}, {}
    for target in TARGETS:
        m = RandomForestRegressor(
            n_estimators=500,max_depth=18,min_samples_leaf=2,
            max_features="sqrt",random_state=42,n_jobs=-1
        )
        m.fit(train[FEATURES], train[target])
        pred = m.predict(test[FEATURES])
        models[target] = m
        metrics[target] = {
            "MAE":float(mean_absolute_error(test[target], pred)),
            "R2":float(r2_score(test[target], pred))
        }
    return models, metrics

def make_prediction_features(df, selected_date):
    d = df.sort_values("time").reset_index(drop=True)
    ts = pd.Timestamp(selected_date)
    prior = d[d["time"] <= ts]
    if prior.empty:
        prior = d.iloc[[0]]
    last = prior.iloc[-1]
    recent7 = prior[prior["time"] > ts-pd.Timedelta(days=7)]
    recent30 = prior[prior["time"] > ts-pd.Timedelta(days=30)]
    if len(prior) >= 2:
        prev = prior.iloc[-2]
        delta_days = max((last["time"]-prev["time"]).total_seconds()/86400.0, 0.0)
    else:
        delta_days = 0.0
    row = {
        "event_ordinal":float(len(prior)),"year":ts.year,"month":ts.month,"day":ts.day,
        "dayofyear":ts.dayofyear,"prev_lat":float(last["lat"]),"prev_lon":float(last["lon"]),
        "prev_mag":float(last["mag"]),"prev_depth":float(last["depth"]),"delta_days":float(delta_days),
        "rolling7_count":float(len(recent7)),"rolling30_count":float(len(recent30)),
        "rolling30_mag_mean":float(recent30["mag"].mean() if len(recent30) else last["mag"]),
        "rolling30_mag_max":float(recent30["mag"].max() if len(recent30) else last["mag"]),
        "rolling30_depth_mean":float(recent30["depth"].mean() if len(recent30) else last["depth"]),
    }
    return pd.DataFrame([row], columns=FEATURES)

def tree_spread(model, x):
    vals = np.array([t.predict(x)[0] for t in model.estimators_], dtype=float)
    return float(np.std(vals))

st.title("🌏 FJM AI Earthquake SULUT")
st.caption("Experimental Random Forest seismic forecasting dashboard | Prof. Dr. Fabian J. Manoppo – FJM AI Data Analyst")

with st.expander("⚠️ Scientific scope / disclaimer"):
    st.info(
        "Output adalah forecast eksperimental berbasis katalog dan Random Forest, bukan ramalan deterministik. "
        "Label sumber adalah zona tektonik terdekat secara aproksimasi, bukan penetapan sumber resmi."
    )

with st.sidebar:
    st.header("Parameter Data")
    start_year = st.number_input("Tahun awal", 1900, 2024, 1900, 10)
    end_year = st.number_input("Tahun akhir", 2000, 2026, 2025, 1)
    minmag = st.slider("Minimum magnitude katalog", 2.5, 5.0, 3.5, 0.1)
    show_history = st.checkbox("Tampilkan gempa historis", True)
    st.caption(f"Filter wilayah: maksimum {MAX_DISTANCE_SULUT_KM:.0f} km dari SULUT")

try:
    with st.spinner("Mengambil katalog USGS dan menyiapkan model..."):
        df = fetch_usgs_data(f"{int(start_year)}-01-01", f"{int(end_year)}-12-31", minmag)
except Exception as e:
    st.error(f"Gagal mengambil data USGS: {e}")
    st.stop()

if df.empty:
    st.warning("Tidak ada data.")
    st.stop()

if len(df) >= 20000:
    st.warning("Query mencapai batas 20.000 event. Naikkan minimum magnitude atau pecah rentang waktu.")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Jumlah Event", f"{len(df):,}")
c2.metric("Magnitude Maks.", f"M {df['mag'].max():.1f}")
c3.metric("Depth Median", f"{df['depth'].median():.1f} km")
c4.metric("Periode", f"{df['time'].min().year}–{df['time'].max().year}")

supervised = build_supervised_dataset(df)
if len(supervised) < 100:
    st.error("Data hasil feature engineering terlalu sedikit.")
    st.stop()

models, metrics = train_models(supervised)
tab1,tab2,tab3,tab4 = st.tabs(["🔮 Prediksi AI","🗺️ Data Historis","📊 Validasi Model","ℹ️ Metodologi"])

with tab1:
    selected_date = st.date_input("Pilih tanggal analisis/prediksi", value=date.today())
    if st.button("PREDIKSI GEMPA", type="primary", use_container_width=True):
        x = make_prediction_features(df, selected_date)
        lat = float(np.clip(models["target_lat"].predict(x)[0], MIN_LAT, MAX_LAT))
        lon = float(np.clip(models["target_lon"].predict(x)[0], MIN_LON, MAX_LON))
        mag = float(np.clip(models["target_mag"].predict(x)[0], 0, 9.5))
        depth = float(np.clip(models["target_depth"].predict(x)[0], 0, 700))
        source, dist = nearest_tectonic_zone(lat, lon)
        sulut_dist = distance_to_sulut_km(lat, lon)
        in_domain = sulut_dist <= MAX_DISTANCE_SULUT_KM

        st.session_state["prediction"] = {
            "date":str(selected_date),"lat":lat,"lon":lon,"mag":mag,"depth":depth,
            "source":source,"dist":dist,"sulut_dist":sulut_dist,"in_domain":in_domain,
            "mag_spread":tree_spread(models["target_mag"],x),
            "depth_spread":tree_spread(models["target_depth"],x),
        }

    r = st.session_state.get("prediction")
    if r:
        a,b,c,d,e = st.columns(5)
        a.metric("Magnitude", f"M {r['mag']:.2f}", help=f"Tree spread ±{r['mag_spread']:.2f}")
        b.metric("Depth", f"{r['depth']:.1f} km", help=f"Tree spread ±{r['depth_spread']:.1f} km")
        c.metric("Latitude", f"{r['lat']:.4f}°")
        d.metric("Longitude", f"{r['lon']:.4f}°")
        e.metric("Jarak dari SULUT", f"{r['sulut_dist']:.1f} km")

        if r["in_domain"]:
            st.success(
                f"**Dalam radius analisis ≤ {MAX_DISTANCE_SULUT_KM:.0f} km dari SULUT**  \n"
                f"Zona sumber terdekat (aproksimasi): **{r['source']}**  \n"
                f"Jarak ke anchor zona tektonik: **{r['dist']:.1f} km**"
            )
        else:
            st.warning(
                f"Hasil RF berada **{r['sulut_dist']:.1f} km dari SULUT**, "
                f"di luar batas analisis {MAX_DISTANCE_SULUT_KM:.0f} km."
            )

        pred = pd.DataFrame([{
            "lat":r["lat"],"lon":r["lon"],"Magnitude":r["mag"],"Depth":r["depth"],"Source":r["source"],"DistanceSulut":r["sulut_dist"],
            "Label":f"★ PREDIKSI AI\nM {r['mag']:.2f} | {r['depth']:.0f} km\n{r['lat']:.4f}, {r['lon']:.4f}\n{r['source']}"
        }])

        layers = []
        if show_history:
            hist = df.copy()
            hist["radius"] = np.clip((hist["mag"]**3)*130, 800, 18000)
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=hist, get_position="[lon, lat]",
                get_fill_color="[80, 120, 180, 85]", get_radius="radius",
                radius_min_pixels=1, radius_max_pixels=8, pickable=True
            ))

        layers += [
            pdk.Layer(
                "ScatterplotLayer", data=pred, get_position="[lon, lat]",
                get_fill_color="[255, 190, 0, 245]", get_line_color="[10,10,10,255]",
                get_radius=15000, radius_min_pixels=16, radius_max_pixels=34,
                stroked=True, filled=True, line_width_min_pixels=4, pickable=True
            ),
            pdk.Layer(
                "TextLayer", data=pred, get_position="[lon, lat]", get_text="'★'",
                get_size=38, get_color="[0,0,0,255]", get_text_anchor="'middle'",
                get_alignment_baseline="'center'", billboard=True
            ),
            pdk.Layer(
                "TextLayer", data=pred, get_position="[lon, lat]", get_text="Label",
                get_size=15, get_color="[0,0,0,255]", get_pixel_offset="[0,-85]",
                get_text_anchor="'middle'", get_alignment_baseline="'bottom'", billboard=True
            )
        ]

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=r["lat"], longitude=r["lon"], zoom=7.8, pitch=25),
            map_style=None,
            tooltip={"html":"<b>★ PREDIKSI AI</b><br/>Magnitude: <b>{Magnitude}</b><br/>Depth: <b>{Depth} km</b><br/>Latitude: <b>{lat}</b><br/>Longitude: <b>{lon}</b><br/>Jarak dari SULUT: <b>{DistanceSulut} km</b><br/>Zona sumber: <b>{Source}</b>"}
        )
        st.pydeck_chart(deck, use_container_width=True)

with tab2:
    dshow = df.sort_values("time", ascending=False).copy()
    dshow["time"] = dshow["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(dshow[["time","place","mag","depth","lat","lon","distance_sulut_km"]], use_container_width=True, hide_index=True)

with tab3:
    st.write("Validasi temporal: 80% event awal untuk training, 20% event terakhir untuk testing.")
    labels = {"target_lat":"Latitude","target_lon":"Longitude","target_mag":"Magnitude","target_depth":"Depth"}
    rows = [{"Target":labels[t],"MAE":metrics[t]["MAE"],"R²":metrics[t]["R2"]} for t in TARGETS]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    imp = pd.DataFrame({"Feature":FEATURES,"Importance":models["target_mag"].feature_importances_}).sort_values("Importance", ascending=False)
    st.bar_chart(imp.set_index("Feature"))

with tab4:
    st.markdown("""
### Pipeline
1. USGS FDSN earthquake catalog.
2. Data cleaning dan temporal ordering.
3. Feature engineering: event sebelumnya, interval waktu, rolling seismicity 7/30 hari.
4. Random Forest Regressor untuk latitude, longitude, magnitude, dan depth event katalog berikutnya.
5. `random_state=42` untuk reproducibility.
6. Temporal hold-out untuk validasi.
7. Titik prediksi ditandai ★ besar.
8. Sumber gempa = nearest tectonic-zone approximation.

### Catatan
Untuk publikasi/engineering final, ganti anchor zona tektonik dengan geometri sumber gempa resmi (GeoJSON/shapefile) dan gunakan PSHA/GMPE serta code desain yang berlaku.
""")
