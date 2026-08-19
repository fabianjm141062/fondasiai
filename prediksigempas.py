import math
from datetime import date
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

st.set_page_config(page_title="FJM Gempa SULUT", page_icon="🌏", layout="wide")

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
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

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(math.sqrt(a))

def distance_to_sulut(lat, lon):
    d = [(haversine_km(lat,lon,p["lat"],p["lon"]), p["name"]) for p in SULUT_REFERENCE_POINTS]
    d.sort(key=lambda x:x[0])
    return d[0]

def nearest_source(lat, lon):
    d = [(haversine_km(lat,lon,z["lat"],z["lon"]), z["name"]) for z in TECTONIC_ZONES]
    d.sort(key=lambda x:x[0])
    return d[0]

@st.cache_data(ttl=21600, show_spinner=False)
def get_usgs_data(starttime, endtime, min_magnitude):
    params = {
        "format":"geojson","starttime":starttime,"endtime":endtime,
        "minlatitude":MIN_LAT,"maxlatitude":MAX_LAT,
        "minlongitude":MIN_LON,"maxlongitude":MAX_LON,
        "minmagnitude":min_magnitude,"orderby":"time-asc","limit":10000
    }
    r = requests.get(USGS_URL, params=params, timeout=45)
    r.raise_for_status()
    rows = []
    for f in r.json().get("features", []):
        p = f.get("properties", {})
        g = f.get("geometry", {}).get("coordinates", [])
        if len(g) < 3 or p.get("mag") is None:
            continue
        lat, lon, depth = float(g[1]), float(g[0]), float(g[2])
        dist, ref = distance_to_sulut(lat, lon)
        if dist <= MAX_DISTANCE_SULUT_KM:
            rows.append({
                "time":pd.to_datetime(p["time"], unit="ms", utc=True).tz_convert(None),
                "place":p.get("place") or "Unknown",
                "mag":float(p["mag"]),"depth":depth,"lat":lat,"lon":lon,
                "distance_sulut_km":dist,"nearest_sulut_ref":ref
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["time","lat","lon","mag"]).sort_values("time").reset_index(drop=True)

def weighted_forecast(df, selected_date, n_events=50):
    ts = pd.Timestamp(selected_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    prior = df[df["time"] <= ts].copy()
    if len(prior) < 5:
        return None
    sample = prior.tail(min(n_events, len(prior))).copy()
    max_time = sample["time"].max()
    age_days = (max_time - sample["time"]).dt.total_seconds()/86400.0
    recency_weight = 1.0/(1.0 + age_days/30.0)
    mag_weight = sample["mag"].clip(lower=1.0)**2
    w = recency_weight * mag_weight
    sw = w.sum()
    lat = float((sample["lat"]*w).sum()/sw)
    lon = float((sample["lon"]*w).sum()/sw)
    mag = float((sample["mag"]*w).sum()/sw)
    depth = float((sample["depth"]*w).sum()/sw)
    recent30 = prior[prior["time"] > ts-pd.Timedelta(days=30)]
    activity = "TINGGI" if len(recent30)>=20 else ("SEDANG" if len(recent30)>=8 else "RENDAH")
    sulut_dist, nearest_ref = distance_to_sulut(lat,lon)
    source_dist, source_name = nearest_source(lat,lon)
    return {"lat":lat,"lon":lon,"mag":mag,"depth":depth,"activity":activity,
            "events_30d":len(recent30),"distance_sulut":sulut_dist,
            "nearest_ref":nearest_ref,"source_name":source_name,"source_distance":source_dist}

st.title("🌏 FJM Gempa SULUT — Streamlit Lite")
st.caption("Analisis Statistik Deterministik | Radius maksimum 300 km dari Sulawesi Utara")

with st.sidebar:
    start_year = st.number_input("Tahun awal data", 2000, 2025, 2000, 1)
    min_mag = st.slider("Minimum magnitude katalog", 3.0, 5.0, 4.0, 0.1)
    n_events = st.slider("Jumlah event terakhir", 20, 150, 50, 10)

try:
    with st.spinner("Mengambil data USGS..."):
        df = get_usgs_data(f"{int(start_year)}-01-01", "2025-12-31", min_mag)
except Exception as e:
    st.error(f"Gagal mengambil data USGS: {e}")
    st.stop()

if df.empty:
    st.warning("Tidak ada data.")
    st.stop()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Jumlah Event", f"{len(df):,}")
c2.metric("Radius", "≤300 km")
c3.metric("Magnitude Maks.", f"M {df['mag'].max():.1f}")
c4.metric("Periode", f"{df['time'].min().year}–{df['time'].max().year}")

selected_date = st.date_input("Pilih tanggal analisis", value=date.today())

if st.button("HITUNG ESTIMASI GEMPA", type="primary", use_container_width=True):
    st.session_state["result"] = weighted_forecast(df, selected_date, n_events)

r = st.session_state.get("result")
if r:
    a,b,c,d,e = st.columns(5)
    a.metric("Estimasi Magnitude", f"M {r['mag']:.2f}")
    b.metric("Estimasi Depth", f"{r['depth']:.1f} km")
    c.metric("Latitude", f"{r['lat']:.4f}°")
    d.metric("Longitude", f"{r['lon']:.4f}°")
    e.metric("Jarak dari SULUT", f"{r['distance_sulut']:.1f} km")
    st.success(f"Aktivitas 30 hari: **{r['activity']}** | Sumber terdekat: **{r['source_name']}** | Jarak zona: **{r['source_distance']:.1f} km**")

    point = pd.DataFrame([{
        "lat":r["lat"],"lon":r["lon"],"Magnitude":r["mag"],"Depth":r["depth"],
        "Source":r["source_name"],"DistanceSulut":r["distance_sulut"],
        "Label":f"★ ESTIMASI\nM {r['mag']:.2f} | {r['depth']:.0f} km\n{r['lat']:.4f}, {r['lon']:.4f}\n{r['source_name']}"
    }])

    hist = df.tail(500).copy()
    hist["radius"] = hist["mag"].apply(lambda x:max(800,min(16000,(x**3)*120)))

    layers = [
        pdk.Layer("ScatterplotLayer",data=hist,get_position="[lon, lat]",get_fill_color="[70,110,170,70]",get_radius="radius",radius_min_pixels=1,radius_max_pixels=7,pickable=True),
        pdk.Layer("ScatterplotLayer",data=point,get_position="[lon, lat]",get_fill_color="[255,190,0,245]",get_line_color="[0,0,0,255]",get_radius=15000,radius_min_pixels=17,radius_max_pixels=34,stroked=True,filled=True,line_width_min_pixels=4,pickable=True),
        pdk.Layer("TextLayer",data=point,get_position="[lon, lat]",get_text="'★'",get_size=38,get_color="[0,0,0,255]",get_text_anchor="'middle'",get_alignment_baseline="'center'",billboard=True),
        pdk.Layer("TextLayer",data=point,get_position="[lon, lat]",get_text="Label",get_size=15,get_color="[0,0,0,255]",get_pixel_offset="[0,-80]",get_text_anchor="'middle'",get_alignment_baseline="'bottom'",billboard=True),
    ]

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=r["lat"],longitude=r["lon"],zoom=7.6,pitch=20),
            map_style=None,
        ),
        use_container_width=True
    )

with st.expander("Data Historis"):
    show = df.sort_values("time", ascending=False).copy()
    show["time"] = show["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(show[["time","place","mag","depth","lat","lon","distance_sulut_km","nearest_sulut_ref"]], use_container_width=True, hide_index=True)

st.caption("Catatan: versi Lite tidak menggunakan Random Forest. Output adalah estimasi statistik deterministik, bukan prediksi pasti kejadian gempa.")
