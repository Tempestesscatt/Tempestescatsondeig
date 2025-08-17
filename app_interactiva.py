# -*- coding: utf-8 -*-
import streamlit as st
import openmeteo_requests
import requests_cache
from retry_requests import retry
import requests
import numpy as np
import time
import matplotlib.pyplot as plt
from metpy.plots import SkewT, Hodograph
from metpy.units import units
import metpy.calc as mpcalc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata, Rbf
from datetime import datetime, timedelta
import pytz
import matplotlib.patheffects as path_effects
from matplotlib.colors import ListedColormap, BoundaryNorm

# --- CONFIGURACIÓ INICIAL ---
st.set_page_config(layout="wide", page_title="Terminal de Temps Sever | Catalunya")
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)
FORECAST_DAYS = 2

# --- DADES DE CIUTATS PRINCIPALS ---
ciutats_catalunya = {
    'Barcelona': {'lat': 41.3851, 'lon': 2.1734},
    'Girona': {'lat': 41.9831, 'lon': 2.8249},
    'Lleida': {'lat': 41.6177, 'lon': 0.6200},
    'Tarragona': {'lat': 41.1189, 'lon': 1.2445},
}

# Llista de nivells de pressió per als mapes de vents i sondejos
PRESS_LEVELS = sorted([1000, 950, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)

# --- INICIALITZACIÓ ---
if 'poble_selector' not in st.session_state: st.session_state.poble_selector = 'Barcelona'
if 'dia_selector' not in st.session_state: st.session_state.dia_selector = 'Avui'
if 'hora_selector' not in st.session_state: st.session_state.hora_selector = f"{datetime.now(pytz.timezone('Europe/Madrid')).hour:02d}:00h"

# --- 1. LÒGICA DE CÀRREGA I CÀLCUL ---
@st.cache_data(ttl=16000)
def carregar_dades_completes(lat, lon, hourly_index):
    p_levels = PRESS_LEVELS
    h_base = ["temperature_2m", "dew_point_2m", "surface_pressure", "cape"]
    h_press = [f"{v}_{p}hPa" for v in ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"] for p in p_levels]
    params = {"latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "arome_seamless", "forecast_days": FORECAST_DAYS}
    try:
        response = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
        hourly = response.Hourly()
        sfc_data = {v: hourly.Variables(i).ValuesAsNumpy()[hourly_index] for i, v in enumerate(h_base)}
        if any(np.isnan(val) for val in sfc_data.values()): return None, "Dades de superfície invàlides."
        p_data = {}
        var_count = len(h_base)
        for i, var in enumerate(["T", "RH", "WS", "WD", "H"]):
            p_data[var] = [hourly.Variables(var_count + i * len(p_levels) + j).ValuesAsNumpy()[hourly_index] for j in range(len(p_levels))]
        sfc_h = mpcalc.pressure_to_height_std(sfc_data["surface_pressure"] * units.hPa).to('meter').m
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = \
            [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]], [0.0], [0.0], [sfc_h]
        for i, p_val in enumerate(p_levels):
            if p_val < sfc_data["surface_pressure"] and all(not np.isnan(p_data[v][i]) for v in ["T", "RH", "WS", "WD", "H"]):
                p_profile.append(p_val); T_profile.append(p_data["T"][i])
                Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["T"][i] * units.degC, p_data["RH"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["WS"][i] * units('km/h'), p_data["WD"][i] * units.degrees)
                u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m); h_profile.append(p_data["H"][i])
        if len(p_profile) < 4: return None, "Perfil atmosfèric massa curt."
        p=np.array(p_profile)*units.hPa; T=np.array(T_profile)*units.degC; Td=np.array(Td_profile)*units.degC
        u=np.array(u_profile)*units('m/s'); v=np.array(v_profile)*units('m/s'); h=np.array(h_profile)*units.meter
        params_calc = {}
        prof = mpcalc.parcel_profile(p, T[0], Td[0]); cape, cin = mpcalc.cape_cin(p, T, Td, prof)
        params_calc['CAPE'] = cape.to('J/kg').m if cape.magnitude > 0 else 0; params_calc['CIN'] = cin.to('J/kg').m
        s_u, s_v = mpcalc.bulk_shear(p, u, v, height=h, depth=6*units.km); params_calc['Shear_0-6km'] = mpcalc.wind_speed(s_u, s_v).to('m/s').m
        _, srh, _ = mpcalc.storm_relative_helicity(h, u, v, depth=1*units.km); params_calc['SRH_0-1km'] = np.sum(srh).to('m^2/s^2').m
        return ((p, T, Td, u, v, h), params_calc), None
    except Exception as e: return None, f"Error processant dades: {e}"

@st.cache_data(ttl=16000)
def obtener_dades_mapa(variable, hourly_index):
    url = "https://api.open-meteo.com/v1/forecast"; lats, lons = np.linspace(40.4, 43, 12), np.linspace(0, 3.5, 12)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": [variable], "models": "arome_seamless", "forecast_days": FORECAST_DAYS}
    try:
        responses = openmeteo.weather_api(url, params=params)
        lats_out, lons_out, data_out = [], [], []
        for r in responses:
            val = r.Hourly().Variables(0).ValuesAsNumpy()[hourly_index]
            if not np.isnan(val): lats_out.append(r.Latitude()); lons_out.append(r.Longitude()); data_out.append(val)
        if not lats_out: return None, "No s'han rebut dades."
        return (lats_out, lons_out, data_out), None
    except Exception as e: return None, str(e)

@st.cache_data(ttl=16000)
def obtener_dades_vents_mapa(nivell, hourly_index):
    url = "https://api.open-meteo.com/v1/forecast"; lats, lons = np.linspace(40.4, 43, 12), np.linspace(0, 3.5, 12)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"], "models": "arome_seamless", "forecast_days": FORECAST_DAYS}
    try:
        responses = openmeteo.weather_api(url, params=params)
        lats_out, lons_out, speed_out, dir_out = [], [], [], []
        for r in responses:
            speed, direction = r.Hourly().Variables(0).ValuesAsNumpy()[hourly_index], r.Hourly().Variables(1).ValuesAsNumpy()[hourly_index]
            if not np.isnan(speed) and not np.isnan(direction): lats_out.append(r.Latitude()); lons_out.append(r.Longitude()); speed_out.append(speed); dir_out.append(direction)
        if not lats_out: return None, "No s'han rebut dades de vent."
        return (lats_out, lons_out, (speed_out, dir_out)), None
    except Exception as e: return None, str(e)

@st.cache_data(ttl=16000)
def obtener_dades_mapa_500hpa(hourly_index):
    url = "https://api.open-meteo.com/v1/forecast"; lats, lons = np.linspace(40.4, 43, 12), np.linspace(0, 3.5, 12)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    variables = ["geopotential_height_500hPa", "temperature_500hPa", "wind_speed_500hPa", "wind_direction_500hPa"]
    params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "arome_seamless", "forecast_days": FORECAST_DAYS}
    try:
        responses = openmeteo.weather_api(url, params=params)
        lats_out, lons_out, geo_out, temp_out, ws_out, wd_out = [], [], [], [], [], []
        for r in responses:
            geo = r.Hourly().Variables(0).ValuesAsNumpy()[hourly_index]; temp = r.Hourly().Variables(1).ValuesAsNumpy()[hourly_index]
            ws = r.Hourly().Variables(2).ValuesAsNumpy()[hourly_index]; wd = r.Hourly().Variables(3).ValuesAsNumpy()[hourly_index]
            if not any(np.isnan([geo, temp, ws, wd])):
                lats_out.append(r.Latitude()); lons_out.append(r.Longitude())
                geo_out.append(geo); temp_out.append(temp); ws_out.append(ws); wd_out.append(wd)
        if not lats_out: return None, "No s'han rebut dades per al nivell de 500 hPa."
        return (lats_out, lons_out, geo_out, temp_out, ws_out, wd_out), None
    except Exception as e: return None, str(e)

# --- 2. FUNCIONS DE VISUALITZACIÓ I LÒGICA ---
def display_metrics(params):
    cols=st.columns(4); metric_map={'CAPE':'J/kg','CIN':'J/kg','Shear_0-6km':'m/s','SRH_0-1km':'m²/s²'}
    for i,(param,unit) in enumerate(metric_map.items()): val=params.get(param); val_str=f"{val:.0f}" if val is not None else "---"; cols[i].metric(label=param,value=f"{val_str} {unit}")

def get_wind_colormap():
    colors = ['#FFFFFF', '#E0F5FF', '#B9E8FF', '#87D7F9', '#5AC7E3', '#2DB8CC', '#3FC3A3', '#5ABF7A', '#75BB51', '#98D849', '#C2E240', '#EBEC38', '#F5D03A', '#FDB43D', '#F7983F', '#E97F41', '#D76643', '#C44E45', '#B23547', '#A22428', '#881015', '#6D002F', '#860057', '#A0007F', '#B900A8', '#D300D0', '#E760E7', '#F6A9F6', '#FFFFFF', '#CCCCCC']
    levels = list(range(0, 95, 5)) + list(range(100, 211, 10))
    cmap = ListedColormap(colors, name='wind_speed_custom'); norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    return cmap, norm, levels

def crear_mapa_500hpa(lats, lons, geo_data, temp_data, wind_speed_data, wind_dir_data):
    extent = [0, 3.5, 40.4, 43]; fig, ax = plt.subplots(figsize=(10, 10), dpi=200, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(extent, crs=ccrs.PlateCarree()); ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=0); ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8, zorder=5); ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', zorder=5)
    grid_lon, grid_lat = np.meshgrid(np.linspace(extent[0], extent[1], 200), np.linspace(extent[2], extent[3], 200))
    grid_geo = griddata((lons, lats), geo_data, (grid_lon, grid_lat), method='cubic'); grid_temp = griddata((lons, lats), temp_data, (grid_lon, grid_lat), method='cubic')
    temp_levels = np.arange(-30, 1, 2); cmap_temp = plt.get_cmap('coolwarm')
    cf = ax.contourf(grid_lon, grid_lat, grid_temp, levels=temp_levels, cmap=cmap_temp, extend='min', alpha=0.7, zorder=2)
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7); cbar.set_label("Temperatura a 500 hPa (°C)", size=10)
    temp_line_levels = np.arange(-30, 1, 1)
    cs_temp = ax.contour(grid_lon, grid_lat, grid_temp, levels=temp_line_levels, colors='gray', linewidths=0.8, linestyles='--', zorder=4)
    ax.clabel(cs_temp, inline=True, fontsize=7, fmt='%1.0f')
    geo_levels = np.arange(5200, 5901, 40)
    cs = ax.contour(grid_lon, grid_lat, grid_geo, levels=geo_levels, colors='black', linewidths=1.5, zorder=3)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
    points_indices_1d = slice(None, None, 5) 
    u, v = mpcalc.wind_components(np.array(wind_speed_data) * units('km/h'), np.array(wind_dir_data) * units.degrees)
    ax.barbs(np.array(lons)[points_indices_1d], np.array(lats)[points_indices_1d], u.to('knots').m[points_indices_1d], v.to('knots').m[points_indices_1d], length=5, zorder=6, transform=ccrs.PlateCarree())
    ax.set_title("Anàlisi a 500 hPa (Altura, Temperatura i Vent)", weight='bold', fontsize=16)
    return fig

def crear_mapa_vents_velocitat(lats, lons, data, nivell):
    extent = [0, 3.5, 40.4, 43]; fig, ax = plt.subplots(figsize=(10, 10), dpi=200, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(extent, crs=ccrs.PlateCarree()); ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=0); ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8, zorder=5); ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', zorder=5)
    speed_data, direction_data = data; cmap, norm, levels = get_wind_colormap(); grid_lon, grid_lat = np.meshgrid(np.linspace(extent[0], extent[1], 200), np.linspace(extent[2], extent[3], 200))
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), method='cubic')
    ax.contourf(grid_lon, grid_lat, grid_speed, levels=levels, cmap=cmap, norm=norm, alpha=0.8, zorder=2, extend='max')
    speeds_ms = np.array(speed_data) * units('km/h'); dirs_deg = np.array(direction_data) * units.degrees; u_comp, v_comp = mpcalc.wind_components(speeds_ms, dirs_deg)
    rbf_u = Rbf(lons, lats, u_comp.to('m/s').m, function='thin_plate', smooth=0); rbf_v = Rbf(lons, lats, v_comp.to('m/s').m, function='thin_plate', smooth=0)
    u_grid = rbf_u(grid_lon, grid_lat); v_grid = rbf_v(grid_lon, grid_lat)
    ax.streamplot(grid_lon, grid_lat, u_grid, v_grid, color='black', linewidth=0.6, density=2.5, arrowsize=0.6, zorder=4)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', shrink=0.7, ticks=levels[::2])
    cbar.set_label(f"Velocitat del Vent (km/h)", size=10); ax.set_title(f"Vent a {nivell} hPa", weight='bold', fontsize=16); return fig

def crear_mapa_vents_professional(lats, lons, data, nivell, lat_sel, lon_sel, nom_poble_sel):
    extent = [0, 3.5, 40.4, 43]; fig, ax = plt.subplots(figsize=(10, 10), dpi=200, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(extent, crs=ccrs.PlateCarree()); ax.add_feature(cfeature.LAND,facecolor="#dddddd",zorder=0); ax.add_feature(cfeature.OCEAN,facecolor='#b0c4de',zorder=0)
    ax.add_feature(cfeature.COASTLINE,edgecolor='black',linewidth=0.8,zorder=5); ax.add_feature(cfeature.BORDERS,linestyle=':',edgecolor='black',linewidth=0.6,zorder=5)
    speed_data,direction_data=data; speeds_ms=np.array(speed_data)*units('km/h'); dirs_deg=np.array(direction_data)*units.degrees; u_comp,v_comp=mpcalc.wind_components(speeds_ms,dirs_deg)
    grid_lon,grid_lat=np.meshgrid(np.linspace(extent[0],extent[1],200),np.linspace(extent[2],extent[3],200))
    rbf_u = Rbf(lons,lats,u_comp.to('m/s').m,function='thin_plate',smooth=0); rbf_v = Rbf(lons,lats,v_comp.to('m/s').m,function='thin_plate',smooth=0)
    u_grid = rbf_u(grid_lon,grid_lat); v_grid = rbf_v(grid_lon,grid_lat); dx,dy=mpcalc.lat_lon_grid_deltas(grid_lon,grid_lat); divergence=mpcalc.divergence(u_grid*units('m/s'),v_grid*units('m/s'),dx=dx,dy=dy)*1e5
    max_conv=np.nanmin(divergence); ax.set_title(f"Flux i Convergència (Màx: {max_conv:.1f}) a {nivell}hPa",weight='bold',fontsize=16)
    cmap_conv_div='coolwarm_r'; max_abs_val=30; levels=np.linspace(-max_abs_val,max_abs_val,15)
    cbar=fig.colorbar(ax.contourf(grid_lon,grid_lat,divergence,levels=levels,cmap=cmap_conv_div,alpha=0.6,zorder=2,extend='both'),ax=ax,orientation='vertical',shrink=0.7)
    cbar.set_label('Convergència (vermell) / Divergència (blau) (x10⁻⁵ s⁻¹)',size=10)
    ax.contour(grid_lon,grid_lat,divergence,levels=[-35,-25,-15],colors='black',linewidths=[1.5,1,0.5],alpha=0.5,zorder=3)
    ax.streamplot(grid_lon,grid_lat,u_grid,v_grid,color='black',linewidth=0.4,density=5.5,arrowsize=0.3,zorder=4)
    ax.plot(lon_sel,lat_sel,'o',markerfacecolor='yellow',markeredgecolor='black',markersize=6,transform=ccrs.Geodetic(),zorder=6)
    ax.text(lon_sel+0.05,lat_sel,nom_poble_sel,transform=ccrs.Geodetic(),zorder=7,fontsize=9,weight='bold',path_effects=[path_effects.withStroke(linewidth=2,foreground='white')]); return fig

def crear_mapa_escalar_professional(lats, lons, data, titol, cmap, levels, unitat):
    extent=[0,3.5,40.4,43]; fig,ax=plt.subplots(figsize=(10,10),dpi=200,subplot_kw={'projection':ccrs.PlateCarree()}); ax.set_extent(extent,crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,facecolor="#E0E0E0",zorder=0); ax.add_feature(cfeature.OCEAN,facecolor='#b0c4de',zorder=0); ax.add_feature(cfeature.COASTLINE,edgecolor='black',linewidth=0.8,zorder=5); ax.add_feature(cfeature.BORDERS,linestyle='-',edgecolor='black',zorder=5)
    grid_lon, grid_lat = np.meshgrid(np.linspace(extent[0],extent[1],200), np.linspace(extent[2],extent[3],200)); grid_data = griddata((lons,lats),data,(grid_lon,grid_lat),method='cubic')
    ax.contourf(grid_lon,grid_lat,grid_data,levels=levels,cmap=cmap,alpha=0.7,zorder=2,extend='max'); contorns = ax.contour(grid_lon,grid_lat,grid_data,levels=levels,colors='black',linewidths=0.5,alpha=0.7,zorder=3)
    ax.clabel(contorns, inline=True, fontsize=8, fmt='%1.0f'); cbar=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(levels[0], levels[-1]), cmap=cmap), ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label(f"{titol} ({unitat})",size=10); ax.set_title(titol,weight='bold',fontsize=16); return fig

def crear_skewt(p, T, Td, u, v):
    fig=plt.figure(figsize=(9,9),dpi=150); skew=SkewT(fig,rotation=45,rect=(0.1,0.1,0.8,0.85)); skew.ax.grid(True,linestyle='-',alpha=0.5)
    skew.plot(p,T,'r',lw=2,label='Temperatura'); skew.plot(p,Td,'g',lw=2,label='Punt de Rosada'); skew.plot_barbs(p,u.to('kt'),v.to('kt'),y_clip_radius=0.03)
    skew.plot_dry_adiabats(color='lightcoral',linestyle='--',alpha=0.7); skew.plot_moist_adiabats(color='cornflowerblue',linestyle='--',alpha=0.7); skew.plot_mixing_lines(color='limegreen',linestyle='--',alpha=0.7)
    prof=mpcalc.parcel_profile(p,T[0],Td[0]); skew.plot(p,prof,'k',linewidth=2,label='Trajectòria Parcel·la'); skew.shade_cape(p,T,prof,color='lightsteelblue',alpha=0.4); skew.shade_cin(p,T,prof,color='lightgrey',alpha=0.4)
    skew.ax.set_ylim(1000,100); skew.ax.set_xlim(-40,40); skew.ax.set_title("Sondeig Vertical (Skew-T)",weight='bold',fontsize=14)
    skew.ax.set_xlabel("Temperatura (°C)"); skew.ax.set_ylabel("Pressió (hPa)"); skew.ax.legend(); return fig

def crear_hodograf(u, v):
    fig,ax=plt.subplots(1,1,figsize=(6,6),dpi=150); h=Hodograph(ax,component_range=60.); h.add_grid(increment=20); h.plot(u.to('kt'),v.to('kt'))
    ax.set_title("Hodògraf",weight='bold'); return fig

# --- 3. INTERFAZ Y FLUJO PRINCIPAL ---
st.markdown('<h1 style="text-align: center; color: #FF4B4B;">🌪️ Terminal d\'Anàlisi de Temps Sever | Catalunya</h1>', unsafe_allow_html=True)
with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    with col1: st.selectbox("Capital de referència:", sorted(ciutats_catalunya.keys()), key="poble_selector")
    with col2: st.selectbox("Dia del pronòstic:", ("Avui", "Demà"), key="dia_selector")
    with col3: st.selectbox("Hora del pronòstic:", options=[f"{h:02d}:00h" for h in range(24)], key="hora_selector")

hora_sel = int(st.session_state.hora_selector.split(':')[0]); dia_sel = st.session_state.dia_selector
hourly_index_sel = hora_sel + (24 if dia_sel == "Demà" else 0)
poble_sel = st.session_state.poble_selector
lat_sel, lon_sel = ciutats_catalunya[poble_sel]['lat'], ciutats_catalunya[poble_sel]['lon']
data_tuple, error_msg = carregar_dades_completes(lat_sel, lon_sel, hourly_index_sel)
if error_msg: st.error(f"No s'ha pogut carregar el sondeig: {error_msg}"); data_tuple = None

tab1, tab2 = st.tabs(["🗺️ Anàlisi de Mapes", "📊 Anàlisi Vertical"])
with tab1:
    col_map_1, col_map_2 = st.columns([2.5, 1.5])
    with col_map_1:
        map_options = ["Flux i Convergència", "Anàlisi a 500hPa", "Vent a 300hPa", "Vent a 700hPa", "CAPE (Energia)", "Humitat a 700hPa"]
        mapa_sel = st.selectbox("Selecciona la capa del mapa:", map_options)
        
        with st.spinner(f"🛰️ Processant dades i generant el mapa de {mapa_sel}..."):
            start_time = time.time()
            error_map = None
            
            if mapa_sel == "Flux i Convergència":
                nivells_convergencia = [p for p in PRESS_LEVELS if p >= 850]
                nivell_vents_sel = st.selectbox("Nivell d'anàlisi:", options=nivells_convergencia, format_func=lambda x: f"{x} hPa")
                map_data_vents, error_map = obtener_dades_vents_mapa(nivell_vents_sel, hourly_index_sel)
                if map_data_vents: lats, lons, data = map_data_vents; st.pyplot(crear_mapa_vents_professional(lats, lons, data, nivell_vents_sel, lat_sel, lon_sel, poble_sel))
            
            elif mapa_sel == "Anàlisi a 500hPa":
                map_data, error_map = obtener_dades_mapa_500hpa(hourly_index_sel)
                if map_data: lats, lons, geo, temp, ws, wd = map_data; st.pyplot(crear_mapa_500hpa(lats, lons, geo, temp, ws, wd))
            
            elif mapa_sel in ["Vent a 300hPa", "Vent a 700hPa"]:
                nivell_hpa = int(mapa_sel.split(' ')[2].replace('hPa', ''))
                map_data_vents, error_map = obtener_dades_vents_mapa(nivell_hpa, hourly_index_sel)
                if map_data_vents: lats, lons, data = map_data_vents; st.pyplot(crear_mapa_vents_velocitat(lats, lons, data, nivell_hpa))
            
            else:
                variable = "cape" if mapa_sel == "CAPE (Energia)" else "relative_humidity_700hPa"
                map_data, error_map = obtener_dades_mapa(variable, hourly_index_sel)
                if map_data:
                    lats, lons, data = map_data
                    if mapa_sel == "CAPE (Energia)": st.pyplot(crear_mapa_escalar_professional(lats, lons, data, "CAPE", "plasma", np.arange(250, 4001, 250), "J/kg"))
                    else: st.pyplot(crear_mapa_escalar_professional(lats, lons, data, "Humitat Relativa a 700hPa", "Greens", np.arange(50, 101, 5), "%"))
            
            if error_map: st.error(f"Error en carregar el mapa: {error_map}")
            
            # Assegurem que la pantalla de càrrega duri almenys 4 segons
            elapsed_time = time.time() - start_time
            if elapsed_time < 4:
                time.sleep(4 - elapsed_time)

    with col_map_2:
        st.subheader("Imatges en Temps Real")
        view_choice = st.radio("Selecciona la vista:", ("Satèl·lit", "Radar de Precipitació"), horizontal=True)
        if view_choice == "Satèl·lit":
            try:
                satellite_url = "https://modeles20.meteociel.fr/satellite/latestsatviscolmtgsp.png"; unique_url = f"{satellite_url}?ver={int(time.time())}"
                headers = {'User-Agent': 'Mozilla/5.0'}; response = requests.get(unique_url, headers=headers)
                if response.status_code == 200: st.image(response.content, caption="Satèl·lit visible. Font: Meteociel", use_container_width=True)
                else: st.warning("No s'ha pogut carregar la imatge del satèl·lit.")
            except Exception as e: st.error(f"Error de xarxa en carregar el satèl·lit.")
        elif view_choice == "Radar de Precipitació":
            try:
                radar_url = "https://www.meteociel.fr/cartes_obs/radar/lastradar_sp_ne.gif"; unique_url = f"{radar_url}?ver={int(time.time())}"
                headers = {'User-Agent': 'Mozilla/5.0'}; response = requests.get(unique_url, headers=headers)
                if response.status_code == 200: st.image(response.content, caption="Radar de precipitació (NE Península). Font: Meteociel", use_container_width=True)
                else: st.warning("No s'ha pogut carregar la imatge del radar.")
            except Exception as e: st.error(f"Error de xarxa en carregar el radar.")

with tab2:
    if data_tuple:
        sounding_data, params_calculats = data_tuple
        st.subheader("Paràmetres Clau del Sondeig")
        st.info("Aquests paràmetres s'han calculat a partir del perfil vertical per al punt i hora seleccionats.")
        display_metrics(params_calculats)
        st.divider()
        col_sondeig_1, col_sondeig_2 = st.columns(2)
        with col_sondeig_1: st.pyplot(crear_skewt(sounding_data[0], sounding_data[1], sounding_data[2], sounding_data[3], sounding_data[4]))
        with col_sondeig_2: st.pyplot(crear_hodograf(sounding_data[3], sounding_data[4]))
    else:
        st.warning("No hi ha dades de sondeo disponibles per mostrar.")
