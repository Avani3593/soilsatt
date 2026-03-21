"""
SoilSat Prediction Script
Runs on GitHub Actions every 6 days.
Fetches latest Sentinel-1 data from GEE,
runs XGBoost model, runs Richards equation,
saves predictions to data/predictions.json
"""

import ee
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────

import ee
import os

service_account = os.environ.get('GEE_SERVICE_ACCOUNT')
credentials = ee.ServiceAccountCredentials(service_account, '/tmp/gee_key.json')
ee.Initialize(credentials, project='winged-polygon-490609-q3')

GEE_PROJECT = 'winged-polygon-490609-q3'

TALUKAS = {
    'baramati': {'lat': 18.15, 'lon': 74.58, 'name': 'Baramati'},
    'indapur':  {'lat': 17.97, 'lon': 75.01, 'name': 'Indapur'},
    'haveli':   {'lat': 18.52, 'lon': 73.92, 'name': 'Haveli'},
    'junnar':   {'lat': 19.20, 'lon': 73.88, 'name': 'Junnar'},
    'shirur':   {'lat': 18.83, 'lon': 74.38, 'name': 'Shirur'},
}

VG_PARAMS = {
    'theta_r': 0.089, 'theta_s': 0.500,
    'alpha':   0.010, 'n':       1.230,
    'Ks':      0.26,  'l':       0.5
}

FEATURE_COLS = [
    'VV_filtered_dB', 'VH_filtered_dB', 'VV_VH_ratio', 'RVI',
    'incidence_angle', 'pass_flag',
    'NDVI', 'SAVI',
    'LST_C', 'precip_30d_mm', 'precip_7d_mm',
    'slope', 'aspect', 'TWI', 'LULC',
    'day_of_year', 'season'
]

# ── GEE DATA FETCHER ──────────────────────────────────────────────────────────

def fetch_sar_features(lat, lon, days_back=12):
    point  = ee.Geometry.Point([lon, lat])
    buffer = point.buffer(500)

    end_date   = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    start_str  = start_date.strftime('%Y-%m-%d')
    end_str    = end_date.strftime('%Y-%m-%d')

    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(buffer)
          .filterDate(start_str, end_str)
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .filter(ee.Filter.eq('instrumentMode', 'IW')))

    if s1.size().getInfo() == 0:
        print(f"  No S1 images found, extending to {days_back*2} days")
        return fetch_sar_features(lat, lon, days_back * 2)

    img    = s1.sort('system:time_start', False).first()
    vv_db  = img.select('VV').rename('VV_filtered_dB')
    vh_db  = img.select('VH').rename('VH_filtered_dB')
    vv_pow = ee.Image(10).pow(vv_db.divide(10))
    vh_pow = ee.Image(10).pow(vh_db.divide(10))
    ratio  = vv_pow.divide(vh_pow).log10().multiply(10).rename('VV_VH_ratio')
    rvi    = vh_pow.multiply(4).divide(vv_pow.add(vh_pow)).rename('RVI')
    angle  = img.select('angle').rename('incidence_angle')

    pass_val  = ee.Algorithms.If(
        ee.String(img.get('orbitProperties_pass')).equals('ASCENDING'),
        ee.Image.constant(1), ee.Image.constant(0)
    )
    pass_flag = ee.Image(pass_val).rename('pass_flag')
    acq_date  = ee.Date(img.get('system:time_start'))

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(buffer)
          .filterDate(acq_date.advance(-15,'day'), acq_date.advance(15,'day'))
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
          .sort('CLOUDY_PIXEL_PERCENTAGE'))

    ndvi = ee.Image(ee.Algorithms.If(
        s2.size().gt(0),
        s2.first().normalizedDifference(['B8','B4']).rename('NDVI'),
        ee.Image.constant(-9999).rename('NDVI')
    ))
    savi = ee.Image(ee.Algorithms.If(
        s2.size().gt(0),
        s2.first().expression(
            '1.5*(NIR-RED)/(NIR+RED+0.5)',
            {'NIR': s2.first().select('B8'), 'RED': s2.first().select('B4')}
        ).rename('SAVI'),
        ee.Image.constant(-9999).rename('SAVI')
    ))

    modis = (ee.ImageCollection('MODIS/061/MOD11A1')
             .filterBounds(buffer)
             .filterDate(acq_date.advance(-8,'day'), acq_date.advance(1,'day'))
             .select('LST_Day_1km').mean())
    lst = modis.multiply(0.02).subtract(273.15).rename('LST_C')

    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterBounds(buffer)
    p30  = (era5.filterDate(acq_date.advance(-30,'day'), acq_date)
               .select('total_precipitation_sum')
               .sum().multiply(1000).rename('precip_30d_mm'))
    p7   = (era5.filterDate(acq_date.advance(-7,'day'), acq_date)
               .select('total_precipitation_sum')
               .sum().multiply(1000).rename('precip_7d_mm'))

    dem    = ee.Image('USGS/SRTMGL1_003')
    slope  = ee.Terrain.slope(dem).rename('slope')
    aspect = ee.Terrain.aspect(dem).rename('aspect')
    area   = ee.Image('WWF/HydroSHEDS/15ACC').rename('flow_acc')
    twi    = (area.add(1).log()
                .subtract(slope.multiply(np.pi/180).tan().add(0.001).log())
                .rename('TWI'))
    lulc   = (ee.Image('COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019')
                .select('discrete_classification').rename('LULC'))

    doy_num   = acq_date.getRelative('day','year').add(1).getInfo()
    doy       = ee.Image.constant(doy_num).rename('day_of_year')
    season_val = 1 if 152 <= doy_num <= 304 else (2 if doy_num >= 305 or doy_num <= 59 else 0)
    season    = ee.Image.constant(season_val).rename('season')

    stack = vv_db.addBands([vh_db, ratio, rvi, angle, pass_flag,
                             ndvi, savi, lst, p30, p7,
                             slope, aspect, twi, lulc, doy, season])

    sample   = stack.sample(region=buffer, scale=10, numPixels=20, geometries=False)
    features = sample.aggregate_mean(stack.bandNames()).getInfo()
    acq_date_str = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()

    return features, acq_date_str


# ── RICHARDS EQUATION ─────────────────────────────────────────────────────────

class RichardsRootZone:
    def __init__(self, params=VG_PARAMS):
        self.p  = params
        self.dz = 2.0
        self.dt = 0.1
        self.z  = np.arange(0, 101, self.dz)
        self.nz = len(self.z)

    def theta_to_h(self, theta):
        p  = self.p
        Se = np.clip((theta - p['theta_r']) / (p['theta_s'] - p['theta_r']), 1e-6, 1-1e-6)
        m  = 1 - 1/p['n']
        return -(1/p['alpha']) * (Se**(-1/m) - 1)**(1/p['n'])

    def K(self, h):
        p  = self.p
        m  = 1 - 1/p['n']
        Se = (1 + (p['alpha'] * np.abs(h))**p['n'])**(-m)
        Se = np.where(h >= 0, 1.0, Se)
        return p['Ks'] * Se**p['l'] * (1-(1-Se**(1/m))**m)**2

    def solve(self, theta_surface, sim_days=7):
        dz    = self.dz
        dt    = self.dt
        nz    = self.nz
        FC    = self.p['theta_r'] + 0.55 * (self.p['theta_s'] - self.p['theta_r'])
        theta = np.full(nz, FC)
        h     = np.array([self.theta_to_h(t) for t in theta])
        daily_avg = []

        for step in range(int(sim_days/dt)):
            h[0]     = self.theta_to_h(theta_surface)
            Km       = 0.5 * (self.K(h[:-1]) + self.K(h[1:]))
            flux     = -Km * (np.diff(h) / dz - 1.0)
            theta[1:-1] += dt * (-np.diff(flux) / dz)
            theta    = np.clip(theta, self.p['theta_r'] + 1e-4, self.p['theta_s'])
            h        = np.array([self.theta_to_h(t) for t in theta])
            if step % int(1.0/dt) == 0:
                daily_avg.append(round(float(theta.mean()), 4))

        idx = lambda cm: int(cm / dz)
        return {
            'forecast_7day': daily_avg[:7],
            'd10':  round(float(theta[idx(10)]), 4),
            'd30':  round(float(theta[idx(30)]), 4),
            'd60':  round(float(theta[idx(60)]), 4),
            'd100': round(float(theta[idx(100)]), 4),
            'root_zone_avg': round(float(theta.mean()), 4),
        }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_predictions():
    print("="*50)
    print(f"SoilSat Prediction — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("="*50)

    # Authenticate GEE
    svc_account = os.environ.get('GEE_SERVICE_ACCOUNT')
    key_file    = '/tmp/gee_key.json'
    credentials = ee.ServiceAccountCredentials(svc_account, key_file)
    ee.Initialize(credentials, project=GEE_PROJECT)
    print("GEE authenticated")

    # Load model
    model  = joblib.load('soil_moisture_model.pkl')
    scaler = joblib.load('sm_scaler.pkl')
    f_cols = joblib.load('feature_cols.pkl') if Path('feature_cols.pkl').exists() else FEATURE_COLS

    richards = RichardsRootZone()
    results  = {}

    for key, info in TALUKAS.items():
        print(f"\nProcessing {info['name']}...")
        try:
            features, acq_date = fetch_sar_features(info['lat'], info['lon'])
            print(f"  SAR date: {acq_date}")

            row = {col: features.get(col, 0) for col in f_cols}
            if row.get('NDVI', -9999) == -9999:
                row['NDVI'] = 0.312
                row['SAVI'] = 0.468

            df  = pd.DataFrame([row])[f_cols]
            X   = scaler.transform(df)
            sm  = float(np.clip(model.predict(X)[0], 0.02, 0.55))
            print(f"  Topsoil SM: {sm:.4f}")

            rz = richards.solve(sm)
            print(f"  Root zone: {rz['root_zone_avg']:.4f}")

            results[key] = {
                'name':          info['name'],
                'lat':           info['lat'],
                'lon':           info['lon'],
                'acq_date':      acq_date,
                'topsoil_sm':    round(sm, 4),
                'root_zone_avg': rz['root_zone_avg'],
                'depth': {
                    'd10':  rz['d10'],
                    'd30':  rz['d30'],
                    'd60':  rz['d60'],
                    'd100': rz['d100'],
                },
                'forecast_7day': rz['forecast_7day'],
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            results[key] = {'name': info['name'], 'error': str(e)}

    output = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'model':        'XGBoost + Richards Equation',
        'ubRMSE':       0.0139,
        'talukas':      results
    }

    Path('data').mkdir(exist_ok=True)
    with open('data/predictions.json', 'w') as f:
        json.dump(output, f, indent=2)

    with open('data/last_run.json', 'w') as f:
        json.dump({'last_run': output['generated_at'], 'status': 'success'}, f)

    print(f"\nDone. Saved data/predictions.json")
    ok = len([v for v in results.values() if 'error' not in v])
    print(f"Talukas succeeded: {ok}/{len(TALUKAS)}")


if __name__ == '__main__':
    run_predictions()
