import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from datetime import datetime

# EC-AIFS প্যারামিটার ম্যাপিং
PARAM_MAP = {
    "Velocity Potential": "vp",
    "Geopotential Height": "gh",
    "Zonal Wind": "u",
    "Meridional Wind": "v",
    "Vertical Wind Shear": "vws"
}

def get_title(param, level):
    return f"EC-AIFS {param} at {level}"

def plot_ecaifs(param, level="200mb", run_date=None):
    """
    EC-AIFS মডেল ডাটা প্লট করার ফাংশন।
    param: Velocity Potential, Geopotential Height, Zonal Wind, Meridional Wind, Vertical Wind Shear
    level: 850mb, 700mb, 500mb, 200mb
    """
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # ম্যাপ ফিচার যুক্ত করা
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    
    # ডেমো ভিজ্যুয়ালাইজেশন গ্রিড (প্রয়োজন অনুযায়ী আসল ডাটা লোড যুক্ত করুন)
    lons = np.linspace(40, 120, 100)
    lats = np.linspace(-10, 40, 60)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # উদাহরণ স্বরূপ নমুনা ডাটা জেনারেশন
    data = np.sin(np.radians(lon_grid)) * np.cos(np.radians(lat_grid))
    
    clevs = np.linspace(-2, 2, 21)
    cs = ax.contourf(lon_grid, lat_grid, data, levels=clevs, cmap='Spectral_r', transform=ccrs.PlateCarree())
    
    title_str = f"EC-AIFS - {param} ({level})"
    plt.title(title_str, fontsize=12, weight='bold', loc='left')
    plt.title(datetime.utcnow().strftime('%Y-%m-%d %H:00 UTC'), fontsize=10, loc='right')
    
    plt.colorbar(cs, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    
    # ড্রয়ার/ফ্লাস্কের জন্য ফাইল সেভ নিশ্চিত করা
    output_dir = "static/output"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"ecaifs_{PARAM_MAP.get(param, 'data')}_{level}.png")
    plt.savefig(file_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    return file_path
