import sys
import os
# Add the parent directory to Python path so we can import the src package
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import (
    EddyMethods,
    download_lists,
    download_cds_data,
    subset_netcdf,
    plot_eddy_detection,
    __version__
)
import numpy as np
import matplotlib.pyplot as plt

# Print package version
print(f"Package version: {__version__}")

#==========Section 1: reader=============
download_f = False
if download_f:
    year, month, day = download_lists(y_start=2017,
                                      y_end=2017,
                                      m_start=1,
                                      m_end=1,
                                      d_start=0,
                                      d_end=1)
    print(f"day: {day}")
    download_cds_data(year=year,
                      month=month,
                      day=day,
                      version='vdt2021')

folder = "./"
filename = folder + 'dt_global_twosat_phy_l4_20170101_vDT2021.nc'

subset_df = subset_netcdf(filepath= filename,
         lon_range = (-70, -31),
         lat_range = (-73, -50),
        time_index= 0,
        variables = ['longitude', 'latitude', 'ugos', 'vgos', 'adt'])
#subset_df = subset_netcdf(filepath= filename,
#         lon_range = (3, 30),
#         lat_range = (50, 80),
#        time_index= 0,
#        variables = ['longitude', 'latitude', 'ugos', 'vgos', 'adt'])

#==========Section 2: filter dataset=============
new_shape = (int(subset_df['ugos'].shape[0]*5), int(subset_df['ugos'].shape[1]*5))
df = EddyMethods.interpolate_grid(subset_df, new_shape)
val_ow, vorticity = EddyMethods.calculate_okubo_weiss(np.array(df['ugos']), np.array(df['vgos']))
ow, cyc_mask, acyc_mask = EddyMethods.eddy_filter(val_ow, vorticity)
[global_minima, global_minima_mask] = EddyMethods.find_global_minima_with_masking(ow, max_eddies=1000, mask_radius=5)


#==========Section 3: algorithm=============
ssh, geos_vel, ugos, vgos, lat, lon = [np.array(arr) for arr in (df['adt'], df['geos_vel'], df['ugos'],
                                                                 df['vgos'], df['latitude'], df['longitude'])]
[eddy_centres, eddy_borders] = EddyMethods.global_detect(ow=ow,
                     mag_uv=geos_vel,
                     u_geo= ugos,
                     v_geo= vgos,
                     ssh=ssh,
                     global_minima=global_minima)


#==========Section 4: plot=============
plot_eddy_detection(ssh, geos_vel, eddy_borders, lat, lon)
