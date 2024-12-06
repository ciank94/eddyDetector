import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.reader import download_lists, download_cds_data, subset_netcdf
from src.eddy_methods import calculate_okubo_weiss, interpolate_grid, eddy_filter, slide_detect
import numpy as np
import matplotlib.pyplot as plt

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

folder = './'
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
df = interpolate_grid(subset_df, new_shape)
val_ow, vorticity = calculate_okubo_weiss(np.array(df['ugos']), np.array(df['vgos']))
ow, cyc_mask, acyc_mask = eddy_filter(val_ow, vorticity)

#==========Section 3: algorithm=============
ssh, geos_vel, ugos, vgos, lat, lon = [np.array(arr) for arr in (df['geos_vel'], df['geos_vel'], df['ugos'],
                                                                 df['vgos'], df['latitude'], df['longitude'])]
[eddy_centres, eddy_borders] = slide_detect(ow=ow,
                     mag_uv=geos_vel,
                     u_geo= ugos,
                     v_geo= vgos,
                     ssh=ssh,
                     block_size = 7,
                     subgrid_size = 3)

plt.contour(ssh, levels = 60, cmap=plt.get_cmap('grey'))
plt.contourf(geos_vel, levels = 30, cmap=plt.get_cmap('hot'))
[plt.plot(eddy_borders[i, :, 0], eddy_borders[i, :, 1], c='r') for i in range(0, eddy_borders.shape[0])]
plt.show()
