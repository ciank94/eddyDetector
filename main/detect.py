# findEddy/main/detect.py
# ===========Section 0: imports=============
import numpy as np
import matplotlib.pyplot as plt
from eddy_source import (
    EddyMethods,
    FileExplorerSLD,
    ReaderSLD,
    Plotting,
)

# ==========Section 1: Parameters=============
datetime_start = "2017-01-01"
datetime_end = "2017-02-01"
filepath = './input_files'
lat_bounds = (-73, -50)
lon_bounds = (-70, -31)

# ==========Section 2: FileExplorer=============
fileExp = FileExplorerSLD(datetime_start, datetime_end) # initiate file explorer for sea-level data (SLD) input files
fileExp.download_check(filepath) # check if files are already downloaded

#==========Section 3: Reader=============
time_index = 0 # index for time dimension (test)
reader = ReaderSLD(fileExp, time_index)
df = reader.subset_netcdf(lon_bounds, lat_bounds)

breakpoint()


#==========Section 2: filter dataset=============
new_shape = (int(subset_df['ugos'].shape[0]*5), int(subset_df['ugos'].shape[1]*5))
df = EddyMethods.interpolate_grid(subset_df, new_shape)
val_ow, vorticity = EddyMethods.calculate_okubo_weiss(np.array(df['ugos']), np.array(df['vgos']))
ow, cyc_mask, acyc_mask = EddyMethods.eddy_filter(val_ow, vorticity)
#[global_minima, global_minima_mask] = EddyMethods.find_global_minima_with_masking(ow, max_eddies=1000, mask_radius=5)


#==========Section 3: algorithm=============
ssh, geos_vel, ugos, vgos, lat, lon = [np.array(arr) for arr in (df['adt'], df['geos_vel'], df['ugos'],
                                                                 df['vgos'], df['latitude'], df['longitude'])]
eddies = EddyMethods.global_detect(ow=ow,
                     mag_uv=geos_vel,
                     u_geo= ugos,
                     v_geo= vgos,
                     ssh=ssh)


#==========Section 4: plot=============
#Plotting.plot_zoomed_eddy(ssh, ugos, vgos, eddies[0], lat, lon)
Plotting.plot_eddy_detection(ssh, geos_vel, eddies, lat, lon)
#Plotting.plot_zoomed_eddy_with_contours(ssh, ugos, vgos, eddies[0], lat, lon, ow)
breakpoint()
