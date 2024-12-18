# eddyDetector/main/detect.py
# ===========Section 0: imports=============
import numpy as np
import matplotlib.pyplot as plt
from eddyDetector import *

# ==========Section 1: Parameters============= (stuff I will change a lot, others in a ./input_files/yaml file)
datetime_start = "2017-01-25"
datetime_end = "2017-02-01"
input_filepath = './input_files'
output_filepath = './output_files'
lat_bounds = (-73, -50)
lon_bounds = (-70, -31)
time_index = 0 # index for time dimension (test)
scale_factor_interpolation = 5

# ==========Section 2: Prepare data=============
fileExp = FileExplorerSLD(datetime_start, datetime_end) # initiate file explorer for sea-level data (SLD) input files
fileExp.download_check(input_filepath) # check if files are already downloaded
reader = ReaderSLD(fileExp, time_index) # initiate reader for sea-level data (SLD) input files at time index
ncfile_subset = reader.subset_netcdf(lon_bounds, lat_bounds) # subset data
interpolator = InterpolateSLD(ncfile_subset) # initiate interpolator with latitude and longitude meshgrid
data = interpolator.interpolate_grid(scale_factor_interpolation) # interpolate data

#==========Section 2: Detect eddies=============
detector = DetectEddiesSLD(data) # initiate eddy detector
detector.okubo_weiss() # calculate Okubo-Weiss field and filter with threshold
breakpoint()

val_ow, vorticity = EddyMethods.calculate_okubo_weiss(np.array(df['ugos']), np.array(df['vgos']))
ow, cyc_mask, acyc_mask = EddyMethods.eddy_filter(val_ow, vorticity)


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
