# eddyDetector/main/detect.py
# ===========Section 0: imports=============
import numpy as np
import matplotlib.pyplot as plt
from eddyDetector import *

# ==========Section 1: Parameters============= (stuff I will change a lot, others in a ./input_files/yaml file)
datetime_start = "2017-01-25"
datetime_end = "2017-02-01"
input_filepath = './input_files' # input folder for sea-level data (SLD)
output_filepath = './output_files' # output folder for eddy properties
lat_bounds = (-73, -50) # latitude bounds
lon_bounds = (-70, -31) # longitude bounds
time_index = 0 # index for time dimension (test)
scale_factor_interpolation = 5 # scale factor for interpolation

# ==========Section 2: Prepare data=============
fileExp = FileExplorerSLD(datetime_start, datetime_end) # initiate file explorer for SLD input files
fileExp.download_check(input_filepath) # check if files are already downloaded and download if not
reader = ReaderSLD(fileExp, time_index) # initiate reader for SLD input files at time index
ncfile_subset = reader.subset_netcdf(lon_bounds, lat_bounds) # subset data with longitude and latitude bounds
interpolator = InterpolateSLD(ncfile_subset) # initiate interpolator with latitude and longitude meshgrid
data = interpolator.interpolate_grid(scale_factor_interpolation) # interpolate data with scale factor

#==========Section 2: Detect eddies and store properties=============
detector = DetectEddiesSLD(data) # initiate eddy detector object and calculate Okubo-Weiss field and filter with threshold
eddy_info = detector.eddy_algorithm() # detect eddies using global minima in the Okubo-Weiss field
output = OutputSLD(eddy_info) # initiate output object with eddy properties
breakpoint()

#==========Section 3: Plot eddies=============
#Plotting.plot_zoomed_eddy(ssh, ugos, vgos, eddies[0], lat, lon)
Plotting.plot_eddy_detection(ssh, geos_vel, eddies, lat, lon)
#Plotting.plot_zoomed_eddy_with_contours(ssh, ugos, vgos, eddies[0], lat, lon, ow)
breakpoint()
