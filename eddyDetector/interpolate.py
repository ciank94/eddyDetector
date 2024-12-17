# stereographic projection preserves angles and shapes near the equator
# lon_0 is the central meridian
# lat_0 is the latitude of origin
# lat_ts is the latitude of true scale, where we want the projection least distorted

import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# Configure logging format to include the class name
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

class InterpolateSLD:
    def __init__(self, df):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self.df = df
        self.lon = self.df['lon']
        self.lat = self.df['lat']
        self.var_list = ['u', 'v', 'ssh', 'net_vel']
        self.interpolated_vars = {}
        self.init_interpolator()
        return

    def init_interpolator(self):
        self.logger.info(f"Initializing interpolator")
        new_shape = int(self.df['u'].shape[0]*5), int(self.df['u'].shape[1]*5)
        lon_new_vals = np.linspace(self.lon.min(), self.lon.max(), new_shape[1])
        lat_new_vals = np.linspace(self.lat.min(), self.lat.max(), new_shape[0])
        lat_new_points, lon_new_points = np.meshgrid(lat_new_vals, lon_new_vals, indexing='ij') # index order for (y, x) grid
        points_new = np.column_stack((lat_new_points.flatten(), lon_new_points.flatten()))
        original_points = (self.lat, self.lon)
        for name in self.var_list:
            interpolator = RegularGridInterpolator(original_points, self.df[name], method='linear', bounds_error=False, fill_value=np.nan)
            self.interpolated_vars[name] = interpolator(points_new).reshape(new_shape[0], new_shape[1])
        #self.test_plot_interpolation(lon_new_vals, lat_new_vals)
        return

    def test_plot_interpolation(self, lon_new_vals, lat_new_vals):
        fig = plt.figure(figsize=(16, 6*len(self.var_list)))
        plt.rcParams.update({'font.size': 12})
        
        for idx, var_name in enumerate(self.var_list):
            # Original data subplot
            plt.subplot(len(self.var_list), 2, 2*idx + 1)
            im = plt.pcolormesh(self.lon, self.lat, self.df[var_name])
            plt.colorbar(im, label=var_name.upper())
            
            # Interpolated data subplot
            plt.subplot(len(self.var_list), 2, 2*idx + 2)
            im = plt.pcolormesh(lon_new_vals, lat_new_vals, self.interpolated_vars[var_name])
            plt.colorbar(im, label=var_name.upper())
        
        plt.tight_layout(h_pad=2.0, w_pad=1.0)
        plt.show()
        return
