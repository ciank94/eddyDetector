import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator

class Plotting:
    def __init__(self):
        pass
    
    @staticmethod
    def plot_eddy_detection(ssh, geos_vel, eddy_borders, lat, lon):
        """Plot the SSH contours, geostrophic velocity, and eddy borders.

        :param ssh: Sea surface height data
        :type ssh: numpy.ndarray
        :param geos_vel: Geostrophic velocity magnitude
        :type geos_vel: numpy.ndarray
        :param eddy_borders: Array containing eddy border coordinates with shape (n_eddies, points, 2)
        :type eddy_borders: numpy.ndarray
        :param lat: Latitude values
        :type lat: numpy.ndarray
        :param lon: Longitude values
        :type lon: numpy.ndarray
        """
        plt.rcParams.update({'font.size': 12})
        
        # Ensure the results directory exists
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

        # Create the results directory if it does not already exist.
        # The exist_ok parameter means that if the directory already exists,
        # os.makedirs will not raise an OSError. If exist_ok is False (the
        # default), os.makedirs will raise FileExistsError if the directory
        # already exists.
        os.makedirs(results_dir, exist_ok=True)
        
        # Create a new figure with a specified size
        # The size is given in inches. The default is 8 inches by 6 inches.
        plt.figure(figsize=(10, 8))
        
        # Plot SSH contours
        plt.contour(lon, lat, ssh, levels=60, cmap=plt.get_cmap('grey'))
        
        # Plot geostrophic velocity
        plt.contourf(lon, lat, geos_vel, levels=30, cmap=plt.get_cmap('hot'))
        
        #Plot eddy borders
        # Create grid coordinates for interpolation
        x_grid = np.arange(geos_vel.shape[1])
        y_grid = np.arange(geos_vel.shape[0])
        
        for i in range(0, eddy_borders.shape[0]):
            # Interpolate x and y coordinates to lon/lat independently
            border_x = np.interp(eddy_borders[i, :, 0], x_grid, lon)
            border_y = np.interp(eddy_borders[i, :, 1], y_grid, lat)
            plt.plot(border_x, border_y, c='w')
        
        plt.colorbar(label='Geostrophic Velocity')
        plt.title('Eddy Detection Results')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(os.path.join(results_dir, 'eddy_detection.png'), dpi=300)
        plt.show()
        return
