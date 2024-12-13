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
        plt.contour(lon, lat, ssh, levels=60, colors='gray', alpha=0.5)
        
        # Plot geostrophic velocity
        plt.contourf(lon, lat, geos_vel, levels=30, cmap='hot')
        plt.colorbar(label='Geostrophic Velocity')
        
        #Plot eddy borders
        # Create grid coordinates for interpolation
        x_grid = np.arange(geos_vel.shape[1])
        y_grid = np.arange(geos_vel.shape[0])
        
        for eddy in eddy_borders:
            # Plot each eddy's center and border
            center = eddy['center']
            border = eddy['border']
            #plt.plot(center[1], center[0], 'wx', markersize=8)  # Eddy center (lon, lat)
            #plt.plot(border[:, 1], border[:, 0], 'w-', linewidth=2)  # Eddy boundary (lon, lat)

            # Interpolate x and y coordinates to lon/lat independently
            border_x = np.interp(border[:, 1], x_grid, lon)
            border_y = np.interp(border[:, 0], y_grid, lat)
            plt.plot(border_x, border_y, c='w')
        
        plt.title('Eddy Detection Results')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(os.path.join(results_dir, 'eddy_detection.png'), dpi=300)
        plt.show()
        return