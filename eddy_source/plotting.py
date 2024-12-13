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
            center_x = np.interp(center[1], x_grid, lon)
            center_y = np.interp(center[0], y_grid, lat)
            border_x = np.interp(border[:, 1], x_grid, lon)
            border_y = np.interp(border[:, 0], y_grid, lat)
            plt.plot(border_x, border_y, c='w')
            plt.plot(center_x, center_y, 'wx', markersize=8)
        
        plt.title('Eddy Detection Results')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(os.path.join(results_dir, 'eddy_detection.png'), dpi=300)
        plt.show()
        return

    @staticmethod
    def plot_zoomed_eddy(ssh, u_geo, v_geo, eddy_info, lat, lon, zoom_radius=10):
        """Plot a zoomed view of a single eddy with SSH contours, velocity quiver plots, and eddy border.

        :param ssh: Sea surface height data
        :type ssh: numpy.ndarray
        :param u_geo: Zonal geostrophic velocity
        :type u_geo: numpy.ndarray
        :param v_geo: Meridional geostrophic velocity
        :type v_geo: numpy.ndarray
        :param eddy_info: Dictionary containing eddy information (center, border)
        :type eddy_info: dict
        :param lat: Latitude values
        :type lat: numpy.ndarray
        :param lon: Longitude values
        :type lon: numpy.ndarray
        :param zoom_radius: Radius of the zoomed region in grid points
        :type zoom_radius: int
        """
        # Get eddy center in grid coordinates
        center_y, center_x = eddy_info['center']
        
        # Define the zoomed region
        y_min = max(0, int(center_y - zoom_radius))
        y_max = min(ssh.shape[0], int(center_y + zoom_radius))
        x_min = max(0, int(center_x - zoom_radius))
        x_max = min(ssh.shape[1], int(center_x + zoom_radius))
        
        # Create the grid for quiver plot (subsample for clarity)
        subsample = 2  # Show every nth point
        y_grid, x_grid = np.mgrid[y_min:y_max:subsample, x_min:x_max:subsample]
        
        # Get the corresponding lat/lon values
        lon_zoom = lon[x_min:x_max]
        lat_zoom = lat[y_min:y_max]
        
        # Create a new figure
        plt.figure(figsize=(10, 8))
        
        # Plot SSH contours in the zoomed region
        plt.contour(lon_zoom, lat_zoom, 
                   ssh[y_min:y_max, x_min:x_max],
                   levels=20, colors='gray', alpha=0.5)
        
        # Plot velocity quivers
        u_zoom = u_geo[y_min:y_max:subsample, x_min:x_max:subsample]
        v_zoom = v_geo[y_min:y_max:subsample, x_min:x_max:subsample]
        lon_quiver = lon[x_min:x_max:subsample]
        lat_quiver = lat[y_min:y_max:subsample]
        
        plt.quiver(lon_quiver, lat_quiver, u_zoom, v_zoom,
                  alpha=0.5, scale=5, width=0.003)
        
        # Plot eddy border
        border = eddy_info['border']
        x_grid_full = np.arange(ssh.shape[1])
        y_grid_full = np.arange(ssh.shape[0])
        border_x = np.interp(border[:, 1], x_grid_full, lon)
        border_y = np.interp(border[:, 0], y_grid_full, lat)
        plt.plot(border_x, border_y, 'r-', linewidth=2, label='Eddy Border')
        
        # Plot eddy center
        center_lon = np.interp(center_x, x_grid_full, lon)
        center_lat = np.interp(center_y, y_grid_full, lat)
        plt.plot(center_lon, center_lat, 'rx', markersize=10, label='Eddy Center')
        
        # Set plot limits to the zoomed region
        plt.xlim(lon_zoom[0], lon_zoom[-1])
        plt.ylim(lat_zoom[0], lat_zoom[-1])
        
        plt.title('Zoomed Eddy View')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        
        # Save the plot
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'zoomed_eddy.png'), dpi=300)
        plt.show()