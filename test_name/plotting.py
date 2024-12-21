import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator, interp1d
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import logging

class PlotEddy:
    def __init__(self, data, eddy_info):
        self.net_vel = data['net_vel']
        self.u = data['u']
        self.v = data['v']
        self.ssh = data['ssh']
        self.lon = data['lon']
        self.lat = data['lat']
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self.eddy_info = eddy_info
        self.plot_eddy_detection(self.ssh, self.net_vel, self.eddy_info, self.lat, self.lon)
        return
    

    @staticmethod
    def plot_eddy_detection(ssh, geos_vel, eddy_borders, lat, lon):
        """
        Plot detected eddies with geographic features.
        
        :param ssh: Sea surface height data
        :param geos_vel: Geostrophic velocity data
        :param eddy_borders: List of detected eddy borders
        :param lat: Latitude coordinates
        :param lon: Longitude coordinates
        """
        # Create figure with a specific projection
        plot_transform = ccrs.PlateCarree()

        plt.figure(figsize=(15, 10))
        ax = plt.axes(projection=plot_transform)

        # Plot SSH
        ssh_plot = ax.contour(lon, lat, ssh, 
                             transform=plot_transform,
                             levels=60,
                             colors='gray',
                             alpha=0.5)
        
        # Plot velocity magnitude
        vel_plot = ax.contourf(lon, lat, geos_vel, 
                              transform=plot_transform,
                              levels=30,
                              cmap='hot')
        plt.colorbar(vel_plot, label='Geostrophic Velocity')
        
        # Plot eddy borders
        for eddy in eddy_borders:
            if 'border' in eddy:
                border_points = eddy['border']
                if len(border_points) > 0:
                    border_y = border_points[:, 0].astype(int)
                    border_x = border_points[:, 1].astype(int)
                    
                    # Plot the border directly
                    ax.plot(lon[border_x], lat[border_y], 
                           c='w', 
                           linewidth=2,
                           label='Eddy Border',
                           transform=plot_transform)

        # Now add coastlines and other features on top
        ax.coastlines(resolution='10m', linewidth=1)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=3)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3, zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
        ax.gridlines(draw_labels=True)

        # Set plot extent to data bounds with some padding
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], 
                     crs=plot_transform)

        # Add title and labels
        plt.title('Eddy Detection with Geographic Features')
        
        # Save the figure
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'eddy_detection_with_geography.png'), 
                    dpi=300, 
                    bbox_inches='tight')
        plt.show()
        return

    @staticmethod
    def plot_zoomed_eddy(ssh, u_geo, v_geo, eddy_info, lat, lon, zoom_radius=20):
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

        plt.contourf(lon_zoom, lat_zoom, 
                    np.sqrt(u_geo[y_min:y_max, x_min:x_max]**2 + v_geo[y_min:y_max, x_min:x_max]**2), 
                    levels=30, cmap='hot')       
        plt.colorbar(label='Geostrophic Velocity')
        
        # Plot velocity quivers
        u_zoom = u_geo[y_min:y_max:subsample, x_min:x_max:subsample]
        v_zoom = v_geo[y_min:y_max:subsample, x_min:x_max:subsample]
        lon_quiver = lon[x_min:x_max:subsample]
        lat_quiver = lat[y_min:y_max:subsample]
        
        plt.quiver(lon_quiver, lat_quiver, u_zoom, v_zoom,
                  alpha=0.5, scale=20, width=0.003)
        
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
        return

    @staticmethod
    def plot_zoomed_eddy_with_contours(ssh, u_geo, v_geo, eddy_info, lat, lon, ow, zoom_radius=20):
        """
        Plot the zoomed-in view of an eddy along with the closed contours of the Okubo-Weiss parameter.

        :param ssh: Sea surface height data
        :param u_geo: U component of geostrophic velocity
        :param v_geo: V component of geostrophic velocity
        :param eddy_info: Information about the detected eddy
        :param lat: Latitude coordinates
        :param lon: Longitude coordinates
        :param ow: Okubo-Weiss parameter data
        :param zoom_radius: Radius for zooming in on the eddy
        """
        
        # Get eddy center in grid coordinates
        center_y, center_x = eddy_info['center']
        
        # Define the zoomed region
        y_min = max(0, int(center_y - zoom_radius))
        y_max = min(ssh.shape[0], int(center_y + zoom_radius))
        x_min = max(0, int(center_x - zoom_radius))
        x_max = min(ssh.shape[1], int(center_x + zoom_radius))

        # Get the corresponding lat/lon values
        lon_zoom = lon[x_min:x_max]
        lat_zoom = lat[y_min:y_max]
        
        # Create a figure and axis for plotting
        plt.figure(figsize=(10, 8))
        
        # Plot the Okubo-Weiss contours
        contour = plt.contour(lon_zoom, lat_zoom, ow[y_min:y_max, x_min:x_max], levels=20, cmap='viridis')
        plt.clabel(contour, inline=True, fontsize=8)

        # Overlay the SSH data
        plt.imshow(ssh[y_min:y_max, x_min:x_max], extent=(lon[x_min], lon[x_max], lat[y_max], lat[y_min]), origin='upper', alpha=0.5, cmap='Blues')

        # Plot the eddy center
        #plt.scatter(lon[center_x], lat[center_y], color='red', marker='o', label='Eddy Center')

        # Plot velocity quivers
        subsample = 2  # Show every nth point
        u_zoom = u_geo[y_min:y_max:subsample, x_min:x_max:subsample]
        v_zoom = v_geo[y_min:y_max:subsample, x_min:x_max:subsample]
        lon_quiver = lon[x_min:x_max:subsample]
        lat_quiver = lat[y_min:y_max:subsample]
        
        plt.quiver(lon_quiver, lat_quiver, u_zoom, v_zoom,
                  alpha=0.5, scale=20, width=0.003)

        # Add titles and labels
        plt.title('Zoomed Eddy with Okubo-Weiss Contours')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        
        # Save the figure
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'zoomed_eddy_with_contours.png'), dpi=300)
        plt.show()
        return
