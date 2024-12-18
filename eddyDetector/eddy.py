from matplotlib import contour
import numpy as np
import logging
import xarray as xr
from typing import Dict, Tuple
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib.path import Path

class DetectEddiesSLD:
    def __init__(self, data):
        self.net_vel = data['net_vel']
        self.u = data['u']
        self.v = data['v']
        self.ssh = data['ssh']
        self.lon = data['lon']
        self.lat = data['lat']
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self.previous_point = None  # Store the previously selected point
        return

    def okubo_weiss(self):
        """
        Okubo-Weiss parameter calculation. Calculates OW parameter from normal strain (Sn) (∂u/∂x - ∂v/∂y),
        shear strain (Ss) (∂v/∂x + ∂u/∂y) and vorticity (ω)

        """
        duy, dux = np.gradient(self.u)   # ∂u/∂y and ∂u/∂x
        dvy, dvx = np.gradient(self.v)   # ∂v/∂y and ∂v/∂x
        self.vorticity = (dvx - duy)
        self.ow = (dux - dvy) ** 2 + (dvx + duy) ** 2 - self.vorticity ** 2  # Sn**2 + Ss**2 + ω**2
        self.logger.info(f"Computing Okubo-Weiss parameter with min value {np.nanmin(self.ow):.2e}, std value {np.nanstd(self.ow[self.ow < 0]):.2e}, and mean value {np.nanmean(self.ow[self.ow < 0]):.2e}")
        self.eddy_filter()
        return 
    
    def eddy_filter(self):
        """
        Filter the Okubo-Weiss field using a threshold value and separate masks for cyclone and anti-cyclone.
        several methods for masking eddies based on OW threshold parameters

        """
        methods = ['Chelton', 'Isern', 'Chaigneau']
        method = methods[1]

        # Chelton et al. 2007
        if method == 'Chelton':
            ow_mask = (self.ow <= -2e-12)
        # Isern-Fontanet et al. 2003 filter:
        elif method == 'Isern':
            threshold_u = -0.2 * np.nanstd(self.ow[self.ow < 0])
            ow_mask = (self.ow <= threshold_u)
        # Chaigneau et al. 2008 filter
        elif method == 'Chaigneau':
            threshold_u = -0.2 * np.nanstd(self.ow)
            threshold_l = -0.3 * np.nanstd(self.ow)
            ow_mask = (self.ow <= threshold_u) & (self.ow >= threshold_l)
        else:
            ow_mask = 1
            self.logger.warning(f'mask must be method from {methods}')

        # Assign mask
        self.ow *= ow_mask

        # Separate masks for cyclone and anti-cyclone depending on the vorticity polarity and magnitude
        self.cyc_mask = self.vorticity < 0
        self.acyc_mask = self.vorticity > 0
        return

    def eddy_algorithm(self):
        """
        Detect eddies using global minima in the Okubo-Weiss field.
        Invalid centers are screened out by masking the point.
        Valid centers result in masking the entire eddy region.

        :param max_eddies: maximum number of eddies to detect (default: 500)
        :return: 
            List of detected eddy information dictionaries
        """
        detected_eddies = []
        ow_mask = self.ow.copy()
        counter = -1
        max_eddies=500
        
        while len(detected_eddies) < max_eddies and counter < 1000:
            counter += 1

            # find current global minimum
            status, y, x = self.find_global_minima(ow_mask, n_minima=5)
            if status == 'break':
                break

            # Try to detect an eddy at this minimum
            [isEddy, eddy_info, ow_new] = self.screen_eddy(ow_mask, y, x)

            if isEddy:
                # Valid center - mask the entire eddy region
                ow_mask = ow_new
                detected_eddies.append(eddy_info)
                self.logger.info(f"Accepted eddy center at y, x: ({y}, {x}) for step {counter}")
            else:
                # Invalid center - just mask this single point
                ow_mask[y:y+2, x:x+2] = np.nan
        
        self.logger.info(f"Found {len(detected_eddies)} eddies")
        return detected_eddies

    def find_global_minima(self, ow_mask, n_minima=5):
        """
        Find global minima in the Okubo-Weiss field and select the furthest one from the previous point.

        :param ow_mask: Okubo-Weiss field with detected eddies masked out
        :param n_minima: Number of minimum points to consider
        :return: tuple (status, y, x)
            status: str - 'break' to end loop, 'ok' to proceed
            y, x: coordinates of the selected minimum
        """
        if np.all(np.isnan(ow_mask)):
            return 'break', None, None

        # Find the n smallest values and their indices
        flat_indices = np.argpartition(ow_mask.flatten(), n_minima)[:n_minima]
        valid_points = []
        
        for idx in flat_indices:
            y, x = np.unravel_index(idx, ow_mask.shape)
            # Check if point is too close to boundaries
            if y <= 3 or x <= 3 or y >= ow_mask.shape[0]-1 or x >= ow_mask.shape[1]-1:
                ow_mask[y, x] = np.nan
                continue
            valid_points.append((y, x))

        if not valid_points:
            return 'break', None, None

        # If this is the first point or no previous point exists
        if self.previous_point is None:
            y, x = valid_points[0]  # Take the first valid point
        else:
            # Calculate distances from previous point to all valid points
            prev_y, prev_x = self.previous_point
            distances = [np.sqrt((y - prev_y)**2 + (x - prev_x)**2) for y, x in valid_points]
            # Select the point with maximum distance
            max_dist_idx = np.argmax(distances)
            y, x = valid_points[max_dist_idx]

        # Update previous point
        self.previous_point = (y, x)
        return 'ok', y, x

    def screen_eddy(self, ow, center_y, center_x):
        """Detect and mask an eddy at the specified center point.
        
        :param ow: Okubo-Weiss parameter field
        :param center_y: y-coordinate of center
        :param center_x: x-coordinate of center
        :return: (is_eddy, eddy_info, masked_ow)
        """
        radius = 20
        m_points = 32
        # Check if center is already masked
        if np.isnan(ow[int(center_y), int(center_x)]):
            return False, None, ow
            
        # Get OW extent in x and y directions
        y_extent = ow[:, int(center_x)]
        x_extent = ow[int(center_y), :]
        
        # Normalize by center value to find extent
        c_value = ow[int(center_y), int(center_x)]
        x_norm = x_extent/c_value
        y_norm = y_extent/c_value
        
        # Find where values drop below threshold in both directions
        threshold = 0.0001
        
        # Find boundaries in x and y directions
        x_b = []
        y_b = []
        
        # Search in x direction
        for i in range(1, min(int(center_x), len(x_extent) - int(center_x), radius)):
            if x_norm[int(center_x)-i] < threshold:
                x_b.append(center_x-i)
                break
        for i in range(1, min(int(center_x), len(x_extent) - int(center_x), radius)):
            if x_norm[int(center_x)+i] < threshold:
                x_b.append(center_x+i)
                break
                
        # Search in y direction
        for i in range(1, min(int(center_y), len(y_extent) - int(center_y), radius)):
            if y_norm[int(center_y)-i] < threshold:
                y_b.append(center_y-i)
                break
        for i in range(1, min(int(center_y), len(y_extent) - int(center_y), radius)):
            if y_norm[int(center_y)+i] < threshold:
                y_b.append(center_y+i)
                break
                
        # Check if we found boundaries in both directions
        if len(x_b) != 2 or len(y_b) != 2:
            return False, None, ow
            
        # Calculate the semi-axes
        x_radius = (x_b[1] - x_b[0]) / 2
        y_radius = (y_b[1] - y_b[0]) / 2
        
        # Calculate the center of the ellipse
        x_center = (x_b[1] + x_b[0]) / 2
        y_center = (y_b[1] + y_b[0]) / 2
        
        # Generate points along the ellipse
        theta = np.linspace(0, 2*np.pi, m_points)
        contour_x = x_center + x_radius * np.cos(theta)
        contour_y = y_center + y_radius * np.sin(theta)
        
        # Stack points
        contour_points = np.column_stack((contour_y, contour_x))

        # Check if variables are symmetric across the center of the eddy
        symmetry_check = self.symmetry_check(contour_points)
        if not symmetry_check:
            return False, None, ow
        
        # Create mask using contour points
        mask = np.zeros_like(ow, dtype=bool)
        
        # Create a grid of points to check
        y_indices, x_indices = np.mgrid[:ow.shape[0], :ow.shape[1]]
        points = np.column_stack((y_indices.ravel(), x_indices.ravel()))
        
        # Check which points are inside the contour
        inside_points = Path(contour_points).contains_points(points)
        mask = inside_points.reshape(ow.shape)
        
        
        # Apply the mask
        new_ow = ow.copy()
        new_ow[mask] = np.nan

        # Check if anticyclone or cyclone
        if np.sum(self.vorticity[mask]) > 0:
            eddy_cylone = "anticyclonic"
        elif np.sum(self.vorticity[mask]) < 0:
            eddy_cylone = "cyclonic"


        # Calculate eddy properties
        eddy_info = {
            'center': (y_center, x_center),
            'radius': (x_radius, y_radius),
            'eddy_cylone': eddy_cylone,
            'ssh_center': self.ssh[int(y_center), int(x_center)],
            'net_vel_center': self.net_vel[int(y_center), int(x_center)],
            'kinetic_energy': np.nanmean(self.net_vel[mask]),
            'border': contour_points,
            'mask': mask,
            'n_points': np.sum(mask),
            'vorticity': np.nanmean(self.vorticity[mask]),
            'vorticity_std': np.nanstd(self.vorticity[mask]),
            'ow': np.nanmean(self.ow[mask]),
            'ow_std': np.nanstd(self.ow[mask])
        }
        return True, eddy_info, new_ow

    def symmetry_check(self, points):
        """
        Check if a contour is symmetric around a center point.

        :param points: Contour points
        :return: True if contour is symmetric, False otherwise
        """
       
        # Extract border points and corresponding values
        y = points[:, 0].astype(int)
        x = points[:, 1].astype(int)
        n_points = len(y)
        u_geo_border = self.u[y, x]
        v_geo_border = self.v[y, x]
        ssh_border = self.ssh[y, x]
        net_vel_border = self.net_vel[y, x]

        if any(np.isnan(u_geo_border)) or any(np.isnan(v_geo_border)) or any(np.isnan(ssh_border)) or any(np.isnan(net_vel_border)):
            return False
    
        # New check for opposite points using np.roll
        u_geo_rolled = np.roll(u_geo_border, n_points // 2)
        v_geo_rolled = np.roll(v_geo_border, n_points // 2)
        ssh_rolled = np.roll(ssh_border, n_points // 2)
        net_vel_rolled = np.roll(net_vel_border, n_points // 2)

        # Compute cosine similarity between opposite points
        dot_products_opposite = np.array([np.dot([u1, v1], [u2, v2]) 
                                    for u1, v1, u2, v2 in zip(u_geo_border, v_geo_border, u_geo_rolled, v_geo_rolled)])
        # Compute cosine similarity between opposite points
        norms_orig = np.linalg.norm(np.column_stack((u_geo_border, v_geo_border)), axis=1)
        norms_rolled = np.linalg.norm(np.column_stack((u_geo_rolled, v_geo_rolled)), axis=1)
        cos_similarities_opposite = dot_products_opposite / (norms_orig * norms_rolled)

        # Compute ratio between opposite points
        ratio_ssh_rolled = np.abs(ssh_rolled) / np.abs(ssh_border)
        ratio_net_vel_rolled = np.abs(net_vel_rolled) / np.abs(net_vel_border)
        
        # Check if contour is symmetric
        velocity_symmetry_check = sum(np.abs(cos_similarities_opposite)<0.5) 
        ssh_symmetry_check = sum((ratio_ssh_rolled < 1/1.5) | (ratio_ssh_rolled > 1.5)) 
        net_vel_symmetry_check = sum((ratio_net_vel_rolled < 1/1.5) | (ratio_net_vel_rolled > 1.5)) 
        if (velocity_symmetry_check/n_points < 0.8) and (ssh_symmetry_check/n_points < 0.9) and (net_vel_symmetry_check/n_points < 0.9):
            return True 
        else:
            return False

class OutputSLD:
    def __init__(self, eddy_info):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self.eddy_info = eddy_info
        return

    def store_properties(self):
        self.logger.info(f"Storing eddy properties")
        return
