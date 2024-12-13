import numpy as np
import xarray as xr
from typing import Dict, Tuple
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.path import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class EddyMethods:
    def __init__(self):
        pass

    @staticmethod
    def global_detect(ow, mag_uv, u_geo, v_geo, ssh, max_eddies=500):
        """
        Detect eddies using global minima in the Okubo-Weiss field.
        Invalid centers are screened out by masking the point.
        Valid centers result in masking the entire eddy region.

        :param ow: Okubo-Weiss parameter for 2D field
        :param mag_uv: net velocity for 2D field
        :param u_geo: zonal component of geostrophic velocity (2D)
        :param v_geo: meridional component of geostrophic velocity (2D)
        :param ssh: sea surface height (2D)
        :param max_eddies: maximum number of eddies to detect (default: 500)
        :return: List of detected eddy information dictionaries
        """
        detected_eddies = []
        current_ow = ow.copy()
        counter = -1
        
        while len(detected_eddies) < max_eddies and counter < 1000:
            counter += 1
            # Find the current global minimum
            if np.all(np.isnan(current_ow)):
                break
                
            # Get the indices of the minimum value, ignoring NaNs
            flat_idx = np.nanargmin(current_ow)
            y, x = np.unravel_index(flat_idx, current_ow.shape)
            
            # Try to detect an eddy at this minimum
            [isEddy, eddy_info, new_ow] = EddyMethods.detect_and_mask_eddy(
                current_ow, y, x, mag_uv, u_geo, v_geo, ssh, radius=7, m_points=32
            )
            
            if isEddy:
                # Valid center - mask the entire eddy region
                current_ow = new_ow
                detected_eddies.append(eddy_info)
                #EddyMethods.plot_eddy_info(ssh, mag_uv, eddy_info)
                logging.info(f"Detected eddy at: ({y}, {x})")
            else:
                # Invalid center - just mask this point
                current_ow[y-2:y+2, x-2:x+2] = np.nan
                logging.info(f"Screened out center at: ({y}, {x})")
        
        print(f"Found {len(detected_eddies)} eddies")
        return detected_eddies

    @staticmethod
    def slide_detect(ow, mag_uv, u_geo, v_geo, ssh, block_size, subgrid_size):
        """
        slides across ow.shape grid with n x n block size and first selects the ow minimum. it then uses an m x m grid
        centred on ow minimum to search for mag_uv minimum

        :param ssh:
        :param v_geo: meridional component of geostrophic velocity (2D)
        :param u_geo: zonal component of geostrophic velocity (2D)
        :param subgrid_size: m x m grid centred on ow local minimum to search uv minimum
        :param block_size: n x n block used as slider to detect local ow minimum
        :param ow: okubo-weiss parameter for 2D field
        :param mag_uv: net velocity for 2D field
        :return: potential_centres

        """
        # slide over a region
        # Calculate number of blocks in each dimension
        slice_y, slice_x = EddyMethods.block_indices(ow, block_size)
        eddy_centres = []
        eddy_borders = []
        for counter, (slice_yi, slice_xi) in enumerate(zip(slice_y, slice_x)):
            # counter to act as block id
            print(f"counter: {counter}, slice y: {slice_yi}, slice x: {slice_xi}")
            ow_i = ow[slice_yi, slice_xi]

            # check if okubo-weiss values are invalid for eddy detection
            if np.all(np.isnan(ow_i)) or np.all(np.nan_to_num(ow_i) == 0):
                print(f"all ow values either nan or zeros")
                continue
            else:
                # todo: find charismatic eddy- use as example;
                # find minima index for ow variable
                uv_centre_y, uv_centre_x = EddyMethods.find_uv_centre(ow_i, ssh, slice_yi, slice_xi, subgrid_size, c_method="other")
                #uv_centre_y, uv_centre_x = find_uv_centre(ow_i, ow, slice_yi, slice_xi, subgrid_size)
                #uv_centre_y, uv_centre_x = find_uv_centre(ow_i, mag_uv, slice_yi, slice_xi, subgrid_size)

                # now screen this centre:ow
                [isEddy, eddy_border] = EddyMethods.screen_centre(uv_centre_y, uv_centre_x, mag_uv, u_geo, v_geo, ssh)
                if isEddy:
                    # expand borders of screened eddies
                    eddy_boundary = EddyMethods.expand_borders(uv_centre_y, uv_centre_x, mag_uv, u_geo, v_geo, ssh, eddy_border)
                    eddy_centres.append([uv_centre_y, uv_centre_x])
                    eddy_borders.append(eddy_boundary)

        # potential_centres = np.array(eddy_centres)
        # plt.pcolormesh(mag_uv)
        # plt.scatter(potential_centres[:,1], potential_centres[:, 0], c='r')
        return np.array(eddy_centres), np.array(eddy_borders)

    @staticmethod
    def find_global_minima_with_masking(ow_field, max_eddies=500, mask_radius=10):
        """
        Iteratively find global minima in the Okubo-Weiss field, masking previously found points.

        :param ow_field: 2D Okubo-Weiss field
        :type ow_field: numpy.ndarray
        :param max_eddies: Maximum number of eddies to find
        :type max_eddies: int, optional
        :param mask_radius: Radius of masking around each found minimum
        :type mask_radius: int, optional
        :return: Tuple containing (global_minima, masked_ow) where global_minima is a list of (y, x) coordinates 
                of global minima and masked_ow is the Okubo-Weiss field with found minima masked
        :rtype: tuple(list, numpy.ndarray)
        """

        # Create a copy of the original field to modify
        masked_ow = ow_field.copy()
        # Mask out zero values
        masked_ow[masked_ow == 0] = np.inf
        global_minima = []

        for _ in range(max_eddies):
            # Find the global minimum
            min_idx = np.unravel_index(np.nanargmin(masked_ow), masked_ow.shape)
            global_minima.append(min_idx)

            # Create a mask around the minimum point
            mask = np.zeros_like(masked_ow, dtype=bool)
            y, x = min_idx
            
            # Create a grid of coordinates
            y_grid, x_grid = np.ogrid[:masked_ow.shape[0], :masked_ow.shape[1]]
            
            # Create circular mask
            mask_area = (y_grid - y)**2 + (x_grid - x)**2 <= mask_radius**2
            
            # Set masked area to a high value to prevent future selection
            masked_ow[mask_area] = np.inf

            # Stop if no more significant minima
            if np.nanmin(masked_ow) == np.inf:
                break


        return global_minima, masked_ow
    @staticmethod
    def expand_borders(uv_centre_y, uv_centre_x, mag_uv, u_geo, v_geo, ssh, eddy_border, lat=None, lon=None):
        """
        expands border of eddies that have passed the first screening;

        :param uv_centre_y:
        :param uv_centre_x:
        :param mag_uv:
        :param u_geo:
        :param v_geo:
        :param eddy_border:
        :return:
        """
        # define contour based on eddy border from screening
        contour_p = eddy_border
        rad_x = 4
        rad_y = 4
        max_rad = 15

        # counted at least at 3- I should count the eddies found with this method;
        x_expansion = True
        y_expansion = True
        break_x_expansion = False
        break_y_expansion = False

        while (rad_x < max_rad) and (rad_y < max_rad) and (x_expansion or y_expansion):
            if (rad_x >= max_rad) and (rad_y >= max_rad):
                break
            if x_expansion:
                rad_x += 1
                [break_x_expansion, x_border, y_border] = EddyMethods.perimeter_check(
                    uv_centre_y, uv_centre_x, mag_uv, u_geo, v_geo, ssh, rad_x, rad_y
                )
                if not break_x_expansion:
                    contour_p = np.array([x_border, y_border]).T
                else:
                    x_expansion = False

            if y_expansion:
                rad_y += 1
                [break_y_expansion, x_border, y_border] = EddyMethods.perimeter_check(
                    uv_centre_y, uv_centre_x, mag_uv, u_geo, v_geo, ssh, rad_x, rad_y
                )
                if not break_y_expansion:
                    contour_p = np.array([x_border, y_border]).T
                else:
                    y_expansion = False
        return contour_p

    @staticmethod
    def perimeter_check(uv_centre_y, uv_centre_x, mag_uv, u_geo, v_geo, ssh, rad_x, rad_y):
        """
        Check the perimeter of a potential eddy for various criteria.

        :param uv_centre_y: Y-coordinate of eddy center
        :type uv_centre_y: int
        :param uv_centre_x: X-coordinate of eddy center
        :type uv_centre_x: int
        :param mag_uv: Magnitude of geostrophic velocity
        :type mag_uv: numpy.ndarray
        :param u_geo: U component of geostrophic velocity
        :type u_geo: numpy.ndarray
        :param v_geo: V component of geostrophic velocity
        :type v_geo: numpy.ndarray
        :param ssh: Sea surface height
        :type ssh: numpy.ndarray
        :param rad_x: X radius of ellipse
        :type rad_x: int
        :param rad_y: Y radius of ellipse
        :type rad_y: int
        :param lat: Latitude coordinates
        :type lat: numpy.ndarray, optional
        :param lon: Longitude coordinates
        :type lon: numpy.ndarray, optional
        :return: tuple of (break_while_loop, x_border, y_border)
        :rtype: tuple(bool, numpy.ndarray, numpy.ndarray)
        """
        circle_iterations = 30
        Sv = 1.1
        break_while_loop = False
        x_border = np.zeros(circle_iterations)
        y_border = np.zeros(circle_iterations)
        uv_border = np.zeros(circle_iterations)
        u_geo_border = np.zeros(circle_iterations)
        v_geo_border = np.zeros(circle_iterations)
        ssh_border = np.zeros(circle_iterations)
        for circle_it, angle in enumerate(np.linspace(0, 2 * np.pi, circle_iterations)):
            x_border[circle_it] = uv_centre_x + (rad_x * np.cos(angle))
            y_border[circle_it] = uv_centre_y + (rad_y * np.sin(angle))
            if ((x_border[circle_it] <= 0) or (y_border[circle_it] <= 0) or (
                    int(x_border[circle_it]) >= mag_uv.shape[1]-1)
                    or (int(y_border[circle_it]) >= mag_uv.shape[0]-1)):
                return True, None, None
        
            
            uv_border[circle_it] = mag_uv[int(y_border[circle_it]), int(x_border[circle_it])]
            u_geo_border[circle_it] = u_geo[int(y_border[circle_it]), int(x_border[circle_it])]
            v_geo_border[circle_it] = v_geo[int(y_border[circle_it]), int(x_border[circle_it])]
            ssh_border[circle_it] = ssh[int(y_border[circle_it]), int(x_border[circle_it])]
            # Check for NaN values
            if np.isnan(uv_border[circle_it]):
                return True, None, None    

        # cosine similarity using np.roll
        v1 = np.array([u_geo_border, v_geo_border]).T
        v2 = np.roll(v1, 1, axis=0)
        dot_product = np.sum(v1 * v2, axis=1)
        mv1 = np.linalg.norm(v1, axis=1)
        mv2 = np.linalg.norm(v2, axis=1)
        cos_theta = dot_product / (mv1 * mv2)
        uv_dot = cos_theta

        # New check for opposite points using np.roll
        u_geo_rolled = np.roll(u_geo_border, circle_iterations // 2)
        v_geo_rolled = np.roll(v_geo_border, circle_iterations // 2)

        # Compute cosine similarity between opposite points
        dot_products_opposite = np.array([np.dot([u1, v1], [u2, v2]) 
                                    for u1, v1, u2, v2 in zip(u_geo_border, v_geo_border, u_geo_rolled, v_geo_rolled)])

        norms_orig = np.linalg.norm(np.column_stack((u_geo_border, v_geo_border)), axis=1)
        norms_rolled = np.linalg.norm(np.column_stack((u_geo_rolled, v_geo_rolled)), axis=1)
        cos_similarities_opposite = dot_products_opposite / (norms_orig * norms_rolled)

        # ssh rolled:
        ssh_rolled = np.roll(ssh_border, circle_iterations // 2)
        ratio_ssh_rolled = np.abs(ssh_rolled) / np.abs(ssh_border)

        # ratio of velocity and ssh on border
        ratio_ssh = np.abs(ssh_border[1::]) / np.abs(ssh_border[0:-1])
        ratio_vel = uv_border[1::] / uv_border[0:-1]

        # criteria for eddy
        #todo: focus on symmetry checks- ssh and velocity
        velocity_symmetry_check = sum(np.abs(cos_similarities_opposite)<0.5) 
        ssh_symmetry_check = sum((ratio_ssh_rolled < 1/1.5) | (ratio_ssh_rolled > 1.5))    
        direction_check = sum(uv_dot < 0.99)
        velocity_check = sum((ratio_vel < 1 / Sv) | (ratio_vel > Sv))
        ssh_check = sum((ratio_ssh < 1 / Sv) | (ratio_ssh > Sv))
        velocity_cv = (np.std(uv_border)/np.mean(uv_border))*100
        ssh_cv = (np.std(np.abs(ssh_border)) / np.mean(np.abs(ssh_border)))*100

        # screening based on criteria:
        # 1. velocity ratio between consecutive points
        # 2. direction change between consecutive points
        # 3. symmetry check for opposite points
        # 4. ssh ratio between consecutive points
        # 5 and 6: coefficient of variation of velocity and ssh on the border
        criteria_1 = velocity_check/circle_iterations < 0.5
        criteria_2 = direction_check/circle_iterations < 0.9
        criteria_3 = velocity_symmetry_check/circle_iterations < 0.5
        criteria_4 = ssh_check/circle_iterations < 0.9
        criteria_5 = ssh_symmetry_check/circle_iterations < 0.5
        criteria_6 = ssh_cv < 90

        # todo: add eddy_identifier
        # if rad_x == 10 and rad_y == 10:
        #     EddyMethods.plot_eddy(uv_centre_y, uv_centre_x, mag_uv, ssh, rad_x, rad_y, x_border, y_border, u_geo_border, v_geo_border)
        #     breakpoint()

        if not all([criteria_1, criteria_2, criteria_3, criteria_4, criteria_5, criteria_6]):
            # print(f"symmetry check: {symmetry_check}")
            # print(f"direction check: {direction_check}")
            # print(f"velocity check: {velocity_check}")
            # print(f"ssh check: {ssh_check}")
            # print(f"velocity cv: {velocity_cv}")
            # print(f"ssh cv: {ssh_cv}")                        
            return True, None, None

        return False, np.array(x_border), np.array(y_border)

    @staticmethod
    def screen_centre(uv_centre_y, uv_centre_x, mag_uv, u_geo, v_geo, ssh):
        """
        function screens centres based on velocity criteria

        :param uv_centre_x:
        :param uv_centre_y:
        :param u_geo:
        :param v_geo:
        :param mag_uv:
        :param ssh:
        :return: screened centre, and x,y boundaries if eddy goes through screen
        """

        is_eddy = False
        break_while_loop = False 
        rad_x = rad_y = 3
        contour_p = []
        [break_while_loop, x_border, y_border] = EddyMethods.perimeter_check(    
            uv_centre_y,
            uv_centre_x,
            mag_uv,
            u_geo,
            v_geo,
            ssh, rad_x, rad_y)

        if not break_while_loop:
            contour_p = [(x, y) for x, y in zip(x_border, y_border)]
            #[contour_p.append((i, j)) for i, j in zip(x_border, y_border)]
            is_eddy = True

        return is_eddy, np.array(contour_p)

    @staticmethod
    def block_indices(grid, block_size):
        """

        :param grid:
        :param block_size:
        :return: slices for each box
        """
        n_blocks_y = int(np.ceil(grid.shape[0] / block_size))
        n_blocks_x = int(np.ceil(grid.shape[1] / block_size))
        slice_y = []
        slice_x = []
        # Vectorized block processing
        for i in range(n_blocks_y):
            for j in range(n_blocks_x):
                # Calculate block boundaries
                i_start = i * block_size
                i_end = min((i + 1) * block_size, grid.shape[0])
                j_start = j * block_size
                j_end = min((j + 1) * block_size, grid.shape[1])
                slice_y.append(slice(i_start, i_end, 1))
                slice_x.append(slice(j_start, j_end, 1))
        return slice_y, slice_x

    @staticmethod
    def find_uv_centre(ow_i, mag_uv, slice_yi, slice_xi, subgrid_size, c_method):
        # first find the minimum ow value in the block
        ow_centre_y, ow_centre_x = np.unravel_index(np.nanargmin(ow_i), ow_i.shape)

        # if we allow these centres, look for it's index on a larger grid;
        ow_centre_y += slice_yi.start
        ow_centre_x += slice_xi.start

        # create a subgrid for selection of velocity minimum
        [y_slice, x_slice] = EddyMethods.slice_subgrid(mag_uv, ow_centre_x, ow_centre_y, subgrid_size)

        # search for a local minimum velocity
        uv_slice = mag_uv[y_slice, x_slice]
        if c_method == "ssh":
            uv_slice = np.abs(uv_slice)
            uv_centre_y, uv_centre_x = np.unravel_index(np.nanargmax(uv_slice), uv_slice.shape)
        else:
            uv_centre_y, uv_centre_x = np.unravel_index(np.nanargmin(uv_slice), uv_slice.shape)
        uv_centre_y += y_slice.start
        uv_centre_x += x_slice.start

        return uv_centre_y, uv_centre_x

    @staticmethod
    def slice_subgrid(grid, center_x, center_y, subgrid_size):
        half_size = subgrid_size // 2
        start_x = max(center_x - half_size, 0)
        end_x = min(center_x + half_size + 1, grid.shape[1])
        start_y = max(center_y - half_size, 0)
        end_y = min(center_y + half_size + 1, grid.shape[0])
        return slice(start_y, end_y, 1), slice(start_x, end_x, 1)

    @staticmethod
    def eddy_filter(ow, vorticity):
        """
        several methods for masking eddies based on OW threshold parameters

        :arg ow: raw okubo-weiss values
        :arg vorticity: vorticity value calculated in okubo-weiss
        :returns ow2: filter okubo-weiss variable
        """
        ow2 = np.copy(ow)  # copy the original matrix
        methods = ['Chelton', 'Isern', 'Chaigneau']
        method = methods[1]

        # Chelton et al. 2007
        if method == 'Chelton':
            ow_mask = (ow <= -2e-12)
        # Isern-Fontanet et al. 2003 filter:
        elif method == 'Isern':
            threshold_u = -0.2 * np.nanstd(ow[ow < 0])
            ow_mask = (ow <= threshold_u)
        # Chaigneau et al. 2008 filter
        elif method == 'Chaigneau':
            threshold_u = -0.2 * np.nanstd(ow)
            threshold_l = -0.3 * np.nanstd(ow)
            ow_mask = (ow <= threshold_u) & (ow >= threshold_l)
        else:
            ow_mask = 1
            print('mask must be method from ' + str(methods))

        # Assign mask
        ow2 *= ow_mask

        # Separate masks for cyclone and anti-cyclone depending on the vorticity polarity and magnitude
        cyc_mask = vorticity < 0
        acyc_mask = vorticity > 0

        return ow2, cyc_mask, acyc_mask

    @staticmethod
    def calculate_okubo_weiss(u_geo, v_geo):
        """
        Okubo-Weiss parameter calculation. Calculates OW parameter from normal strain (Sn) (∂u/∂x - ∂v/∂y),
        shear strain (Ss) (∂v/∂x + ∂u/∂y) and vorticity (ω)

        :arg u_geo: u component of velocity (j x i)
        :arg v_geo: v component of velocity field (j x i)
        :returns OW: Okubo-Weiss field (j x i)
        """
        duy, dux = np.gradient(u_geo, 1, 1)   # ∂u/∂y and ∂u/∂x
        dvy, dvx = np.gradient(v_geo, 1, 1)   # ∂v/∂y and ∂v/∂x
        vorticity = (dvx - duy)
        ow = (dux - dvy) ** 2 + (dvx + duy) ** 2 - vorticity ** 2  # Sn**2 + Ss**2 + ω**2
        return ow, vorticity

    @staticmethod
    def plot_eddy(uv_centre_y, uv_centre_x, mag_uv, ssh, rad_x, rad_y, x_border, y_border, u_geo_border, v_geo_border):
        """Plot the detected eddy with SSH contours and velocity magnitude.
        :param uv_centre_y: Y-coordinate of eddy center
        :type uv_centre_y: int
        :param uv_centre_x: X-coordinate of eddy center
        :type uv_centre_x: int
        :param mag_uv: Magnitude of geostrophic velocity
        :type mag_uv: numpy.ndarray
        :param ssh: Sea surface height
        :type ssh: numpy.ndarray
        :param rad_x: X radius of ellipse
        :type rad_x: int
        :param rad_y: Y radius of ellipse
        :type rad_y: int
        :param x_border: X border coordinates of the eddy
        :type x_border: numpy.ndarray
        :param y_border: Y border coordinates of the eddy
        :type y_border: numpy.ndarray
        """   

        # Define plot region around eddy
        margin = max(rad_x, rad_y) + 5
        y_min = max(0, int(uv_centre_y - margin))
        y_max = min(mag_uv.shape[0], int(uv_centre_y + margin))
        x_min = max(0, int(uv_centre_x - margin))
        x_max = min(mag_uv.shape[1], int(uv_centre_x + margin))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot SSH contours and velocity magnitude in the region
        plt.contour(x_min + np.arange(x_max-x_min), y_min + np.arange(y_max-y_min), 
                    ssh[y_min:y_max, x_min:x_max], levels=60, colors='grey', alpha=0.5)
        g_vel = plt.contourf(x_min + np.arange(x_max-x_min), y_min + np.arange(y_max-y_min),
                        mag_uv[y_min:y_max, x_min:x_max], levels=30, cmap='hot')
            
        # Plot eddy border and center
        plt.plot(x_border, y_border, 'w-o', linewidth=2, markersize=4)
        plt.plot(uv_centre_x, uv_centre_y, 'wo', markersize=8)
        plt.quiver(x_border[::2], y_border[::2], u_geo_border[::2], v_geo_border[::2],
                color='k', scale=10, alpha=0.7)
            
        fig.colorbar(g_vel, ax=ax, label='Geostrophic Velocity')
        plt.title('Eddy Detection Check')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.show()
        return

    @staticmethod
    def plot_local_ow(ow, center_y, center_x, radius=3, m_points=32, ssh=None, u_geo=None, v_geo=None):
        """
        Plot the Okubo-Weiss parameter in a region around a specified center point with SSH contours and velocity vectors.
        
        :param ow: 2D Okubo-Weiss parameter field
        :param center_y: Y-coordinate of center point
        :param center_x: X-coordinate of center point
        :param radius: Radius of region to plot around center (default: 3)
        :param m_points: Number of points to use for the contour (default: 32)
        :param ssh: Sea surface height field (optional)
        :param u_geo: U component of geostrophic velocity (optional)
        :param v_geo: V component of geostrophic velocity (optional)
        """
        # Define the region to plot
        y_min = max(0, int(center_y - radius))
        y_max = min(ow.shape[0], int(center_y + radius + 1))
        x_min = max(0, int(center_x - radius))
        x_max = min(ow.shape[1], int(center_x + radius + 1))
        
        # Extract the local region
        local_ow = ow[y_min:y_max, x_min:x_max]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot OW parameter
        plt.imshow(local_ow, cmap='Reds_r', origin='lower')
        plt.colorbar(label='Okubo-Weiss Parameter')
        
        # Add SSH contours if available
        if ssh is not None:
            local_ssh = ssh[y_min:y_max, x_min:x_max]
            x_grid = np.arange(x_min, x_max)
            y_grid = np.arange(y_min, y_max)
            plt.contour(x_grid - x_min, y_grid - y_min, local_ssh, 
                       levels=20, colors='grey', alpha=0.5, linewidths=0.5)
        
        # Add velocity vectors if available
        if u_geo is not None and v_geo is not None:
            local_u = u_geo[y_min:y_max, x_min:x_max]
            local_v = v_geo[y_min:y_max, x_min:x_max]
            
            # Create a grid for quiver plot (subsample for clarity)
            step = 2
            y_indices, x_indices = np.mgrid[0:local_u.shape[0]:step, 0:local_u.shape[1]:step]
            
            plt.quiver(x_indices, y_indices, 
                      local_u[::step, ::step], local_v[::step, ::step],
                      color='k', scale=20, alpha=0.3)
        
         # Proceed with full analysis
        # (rest of the method remains unchanged)
        theta = np.linspace(0, 2*np.pi, m_points)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        circle_x = center_x - x_min + radius * cos_theta
        circle_y = center_y - y_min + radius * sin_theta
        
        points = np.column_stack((circle_y, circle_x))
        valid_points = (points[:, 0] >= 0) & (points[:, 0] < local_ow.shape[0]) & \
                      (points[:, 1] >= 0) & (points[:, 1] < local_ow.shape[1])
        points = points[valid_points]
        
        # Get OW values at the points directly
        ow_vals = local_ow[points[:, 0].astype(int), points[:, 1].astype(int)]
        valid_mask = ~np.isnan(ow_vals)
        
        # Get OW values at circle points
        contour_points = []
        for x, y in zip(circle_x, circle_y):
            if 0 <= x < local_ow.shape[1] and 0 <= y < local_ow.shape[0]:
                ow_val = ow_vals
                # Scale the radius based on OW value
                scale = np.clip(1.0 - ow_val/np.max(np.abs(local_ow)), 0.5, 1.5)
                contour_points.append((x + (radius * (scale-1) * np.cos(theta[len(contour_points)])),
                                    y + (radius * (scale-1) * np.sin(theta[len(contour_points)]))))
        
        if contour_points:
            contour_x, contour_y = zip(*contour_points)
            # Close the contour
            contour_x = list(contour_x) + [contour_x[0]]
            contour_y = list(contour_y) + [contour_y[0]]
            plt.plot(contour_x, contour_y, 'b-', label='OW Contour')
        
        # Mark the center point
        plt.plot(center_x - x_min, center_y - y_min, 'k+', markersize=10, label='Center')
        
        plt.title(f'Local Okubo-Weiss Parameter (r={radius})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
        return

    @staticmethod
    def detect_and_mask_eddy(ow, center_y, center_x, mag_uv, u_geo, v_geo, ssh, radius, m_points=32):
        """
        Detect and mask an eddy in the Okubo-Weiss field.
        
        :param ow: 2D Okubo-Weiss parameter field
        :param center_y: Y-coordinate of center point
        :param center_x: X-coordinate of center point
        :param mag_uv: Magnitude of geostrophic velocity
        :param u_geo: U component of geostrophic velocity
        :param v_geo: V component of geostrophic velocity
        :param ssh: Sea surface height field
        :param radius: Radius of eddy to detect
        :param m_points: Number of points to use for contour analysis
        :return: True if eddy is detected, False otherwise
        """ 
        
        logging.info(f"Starting detection for center: ({center_y}, {center_x})")
        # Quick early checks
        if np.isnan(ow[center_y, center_x]):
            logging.warning(f"Center point is NaN: ({center_y}, {center_x})")
            return False, None, ow
            
        # Get local region for analysis
        y_min = max(0, int(center_y - radius))
        y_max = min(ow.shape[0], int(center_y + radius + 1))
        x_min = max(0, int(center_x - radius))
        x_max = min(ow.shape[1], int(center_x + radius + 1))
        local_ow = ow[y_min:y_max, x_min:x_max]
        #EddyMethods.plot_local_ow(ow, center_y, center_x, radius=radius, m_points=m_points, 
        #                          ssh=ssh, u_geo=u_geo, v_geo=v_geo)   
        logging.info(f"Local OW shape: {local_ow.shape}")
        
        # Check valid points in local region
        valid_ratio = np.sum(~np.isnan(local_ow)) / local_ow.size
        if valid_ratio < 0.8:
            logging.warning(f"Not enough valid points in local region: {valid_ratio}")
            return False, None, ow
            
        # Proceed with full analysis
        # (rest of the method remains unchanged)
        theta = np.linspace(0, 2*np.pi, m_points)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        circle_x = center_x - x_min + radius * cos_theta
        circle_y = center_y - y_min + radius * sin_theta
        
        points = np.column_stack((circle_y, circle_x))
        valid_points = (points[:, 0] >= 0) & (points[:, 0] < local_ow.shape[0]) & \
                      (points[:, 1] >= 0) & (points[:, 1] < local_ow.shape[1])
        points = points[valid_points]
        
        if len(points) < m_points * 0.7:
            logging.warning(f"Not enough valid points for full analysis: {len(points)}")
            return False, None, ow
            
        # Get OW values at the points directly
        ow_vals = local_ow[points[:, 0].astype(int), points[:, 1].astype(int)]
        valid_mask = ~np.isnan(ow_vals)
        
        if np.sum(valid_mask) < m_points * 0.7:
            logging.warning(f"Not enough valid OW values: {np.sum(valid_mask)}")
            return False, None, ow
            
        # Scale the radius based on OW value
        # 0.5 is minimum scale and 1.5 is maximum scale
        # This is done to avoid very small/large radii
        scale = np.clip(1.0 - ow_vals/np.max(np.abs(local_ow)), 0.5, 1.5)
        
        # Project the scaled points onto the global coordinate system
        global_x = x_min + points[valid_mask, 1] + (radius * (scale[valid_mask]-1) * cos_theta[valid_points][valid_mask])
        global_y = y_min + points[valid_mask, 0] + (radius * (scale[valid_mask]-1) * sin_theta[valid_points][valid_mask])
        
        # Prepare the contour points for masking
        contour_points = np.column_stack((global_y, global_x))
        if len(contour_points) < 3:
            logging.warning(f"Not enough contour points: {len(contour_points)}")
            return False, None, ow
        
        # Create smaller mask first as quick check
        small_box_size = 5
        y_min_test = max(0, int(center_y - small_box_size))
        y_max_test = min(ow.shape[0], int(center_y + small_box_size + 1))
        x_min_test = max(0, int(center_x - small_box_size))
        x_max_test = min(ow.shape[1], int(center_x + small_box_size + 1))
        
        y_coords_test, x_coords_test = np.mgrid[y_min_test:y_max_test, x_min_test:x_max_test]
        points_test = np.column_stack((y_coords_test.ravel(), x_coords_test.ravel()))

        # Quick test on small region
        local_mask_test = EddyMethods.is_point_inside(contour_points, points_test)
        if not np.any(local_mask_test):
            logging.warning(f"No points in mask for small region")
            return False, None, ow

        # Check SSH values at the contour points
        ssh_values = ssh[points_test[:, 0].astype(int), points_test[:, 1].astype(int)]
        center_ssh_value = np.nanmean(ssh_values)
        valid_ssh_mask = (ssh_values >= center_ssh_value - 0.5) & (ssh_values <= center_ssh_value + 0.5)
        
        if np.sum(valid_ssh_mask) < len(valid_ssh_mask) * 0.9:
            logging.warning(f"Not enough valid SSH values at contour points: {np.sum(valid_ssh_mask)}")
            return False, None, ow
            
        # If we passed all quick checks, do full mask
        y_min_cont = max(0, int(np.floor(np.min(contour_points[:, 0]))))
        y_max_cont = min(ow.shape[0], int(np.ceil(np.max(contour_points[:, 0]))) + 1)
        x_min_cont = max(0, int(np.floor(np.min(contour_points[:, 1]))))
        x_max_cont = min(ow.shape[1], int(np.ceil(np.max(contour_points[:, 1]))) + 1)
        
        y_coords, x_coords = np.mgrid[y_min_cont:y_max_cont, x_min_cont:x_max_cont]
        points = np.column_stack((y_coords.ravel(), x_coords.ravel()))
        
        local_mask = EddyMethods.is_point_inside(contour_points, points)
        
        mask = np.zeros_like(ow, dtype=bool)
        mask[y_min_cont:y_max_cont, x_min_cont:x_max_cont] = local_mask.reshape(y_max_cont-y_min_cont, x_max_cont-x_min_cont)
        
        # Get the grid coordinates of points inside the eddy
        y_coords, x_coords = np.where(mask)
        eddy_points = np.column_stack((y_coords, x_coords))
        
        masked_ow = ow.copy()
        masked_ow[mask] = np.nan
        
        eddy_info = {
            'center': np.array([center_y, center_x]),
            'border': contour_points,
            'points': eddy_points  # Array of (y,x) coordinates of points inside eddy
        }
        
        return True, eddy_info, masked_ow

    @staticmethod
    def plot_eddy_info(ssh, mag_uv, eddy_info, lat=None, lon=None):
        """
        Plot eddy information (center and boundary) on the full grid with SSH contours and velocity magnitude.

        :param ssh: Sea surface height field
        :param mag_uv: Magnitude of geostrophic velocity
        :param eddy_info: Dictionary containing eddy information (center, border, mask)
        :param lat: Optional latitude values for axis labels
        :param lon: Optional longitude values for axis labels
        """
        plt.figure(figsize=(10, 8))
        
        # Plot SSH contours
        if lat is not None and lon is not None:
            plt.contour(lon, lat, ssh, levels=60, colors='gray', alpha=0.5)
            x_coords = lon
            y_coords = lat
        else:
            plt.contour(ssh, levels=60, colors='gray', alpha=0.5)
            x_coords = np.arange(ssh.shape[1])
            y_coords = np.arange(ssh.shape[0])
        
        # Plot velocity magnitude
        plt.contourf(x_coords, y_coords, mag_uv, levels=30, cmap='hot')
        plt.colorbar(label='Geostrophic Velocity')
        
        if eddy_info is not None:
            # Plot eddy center
            center = eddy_info['center']
            if lat is not None and lon is not None:
                center_y = np.interp(center[0], np.arange(len(lat)), lat)
                center_x = np.interp(center[1], np.arange(len(lon)), lon)
            else:
                center_y, center_x = center
            plt.plot(center_x, center_y, 'w*', markersize=10, label='Eddy Center')
            
            # Plot eddy boundary
            border = eddy_info['border']
            if lat is not None and lon is not None:
                border_y = np.interp(border[:, 0], np.arange(len(lat)), lat)
                border_x = np.interp(border[:, 1], np.arange(len(lon)), lon)
            else:
                border_y, border_x = border[:, 0], border[:, 1]
            plt.plot(border_x, border_y, 'w-', linewidth=2, label='Eddy Boundary')
        
        plt.title('Eddy Detection Results')
        if lat is not None and lon is not None:
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
        else:
            plt.xlabel('Grid Points (X)')
            plt.ylabel('Grid Points (Y)')
        
        plt.legend()
        plt.show()
        return

    @staticmethod
    def is_point_inside(contour, points):
        """
        Check if points are inside a contour.

        :param contour: Contour points (n, 2)
        :param points: Points to check (m, 2)
        :return: Boolean array (m,) indicating if each point is inside the contour
        """
        contour_path = Path(contour)
        return contour_path.contains_points(points)

    @staticmethod
    def interpolate_grid(subset_df: Dict[str, xr.DataArray],
                        new_shape: Tuple[int, int]):
        """
        interpolate a 2D matrix using linear interpolation.

        :param subset_df: dictionary with data subsetted from .nc file
        :param new_shape: tuple specifying the desired shape of the interpolated matrix (new_n, new_m).
        :returns: 2D array of interpolated values.
                """

        new_df = {}
        exception_var = ['longitude', 'latitude']
        for key in subset_df.keys():
            if key not in exception_var:
                print(key)
                matrix = subset_df[key]
                # Calculate the zoom factors for each dimension
                zoom_factors = (new_shape[0] / matrix.shape[0], new_shape[1] / matrix.shape[1])
                new_df[key] = zoom(matrix, zoom_factors, order=1)

        lats = subset_df['ugos'].latitude.values  # Get latitude values
        lons = subset_df['ugos'].longitude.values  # Get longitude values
        n_lat = len(lats)
        n_lon = len(lons)

        # Desired number of interpolated values
        int_num_lat = new_df['ugos'].shape[0]
        int_num_lon = new_df['ugos'].shape[1]

        # Original indices
        original_id_lat = np.linspace(0, n_lat - 1, num=n_lat)
        original_id_lon = np.linspace(0, n_lon - 1, num=n_lon)

        # New indices for interpolation
        new_id_lat = np.linspace(0, n_lat - 1, num=int_num_lat)
        new_id_lon = np.linspace(0, n_lon - 1, num=int_num_lon)

        # Perform interpolation
        lat2 = np.interp(new_id_lat, original_id_lat, lats)
        lon2 = np.interp(new_id_lon, original_id_lon, lons)
        new_df['longitude'] = lon2
        new_df['latitude'] = lat2

        return new_df
