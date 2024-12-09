import numpy as np
import xarray as xr
from typing import Dict, Tuple
from scipy.ndimage import zoom


def global_detect(ow, mag_uv, u_geo, v_geo, ssh, global_minima):
    """
    Detect eddies using global minima in the Okubo-Weiss field.

    :param ow: Okubo-Weiss parameter for 2D field
    :param mag_uv: net velocity for 2D field
    :param u_geo: zonal component of geostrophic velocity (2D)
    :param v_geo: meridional component of geostrophic velocity (2D)
    :param ssh: sea surface height (2D)
    :param global_minima: list of global minima coordinates
    :return: eddy_centres, eddy_borders
    """
    
    eddy_centres = np.zeros([len(global_minima), 2])
    eddy_borders = np.zeros([len(global_minima), 20,  2])

    # Track number of valid eddies
    valid_eddy_count = 0

    for counter, (y, x) in enumerate(global_minima):
        uv_centre_y, uv_centre_x = y, x
        [isEddy, eddy_border] = screen_centre2(uv_centre_y, uv_centre_x, mag_uv, u_geo, v_geo, ssh)
        if isEddy:
            eddy_boundary = expand_borders(uv_centre_y, uv_centre_x, mag_uv, u_geo, v_geo, ssh, eddy_border)
            eddy_centres[counter, :] = [uv_centre_y, uv_centre_x]
            eddy_borders[counter, :, :] = eddy_boundary
            valid_eddy_count += 1

    print(f"valid eddy count: {valid_eddy_count}")
    return (eddy_centres[:valid_eddy_count, :], 
            eddy_borders[:valid_eddy_count, :, :])


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
    slice_y, slice_x = block_indices(ow, block_size)
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
            uv_centre_y, uv_centre_x = find_uv_centre(ow_i, ssh, slice_yi, slice_xi, subgrid_size, c_method="other")
            #uv_centre_y, uv_centre_x = find_uv_centre(ow_i, ow, slice_yi, slice_xi, subgrid_size)
            #uv_centre_y, uv_centre_x = find_uv_centre(ow_i, mag_uv, slice_yi, slice_xi, subgrid_size)

            # now screen this centre:ow
            [isEddy, eddy_border] = screen_centre(uv_centre_y, uv_centre_x, mag_uv, u_geo, v_geo, ssh)

            if isEddy:
                # expand borders of screened eddies
                eddy_boundary = expand_borders(uv_centre_y, uv_centre_x, mag_uv, u_geo, v_geo, ssh, eddy_border)

                eddy_centres.append([uv_centre_y, uv_centre_x])
                eddy_borders.append(eddy_boundary)

    # potential_centres = np.array(eddy_centres)
    # plt.pcolormesh(mag_uv)
    # plt.scatter(potential_centres[:,1], potential_centres[:, 0], c='r')
    return np.array(eddy_centres), np.array(eddy_borders)


def find_global_minima_with_masking(ow_field, max_eddies=500, mask_radius=10):
    """
    Iteratively find global minima in the Okubo-Weiss field, masking previously found points.
    
    Parameters:
    -----------
    ow_field : numpy.ndarray
        2D Okubo-Weiss field
    max_eddies : int, optional
        Maximum number of eddies to find
    mask_radius : int, optional
        Radius of masking around each found minimum
    
    Returns:
    --------
    global_minima : list
        List of (y, x) coordinates of global minima
    masked_ow : numpy.ndarray
        Okubo-Weiss field with found minima masked
    """
    import numpy as np
    from scipy.ndimage import distance_transform_cdt

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

def expand_borders(uv_centre_y, uv_centre_x, mag_uv, u_geo, v_geo, ssh, eddy_border):
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

    circle_iterations = 20
    Sv = 1.1
    rad_x = 4
    rad_y = 4
    max_rad = 15
    uv_border = np.zeros(circle_iterations)
    uv_dot = np.zeros(circle_iterations - 1)
    u_geo_border = np.copy(uv_border)
    v_geo_border = np.copy(uv_border)
    x_border = np.copy(uv_border)
    y_border = np.copy(uv_border)
    ssh_border = np.copy(uv_border)
    switch_radx = True
    break_while_loop = False
    # todo: start with a radius of 3 and break loop if it doesn't pass checks- (not an eddy)- from there it can b

    contour_p = eddy_border

    # counted at least at 3- I should count the eddies found with this method;
    x_expansion = True
    y_expansion = True
    break_x_expansion = False
    break_y_expansion = False

    # while (rad_x <= max_rad) & (rad_y <= max_rad):
    while x_expansion or y_expansion:
        if (rad_x >= max_rad) or (rad_y >= max_rad):
            break
        if x_expansion:
            rad_x += 1
            for circle_it, angle in enumerate(np.linspace(0, 2 * np.pi, circle_iterations)):
                x_border[circle_it] = uv_centre_x + (rad_x * np.cos(angle))
                y_border[circle_it] = uv_centre_y + (rad_y * np.sin(angle))
                if ((x_border[circle_it] <= 0) or (y_border[circle_it] <= 0) or (
                        int(x_border[circle_it]) >= mag_uv.shape[1]-1)
                        or (int(y_border[circle_it]) >= mag_uv.shape[0]-1)):
                    break_while_loop = True
                    break
                else:
                    uv_border[circle_it] = mag_uv[int(y_border[circle_it]), int(x_border[circle_it])]
                    u_geo_border[circle_it] = u_geo[int(y_border[circle_it]), int(x_border[circle_it])]
                    v_geo_border[circle_it] = v_geo[int(y_border[circle_it]), int(x_border[circle_it])]
                    ssh_border[circle_it] = ssh[int(y_border[circle_it]), int(x_border[circle_it])]
                if np.isnan(uv_border[circle_it]):
                    break_while_loop = True
                    break
            if not break_while_loop:
                ratio_v = uv_border[1::] / uv_border[0:-1]
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

                # ratio of velocity and ssh on border
                ratio_ssh = np.abs(ssh_border[1::]) / np.abs(ssh_border[0:-1])
                ratio_vel = uv_border[1::] / uv_border[0:-1]

                # criteria for eddy
                symmetry_check = sum(np.abs(cos_similarities_opposite)<0.5)     
                direction_check = sum(uv_dot < 0.99)
                velocity_check = sum((ratio_v < 1 / Sv) | (ratio_v > Sv))
                ssh_check = sum((ratio_ssh < 1 / Sv) | (ratio_ssh > Sv))
                velocity_cv = (np.std(uv_border)/np.mean(uv_border))*100
                ssh_cv = (np.std(np.abs(ssh_border)) / np.mean(np.abs(ssh_border)))*100

                # screening based on criteria:
                criteria_1 = velocity_check/circle_iterations < 0.7
                criteria_2 = direction_check/circle_iterations < 0.7
                criteria_3 = symmetry_check/circle_iterations < 0.7
                criteria_4 = ssh_check/circle_iterations < 0.7
                criteria_5 = velocity_cv < 50
                criteria_6 = ssh_cv < 5

                # Criteria 1: velocity ratio between consecutive points
                # Criteria 2: direction change between consecutive points
                # Criteria 3: symmetry check for opposite points
                # Criteria 4: ssh ratio between consecutive points
                # Criteria 5 and 6: coefficient of variation of velocity and ssh on the border

                if not all([criteria_1, criteria_2, criteria_3, criteria_4, criteria_5, criteria_6]):
                    print(f"symmetry check: {symmetry_check}")
                    print(f"direction check: {direction_check}")
                    print(f"velocity check: {velocity_check}")
                    print(f"ssh check: {ssh_check}")
                    print(f"velocity cv: {velocity_cv}")
                    print(f"ssh cv: {ssh_cv}")                        
                    break_x_expansion = True

                if not break_x_expansion:
                    contour_p = np.array([x_border, y_border]).T
                # [contour_p.append((i, j)) for i, j in zip(x_border, y_border)]
                else:
                    x_expansion = False

        if y_expansion:
            rad_y += 1
            for circle_it, angle in enumerate(np.linspace(0, 2 * np.pi, circle_iterations)):
                x_border[circle_it] = uv_centre_x + (rad_x * np.cos(angle))
                y_border[circle_it] = uv_centre_y + (rad_y * np.sin(angle))
                if ((x_border[circle_it] <= 0) or (y_border[circle_it] <= 0) or (
                        int(x_border[circle_it]) >= mag_uv.shape[1]-1)
                        or (int(y_border[circle_it]) >= mag_uv.shape[0]-1)):
                    break_while_loop = True
                    break
                else:
                    uv_border[circle_it] = mag_uv[int(y_border[circle_it]), int(x_border[circle_it])]
                    u_geo_border[circle_it] = u_geo[int(y_border[circle_it]), int(x_border[circle_it])]
                    v_geo_border[circle_it] = v_geo[int(y_border[circle_it]), int(x_border[circle_it])]
                    ssh_border[circle_it] = ssh[int(y_border[circle_it]), int(x_border[circle_it])]
                if np.isnan(uv_border[circle_it]):
                    break_while_loop = True
                    break
            if not break_while_loop:
                ratio_v = uv_border[1::] / uv_border[0:-1]
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

                 # ratio of velocity and ssh on border
                ratio_ssh = np.abs(ssh_border[1::]) / np.abs(ssh_border[0:-1])
                ratio_vel = uv_border[1::] / uv_border[0:-1]

                # criteria for eddy
                symmetry_check = sum(np.abs(cos_similarities_opposite)<0.5)     
                direction_check = sum(uv_dot < 0.99)
                velocity_check = sum((ratio_v < 1 / Sv) | (ratio_v > Sv))
                ssh_check = sum((ratio_ssh < 1 / Sv) | (ratio_ssh > Sv))
                velocity_cv = (np.std(uv_border)/np.mean(uv_border))*100
                ssh_cv = (np.std(np.abs(ssh_border)) / np.mean(np.abs(ssh_border)))*100

                # screening based on criteria:
                criteria_1 = velocity_check/circle_iterations < 0.7
                criteria_2 = direction_check/circle_iterations < 0.7
                criteria_3 = symmetry_check/circle_iterations < 0.7
                criteria_4 = ssh_check/circle_iterations < 0.7
                criteria_5 = velocity_cv < 50
                criteria_6 = ssh_cv < 5

                # Criteria 1: velocity ratio between consecutive points
                # Criteria 2: direction change between consecutive points
                # Criteria 3: symmetry check for opposite points
                # Criteria 4: ssh ratio between consecutive points
                # Criteria 5 and 6: coefficient of variation of velocity and ssh on the border

                if not all([criteria_1, criteria_2, criteria_3, criteria_4, criteria_5, criteria_6]):
                    print(f"symmetry check: {symmetry_check}")
                    print(f"direction check: {direction_check}")
                    print(f"velocity check: {velocity_check}")
                    print(f"ssh check: {ssh_check}")
                    print(f"velocity cv: {velocity_cv}")
                    print(f"ssh cv: {ssh_cv}")                        
                    break_y_expansion = True

                if not break_y_expansion:
                    contour_p = np.array([x_border, y_border]).T
                    # [contour_p.append((i, j)) for i, j in zip(x_border, y_border)]
                else:
                    y_expansion = False

        if break_while_loop:
            break

            #
            # fig, axs = plt.subplots()
            # xx = np.arange(0, mag_uv.shape[0], 1)
            # yy = np.arange(0, mag_uv.shape[1], 1)
            # [ux, yx] = np.meshgrid(yy, xx)
            # sk = 10
            # axs.contourf(mag_uv, levels=50)
            # axs.quiver(
            #     ux[::sk, ::sk],
            #     yx[::sk, ::sk],
            #     u_geo[::sk, ::sk],
            #     v_geo[::sk, ::sk],
            #     edgecolors="k",
            #     alpha=0.5,
            #     linewidths=0.01,
            # )
            # axs.plot(x_border, y_border)
            # plt.show()
            #
            # breakpoint()

    return np.array(contour_p)

def perimeter_check(uv_centre_y, uv_centre_x, mag_uv, u_geo, v_geo, ssh):
    """
    Check the perimeter of a potential eddy for various criteria.

    :param uv_centre_y: y-coordinate of eddy center
    :param uv_centre_x: x-coordinate of eddy center
    :param mag_uv: magnitude of velocity
    :param u_geo: zonal component of geostrophic velocity
    :param v_geo: meridional component of geostrophic velocity
    :param ssh: sea surface height
    :return: tuple of (break_while_loop, criteria values)
    """
    circle_iterations = 20
    Sv = 1.1
    rad_x = rad_y = 3
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

    # ratio of velocity and ssh on border
    ratio_ssh = np.abs(ssh_border[1::]) / np.abs(ssh_border[0:-1])
    ratio_vel = uv_border[1::] / uv_border[0:-1]

    # criteria for eddy
    #todo: focus on symmetry checks- ssh and velocity
    symmetry_check = sum(np.abs(cos_similarities_opposite)<0.5)     
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
    criteria_1 = velocity_check/circle_iterations < 0.7
    criteria_2 = direction_check/circle_iterations < 0.7
    criteria_3 = symmetry_check/circle_iterations < 0.7
    criteria_4 = ssh_check/circle_iterations < 0.7
    criteria_5 = velocity_cv < 50
    criteria_6 = ssh_cv < 5
    if not all([criteria_1, criteria_2, criteria_3, criteria_4, criteria_5, criteria_6]):
        print(f"symmetry check: {symmetry_check}")
        print(f"direction check: {direction_check}")
        print(f"velocity check: {velocity_check}")
        print(f"ssh check: {ssh_check}")
        print(f"velocity cv: {velocity_cv}")
        print(f"ssh cv: {ssh_cv}")                        
        return True, None, None

    return False, np.array(x_border), np.array(y_border)



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
    circle_iterations = 30
    Sv = 1.1
    rad_x = rad_y = 3
    max_rad = 15
    uv_border = np.zeros(circle_iterations)
    uv_dot = np.zeros(circle_iterations - 1)
    u_geo_border = np.copy(uv_border)
    v_geo_border = np.copy(uv_border)
    ssh_border = np.copy(uv_border)
    x_border = np.copy(uv_border)
    y_border = np.copy(uv_border)
    switch_radx = True
    break_while_loop = False
    # todo: start with a radius of 3 and break loop if it doesn't pass checks- (not an eddy)- from there it can be
    radius = 3
    contour_p = []
    for circle_it, angle in enumerate(np.linspace(0, 2 * np.pi, circle_iterations)):
        x_border[circle_it] = uv_centre_x + (rad_x * np.cos(angle))
        y_border[circle_it] = uv_centre_y + (rad_y * np.sin(angle))
        if ((x_border[circle_it] <= 0) or (y_border[circle_it] <= 0) or (
            int(x_border[circle_it]) >= mag_uv.shape[1]-1)
            or (int(y_border[circle_it]) >= mag_uv.shape[0]-1)):
            break_while_loop = True
            break
        else:
            uv_border[circle_it] = mag_uv[int(y_border[circle_it]), int(x_border[circle_it])]
            u_geo_border[circle_it] = u_geo[int(y_border[circle_it]), int(x_border[circle_it])]
            v_geo_border[circle_it] = v_geo[int(y_border[circle_it]), int(x_border[circle_it])]
            ssh_border[circle_it] = ssh[int(y_border[circle_it]), int(x_border[circle_it])]
        if np.isnan(uv_border[circle_it]):
            break_while_loop = True
            break
    if not break_while_loop:
        ratio_v = uv_border[1::] / uv_border[0:-1]
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

        # ratio of velocity and ssh on border
        ratio_ssh = np.abs(ssh_border[1::]) / np.abs(ssh_border[0:-1])
        ratio_vel = uv_border[1::] / uv_border[0:-1]

        # criteria for eddy
        symmetry_check = sum(np.abs(cos_similarities_opposite)<0.5)     
        direction_check = sum(uv_dot < 0.99)
        velocity_check = sum((ratio_v < 1 / Sv) | (ratio_v > Sv))
        ssh_check = sum((ratio_ssh < 1 / Sv) | (ratio_ssh > Sv))
        velocity_cv = (np.std(uv_border)/np.mean(uv_border))*100
        ssh_cv = (np.std(np.abs(ssh_border)) / np.mean(np.abs(ssh_border)))*100

        # screening based on criteria:
        criteria_1 = velocity_check/circle_iterations < 0.7
        criteria_2 = direction_check/circle_iterations < 0.7
        criteria_3 = symmetry_check/circle_iterations < 0.7
        criteria_4 = ssh_check/circle_iterations < 0.7
        criteria_5 = velocity_cv < 50
        criteria_6 = ssh_cv < 5

        # Criteria 1: velocity ratio between consecutive points
        # Criteria 2: direction change between consecutive points
        # Criteria 3: symmetry check for opposite points
        # Criteria 4: ssh ratio between consecutive points
        # Criteria 5 and 6: coefficient of variation of velocity and ssh on the border

        if not all([criteria_1, criteria_2, criteria_3, criteria_4, criteria_5, criteria_6]):
            print(f"symmetry check: {symmetry_check}")
            print(f"direction check: {direction_check}")
            print(f"velocity check: {velocity_check}")
            print(f"ssh check: {ssh_check}")
            print(f"velocity cv: {velocity_cv}")
            print(f"ssh cv: {ssh_cv}")                        
            break_while_loop = True
    
    if not break_while_loop:
        [contour_p.append((i, j)) for i, j in zip(x_border, y_border)]
        is_eddy = True

    return is_eddy, np.array(contour_p)

def screen_centre2(uv_centre_y, uv_centre_x, mag_uv, u_geo, v_geo, ssh):
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
    contour_p = []
    [break_while_loop, x_border, y_border] = perimeter_check(    
        uv_centre_y,
        uv_centre_x,
        mag_uv,
        u_geo,
        v_geo,
        ssh)

    
    if not break_while_loop:
        contour_p = [(x, y) for x, y in zip(x_border, y_border)]
        #[contour_p.append((i, j)) for i, j in zip(x_border, y_border)]
        is_eddy = True

    return is_eddy, np.array(contour_p)


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


def find_uv_centre(ow_i, mag_uv, slice_yi, slice_xi, subgrid_size, c_method):
    # first find the minimum ow value in the block
    ow_centre_y, ow_centre_x = np.unravel_index(np.nanargmin(ow_i), ow_i.shape)

    # if we allow these centres, look for it's index on a larger grid;
    ow_centre_y += slice_yi.start
    ow_centre_x += slice_xi.start

    # create a subgrid for selection of velocity minimum
    [y_slice, x_slice] = slice_subgrid(mag_uv, ow_centre_x, ow_centre_y, subgrid_size)

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


def slice_subgrid(grid, center_x, center_y, subgrid_size):
    half_size = subgrid_size // 2
    start_x = max(center_x - half_size, 0)
    end_x = min(center_x + half_size + 1, grid.shape[1])
    start_y = max(center_y - half_size, 0)
    end_y = min(center_y + half_size + 1, grid.shape[0])
    return slice(start_y, end_y, 1), slice(start_x, end_x, 1)


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
    # Isern-Fontanet et al. 203 filter:
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


