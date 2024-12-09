import numpy as np
import matplotlib.pyplot as plt
from src import (
    download_lists, 
    download_cds_data, 
    subset_netcdf, 
    interpolate_grid, 
    calculate_okubo_weiss, 
    eddy_filter, 
    find_global_minima_with_masking
)

def test_global_minima_visualization():
    """
    Test function to visualize global minima in Okubo-Weiss field
    """
    # Download and process data
    folder = "./"
    filename = folder + 'dt_global_twosat_phy_l4_20170101_vDT2021.nc'

    subset_df = subset_netcdf(filepath= filename,
             lon_range = (-70, -31),
             lat_range = (-73, -50),
            time_index= 0,
            variables = ['longitude', 'latitude', 'ugos', 'vgos', 'adt'])
    
    # Interpolate grid
    new_shape = (int(subset_df['ugos'].shape[0]*5), int(subset_df['ugos'].shape[1]*5))
    df = interpolate_grid(subset_df, new_shape)
    
    # Calculate Okubo-Weiss
    val_ow, vorticity = calculate_okubo_weiss(np.array(df['ugos']), np.array(df['vgos']))
    ow, cyc_mask, acyc_mask = eddy_filter(val_ow, vorticity)
    
    # Find global minima
    [global_minima, global_minima_mask] = find_global_minima_with_masking(ow, mask_radius=5, max_eddies=1500)
    print(len(global_minima))

    # Visualization
    plt.figure(figsize=(15, 10))

    # Plot Okubo-Weiss field
    vmin = ow.min()
    vmax = ow.min() + 0.1  # or some other appropriate upper bound
    plt.pcolormesh(ow, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Okubo-Weiss')

    # Plot ADT contours
    plt.contour(df['adt'], colors='white', alpha=0.5, levels=35)

    # Plot global minima points
    global_minima_y, global_minima_x = zip(*global_minima)
    plt.scatter(global_minima_x, global_minima_y, color='red', marker='x', s=100, label='Global Minima')

    plt.title('Global Minima in Okubo-Weiss Field')
    plt.xlabel('Longitude Index')
    plt.ylabel('Latitude Index')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: print out some information about the minima
    print(f"Number of global minima found: {len(global_minima)}")
    for i, (y, x) in enumerate(global_minima, 1):
        print(f"Minimum {i}: Location (y, x) = ({y}, {x}), OW value = {ow[y, x]}")

if __name__ == '__main__':
    test_global_minima_visualization()
