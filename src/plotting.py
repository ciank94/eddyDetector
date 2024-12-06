import matplotlib.pyplot as plt
import numpy as np
import os

def plot_eddy_detection(ssh, geos_vel, eddy_borders):
    """
    Plot the SSH contours, geostrophic velocity, and eddy borders.

    Parameters
    ----------
    ssh : numpy.ndarray
        Sea surface height data
    geos_vel : numpy.ndarray
        Geostrophic velocity magnitude
    eddy_borders : numpy.ndarray
        Array containing eddy border coordinates with shape (n_eddies, points, 2)
    """
    plt.rcParams.update({'font.size': 12})
    
    # Ensure the results directory exists
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Plot SSH contours
    plt.contour(ssh, levels=60, cmap=plt.get_cmap('grey'))
    
    # Plot geostrophic velocity
    plt.contourf(geos_vel, levels=30, cmap=plt.get_cmap('hot'))
    
    #Plot eddy borders
    [plt.plot(eddy_borders[i, :, 0], eddy_borders[i, :, 1], c='r') 
    for i in range(0, eddy_borders.shape[0])]
    
    plt.colorbar(label='Geostrophic Velocity')
    plt.title('Eddy Detection Results')
    plt.xlabel('Longitude Index')
    plt.ylabel('Latitude Index')
    plt.savefig(os.path.join(results_dir, 'eddy_detection.png'), dpi=300)
    return
