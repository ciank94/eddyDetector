import xarray as xr
import numpy as np
from typing import Dict, Union, List
import cdsapi
import zipfile
import os

def download_lists(y_start: int,
                   y_end: int,
                   m_start: int,
                   m_end: int,
                   d_start: int,
                   d_end: int):
    """
    initiate and store time indices for downloading cds data

    :param y_start:
    :param y_end:
    :param m_start:
    :param m_end:
    :param d_start:
    :param d_end:
    :return:
    """

    year = []
    month = []
    day = []
    [year.append(f"{i:02d}") for i in range(y_start, y_end + 1)]  # ensures two-digit string, leading zeros if needed
    [month.append(f"{i:02d}") for i in range(m_start, m_end + 1)]  # ensures two-digit string, leading zeros if needed
    [day.append(f"{i:02d}") for i in range(d_start, d_end + 1)]  # ensures two-digit string, leading zeros if needed
    return year, month, day



def unzip_files():
    """
    unzip dt_global_twosat_phy_l4 files and remove .zip file

    """
    # Extract the zip file
    # Find the zip file in the specified directory
    zip_files = [f for f in os.listdir("./") if f.endswith('.zip')]

    # Extract the zip file
    zip_path = os.path.join("./", zip_files[0])
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("./")

    # Delete the zip file
    os.remove(zip_path)
    return


def download_cds_data(year: Union[str, List[str]],
                      month: Union[str, List[str]],
                      day: Union[str, List[str]],
                      version: str = "vdt2024"):
    """
    function to download data from the climate data store

    :param year:
    :param month:
    :param day:
    :param version:
    :return:
    """

    dataset = "satellite-sea-level-global"
    request = {
        "variable": ["daily"],
        "year": year,
        "month": month,
        "day": day,
        "version": version
    }
    client = cdsapi.Client()
    client.retrieve(dataset, request).download()

    # rename the file:
    unzip_files()
    return





def subset_netcdf(
        filepath: str,
        lon_range: tuple[float, float],
        lat_range: tuple[float, float],
        time_index: int = 0,
        variables: list[str] = None
) -> Dict[str, xr.DataArray]:
    """
    Subset variables from a NetCDF file using xarray within specified geographical bounds.

    :param filepath: Path to the NetCDF file
    :type filepath: str
    :param lon_range: (minimum longitude, maximum longitude) for the subset
    :type lon_range: tuple[float, float]
    :param lat_range: (minimum latitude, maximum latitude) for the subset
    :type lat_range: tuple[float, float]
    :param time_index: Index for the time dimension (default: 0)
    :type time_index: int, optional
    :param variables: List of variables to subset. If None, defaults to ['ugos', 'vgos', 'adt']
    :type variables: list[str], optional
    :return: Dictionary containing the subsetted variables including:
             - Original velocity components (uu, vv)
             - Computed geostrophic velocity magnitude (geos_vel)
             - Absolute dynamic topography (adt)
    :rtype: Dict[str, xr.DataArray]
    :raises ValueError: If longitude or latitude ranges are invalid
    :raises FileNotFoundError: If the specified file does not exist
    """
    # Input validation
    if lon_range[0] >= lon_range[1] or lat_range[0] >= lat_range[1]:
        raise ValueError("Invalid longitude or latitude ranges")

    # Set default variables if none provided
    if variables is None:
        variables = ['longitude' ,'latitude', 'ugos', 'vgos', 'adt']

    try:
        # Open the dataset using context manager
        with xr.open_dataset(filepath, engine='netcdf4') as ds:
            # Extract longitude and latitude bounds
            lon_mask = (ds['lon_bnds']['longitude'] >= lon_range[0]) & \
                       (ds['lon_bnds']['longitude'] <= lon_range[1])
            lat_mask = (ds['lat_bnds']['latitude'] >= lat_range[0]) & \
                       (ds['lat_bnds']['latitude'] <= lat_range[1])

            # Initialize result dictionary
            result = {}

            # Subset velocity components
            if 'ugos' in variables and 'vgos' in variables:
                result['ugos'] = ds['ugos'].isel(time=time_index).where(lon_mask & lat_mask, drop=True)
                result['vgos'] = ds['vgos'].isel(time=time_index).where(lon_mask & lat_mask, drop=True)

                # Compute geostrophic velocity magnitude
                result['geos_vel'] = np.sqrt(result['ugos'] ** 2 + result['vgos'] ** 2)

            # Subset additional variables
            for var in variables:
                if var not in ['longitude', 'latitude', 'ugos', 'vgos'] and var in ds:
                    result[var] = ds[var].isel(time=time_index).where(lon_mask & lat_mask, drop=True)

            if 'longitude' in variables and 'latitude' in variables:
                result['longitude'] = ds['lon_bnds']['longitude'][lon_mask].data
                result['latitude'] = ds['lat_bnds']['latitude'][lat_mask].data
            return result

    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find NetCDF file: {filepath}")
    except Exception as e:
        raise Exception(f"Error processing NetCDF file: {str(e)}")