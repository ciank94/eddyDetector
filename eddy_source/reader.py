from turtle import st
import xarray as xr
import numpy as np
from typing import Dict, Union, List
import cdsapi
import zipfile
import os
import datetime
import logging

# Configure logging format to include the class name
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

class FileExplorerSLD:
    def __init__(self, datetime_start, datetime_end):
        """Initialize the FileExplorerSLD class.

        :param datetime_start: Start date in the format 'YYYY-MM-DD'
        :type datetime_start: str
        :param datetime_end: End date in the format 'YYYY-MM-DD'
        :type datetime_end: str
        """
        self.file_prefix = "dt_global_twosat_phy_l4"
        self.file_suffix = "vDT2024.nc"
        start_date = datetime.datetime.strptime(datetime_start, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(datetime_end, "%Y-%m-%d")
        
        # Initialize lists for years, months, and days
        self.datetime = []
        current_datetime = start_date
        while current_datetime <= end_date:
            self.datetime.append(current_datetime)
            current_datetime += datetime.timedelta(days=1)
        self.n_dates = len(self.datetime)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing FileExplorerSLD with start date: {datetime_start} and end date: {datetime_end}"
                         f"and number of dates: {self.n_dates}")
        self.logger.info(f"Files with prefix: {self.file_prefix} and suffix: {self.file_suffix}")
        return

    def download_check(self, filepath):
        """check if files are already downloaded

        :param filepath: input files directory
        """
        self.download_list = []
        for date in self.datetime:
            date_str = date.strftime('%Y%m%d')
            file_path = os.path.join(filepath, f"{self.file_prefix}_{date_str}_{self.file_suffix}")
            if not os.path.exists(file_path):
                self.logger.info(f"File not found: {file_path}, adding to download list")
                self.download_list.append(date)
            else:
                self.logger.info(f"File found: {file_path}")

        self.logger.info(f"Number of files to download: {len(self.download_list)}") 
        if len(self.download_list) == 0:
            self.logger.info("No files to download, exiting")
            return
        self.download_files(filepath)
        return

    def download_files(self, filepath):
        """download files from cds api

        :param filepath: input files directory
        :return: 
        """
        for date in self.download_list:
            dataset = "satellite-sea-level-global"
            zip_file = "sea_level_data.zip"
            request = {
                "variable": ["daily"],
                "year": f"{date.year}",
                "month": f"{date.month:02d}",
                "day": f"{date.day:02d}",
                "version": "vdt2024"
            }
            client = cdsapi.Client()
            client.retrieve(dataset, request, zip_file)

            # Extract the ZIP file
            extracted_files = []
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Extract all files to a specific directory (optional)
                zip_ref.extractall(filepath)
                extracted_files = zip_ref.namelist()  # List the files in the ZIP

            # Print the list of extracted files
            for file in extracted_files:
                self.logger.info(f"Extracted file: {file} from {zip_file} to {filepath}")

            # Delete the zip file
            self.logger.info(f"Deleting zip file: {zip_file}")
            os.remove(zip_file)
        self.logger.info(f"Finished downloading {len(self.download_list)} files")
        return

class ReaderSLD:
    def __init__(self):
        pass

    @staticmethod
    def subset_netcdf(
        filepath: str,
        lon_range: tuple[float, float],
        lat_range: tuple[float, float],
        time_index: int = 0,
        variables: list[str] | None = None) -> Dict[str, xr.DataArray]:
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