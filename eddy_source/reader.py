from turtle import st
import xarray as xr
import numpy as np
from typing import Dict, Union, List
import cdsapi
import zipfile
import os
import datetime
import logging


class SeaLevelDataReader:
    def __init__(self, datetime_start, datetime_end):
        start_date = datetime.datetime.strptime(datetime_start, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(datetime_end, "%Y-%m-%d")
        
        # Initialize lists for years, months, and days
        self.datetime = []
        current_datetime = start_date
        while current_datetime <= end_date:
            self.datetime.append(current_datetime)
            current_datetime += datetime.timedelta(days=1)
        self.n_dates = len(self.datetime)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Start date: {start_date}")
        self.logger.info(f"End date: {end_date}")
        self.logger.info(f"Number of dates: {self.n_dates}")
        return

    def check_files_exist(self, filepath):
        # Check if the file exists
        file_prefix = "dt_global_twosat_phy_l4"
        file_suffix = "vDT2024.nc"
        self.download_list = []
        for date in self.datetime:
            date_str = date.strftime('%Y%m%d')
            file_path = os.path.join(filepath, f"{file_prefix}_{date_str}_{file_suffix}")
            if not os.path.exists(file_path):
                self.logger.info(f"File not found: {file_path}, adding to download list")
                self.download_list.append(date)
            else:
                self.logger.info(f"File found: {file_path}")

    @staticmethod
    def download_sea_level_netcdf(filepath, datetime_start, datetime_end):
        """
        Download NetCDF data from a Climate Data Store.

        :param datetime_start: Start date in the format "YYYY-MM-DD"
        :param datetime_end: End date in the format "YYYY-MM-DD"
        :param filepath: The file path to save the downloaded data.
        :return: None
        """

        start_date = datetime.datetime.strptime(datetime_start, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(datetime_end, "%Y-%m-%d")

        # Create simple lists for API
        year = [str(start_date.year)]  # Just need the start year since it's the same
        
        
        # Get list of months
        month = []
        for m in range(start_date.month, end_date.month + 1):
            month.append(f"{m:02d}")  # Zero-pad to ensure '01' format
            
        # Always use full month of days since API will filter
        day = [f"{d:02d}" for d in range(0, 32)]

        logging.info(f"Downloading data for year (s): {year}, month (s): {month}, day (s): {day}")        
        
        # Download data using API   
        Reader.download_cds_data(filepath, year, month, day)
        
        return 

    @staticmethod
    def download_cds_data(filepath:str, year: Union[str, List[str]],
                      month: Union[str, List[str]],
                      day: Union[str, List[str]],
                      version: str = "vdt2024"):
        """
        function to download data from the climate data store

        :param filepath: The file path to save the downloaded data.
        :param year: list of years
        :param month: list of months
        :param day: list of days
        :param version: sea level data version
        :return:
        """

        dataset = "satellite-sea-level-global"
        zip_file = "sea_level_data.zip"
        request = {
            "variable": ["daily"],
            "year": year,
            "month": month,
            "day": day,
            "version": version
        }
        client = cdsapi.Client()
        client.retrieve(dataset, request, zip_file)
        logging.info(f"Data saved to: {zip_file}")

        # Extract the ZIP file
        extracted_files = []
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Extract all files to a specific directory (optional)
            zip_ref.extractall(filepath)
            extracted_files = zip_ref.namelist()  # List the files in the ZIP

        # Print the list of extracted files
        for file in extracted_files:
            logging.info(f"Extracted file: {file} from {zip_file}")

        # Delete the zip file
        logging.info(f"Deleting zip file: {zip_file}")
        os.remove(zip_file)
        return



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