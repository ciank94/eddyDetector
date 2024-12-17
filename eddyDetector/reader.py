import xarray as xr
import numpy as np
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
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"Initializing {self.__class__.__name__} with start date: {datetime_start} and end date: {datetime_end}"
                         f"and number of dates: {self.n_dates}")
        self.logger.info(f"Files with prefix: {self.file_prefix} and suffix: {self.file_suffix}")
        return

    def download_check(self, filepath):
        """check if files are already downloaded

        :param filepath: input files directory
        """
        self.download_list = []
        self.file_list = []
        for date in self.datetime:
            date_str = date.strftime('%Y%m%d')
            filename = os.path.join(filepath, f"{self.file_prefix}_{date_str}_{self.file_suffix}")
            if not os.path.exists(filename):
                self.download_list.append(date)

        if len(self.download_list) == 0:
            self.logger.info("No files to download")
        else:
            self.logger.info(f"Downloading {len(self.download_list)} file (s)")
            self.download_files(filepath)

        # After downloading, update file_list with new files
        for date in self.datetime:
            date_str = date.strftime('%Y%m%d')
            filename = os.path.join(filepath, f"{self.file_prefix}_{date_str}_{self.file_suffix}")
            if os.path.exists(filename):
                self.file_list.append(filename)
                self.logger.info(f"File appended to list: {filename}")
        self.logger.info(f"Total files available: {len(self.file_list)}")
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
    def __init__(self, fileExp, time_index):
        self.time_index = time_index
        file_list = fileExp.file_list
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"Initializing {self.__class__.__name__} with time index: {self.time_index}")
        self.file = file_list[time_index]
        self.read_ncfile() # read the netcdf file
        return

    def read_ncfile(self):
        self.logger.info(f"Reading NetCDF file: {self.file}")
        self.data = xr.open_dataset(self.file, engine='netcdf4')
        self.logger.info(f"Finished reading NetCDF file: {self.file}")
        return

    def subset_netcdf(self, lon_range, lat_range):
        # lon_lat_mask
        lon_mask = (self.data['lon_bnds']['longitude'] >= lon_range[0]) & \
                        (self.data['lon_bnds']['longitude'] <= lon_range[1])
        lat_mask = (self.data['lat_bnds']['latitude'] >= lat_range[0]) & \
                        (self.data['lat_bnds']['latitude'] <= lat_range[1])
        self.logger.info(f"Created longitude and latitude masks for lon_range: {lon_range} and lat_range: {lat_range}")

        # Initialize dataframe dictionary
        df = {}

        # Subset velocity components
        df['u'] = np.array(self.data['ugos'].isel(time=self.time_index).where(lon_mask & lat_mask, drop=True))
        df['v'] = np.array(self.data['vgos'].isel(time=self.time_index).where(lon_mask & lat_mask, drop=True))

        # Compute geostrophic velocity magnitude
        df['net_vel'] = np.sqrt(df['u'] ** 2 + df['v'] ** 2)

        # subset adt
        df['ssh'] = np.array(self.data['adt'].isel(time=self.time_index).where(lon_mask & lat_mask, drop=True))

        # subset longitude and latitude
        df['lon'] = np.array(self.data['lon_bnds']['longitude'][lon_mask].data)
        df['lat'] = np.array(self.data['lat_bnds']['latitude'][lat_mask].data)
  
        self.logger.info(f"Created dataframe with keys: {df.keys()}")
        self.data.close()
        self.logger.info(f"Closed NetCDF file: {self.file}")
        return df

    