# eddyDetector

A Python package for efficient eddy detection in oceanographic data using a hybrid approach based on:
- Okubo-Weiss (OW) parameter
- Geostrophic velocities (u, v components)
- Sea Surface Height (SSH)

## Installation

### 1. Clone the Repository
```bash
git clone git@github.com:ciank94/eddyDetector.git
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate
```

### 3. Install in development mode

```bash
 python -m pip install -e {path/to/eddyDetector}
```
### 4. Install dependencies

```bash
python -m pip install cdsapi numpy xarray scipy matplotlib netcdf4
```
## Directory Structure

Important folders and files in the project:

```bash
eddyDetector/
├── eddyDetector/
│   ├── __init__.py
│   ├── eddy.py
│   ├── interpolate.py
│   ├── plotting.py
│   └── reader.py
├── figures/
├── input_files/
├── output_files/
├── tests/
├── usage/
│   └── detect.py
├── LICENSE
├── pyproject.toml
└── README.md
```


## Project Structure

### Source Code (`eddyDetector/`)
- **`__init__.py`**: Module initialization
- **`eddy.py`**: Eddy detection algorithm
- **`interpolate.py`**: Data interpolation
- **`plotting.py`**: Visualization
- **`reader.py`**: Data acquisition and preprocessing

### Figures (`figures/`)
Storage for generated plots

### Input (`input_files/`)
input data storage

### Output (`output_files/`)
- eddy detection results
- eddy detection statistics

### Tests (`tests/`)
Files for testing the code.

## Usage (see for example `usage/detect.py`)

first, import all modules from the eddyDetector package:

```python
# ===========Section 0: imports=============
from eddyDetector import *
```

then, define parameters for data acquisition, preprocessing, and simulation:

```python
# ==========Section 1: Parameters============= (stuff I will change a lot, others in a ./input_files/yaml file)
datetime_start = "2017-01-01"
datetime_end = "2017-02-01"
input_filepath = './input_files'
output_filepath = './output_files'
lat_bounds = (-73, -50)
lon_bounds = (-70, -31)
time_index = 0 # index for time dimension (test)
scale_factor_interpolation = 5
```

preprocess data:

```python	
# ==========Section 2: Prepare data=============
fileExp = FileExplorerSLD(datetime_start, datetime_end) # initiate file explorer for sea-level data (SLD) input files
fileExp.download_check(input_filepath) # check if files are already downloaded
reader = ReaderSLD(fileExp, time_index) # initiate reader for sea-level data (SLD) input files at time index
ncfile_subset = reader.subset_netcdf(lon_bounds, lat_bounds) # subset data
interpolator = InterpolateSLD(ncfile_subset) # initiate interpolator with latitude and longitude meshgrid
data = interpolator.interpolate_grid(scale_factor_interpolation) # interpolate data
```

## License
MIT License - See LICENSE file for details