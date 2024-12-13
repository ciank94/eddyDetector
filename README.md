# findEddy

A Python package for efficient eddy detection in oceanographic data using a hybrid approach based on:
- Okubo-Weiss (OW) parameter
- Geostrophic velocities (u, v components)
- Sea Surface Height (SSH)

## Installation

### 1. Clone the Repository
```bash
git clone git@github.com:ciank94/findEddy.git
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
 python -m pip install -e path/to/findEddy
```
### 4. Install dependencies

```bash
python -m pip install cdsapi numpy xarray scipy matplotlib netcdf4 plotly
```

## Project Structure

### Source Code (`src/`)
- **`__init__.py`**: Package initialization and public API definitions
- **`reader.py`**: Data acquisition and preprocessing
  - Download functionality for oceanographic data
  - NetCDF file subsetting capabilities
- **`eddy_methods.py`**: Core eddy detection algorithms
  - Okubo-Weiss parameter calculation
  - Eddy detection and filtering methods
- **`plotting.py`**: Visualization tools
  - Eddy detection result plotting
  - Geostrophic velocity visualization

### Tests (`tests/`)
- **`test.py`**: Integration tests covering the complete workflow
  - Data download
  - Processing
  - Visualization

### Output (`results/`)
- Storage for generated plots and analysis results
- Eddy detection visualizations

## Usage

```python
from src import (
   EddyMethods,
   Reader,
   Plotting,
   __version__
)

# Download and process data
year, month, day = Reader.download_lists(y_start=2017, y_end=2017, m_start=1, m_end=1)
Reader.download_cds_data(year=year, month=month, day=day)

# Detect and visualize eddies
# See test.py for complete workflow example
```

## License
MIT License - See LICENSE file for details