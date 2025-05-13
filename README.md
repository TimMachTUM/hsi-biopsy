# HSI-Biopsy

A Python-based toolkit for hyperspectral imaging (HSI) analysis of biopsy samples, designed to streamline data loading, preprocessing, visualization, and feature extraction for medical research.

## Overview

This project focuses on the analysis of hyperspectral imaging (HSI) data for biopsy samples, with the goal of improving diagnostic capabilities through advanced imaging techniques.

## Prerequisites

- Python 3.8 or higher
- pip
- A Unix-like shell (bash) or Windows PowerShell

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/TimMachTUM/hsi-biopsy.git
   cd hsi-biopsy
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   # or .\.venv\Scripts\activate   # Windows
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Copy the example environment file and set your data paths:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and provide the absolute paths to your HSI data directory and metadata CSV:

   ```dotenv
   HSI_DATA_DIR=/absolute/path/to/hsi/mat/files
   METADATA_CSV_PATH=/absolute/path/to/processed/biopsy_metadata.csv
   ```

## Data Preparation

- Place your raw HSI data (`.mat` files) in the directory pointed to by `HSI_DATA_DIR`.
- Generate the metadata CSV by running the Excel processing script:
  ```bash
  python scripts/process_excel.py
  ```

## Usage

```python
from src.dataset import HSIDataset

# Initialize dataset (reads paths from .env)
ds = HSIDataset()

# Number of HSI cubes in the dataset
print(len(ds))

# Load a single HSI cube by index
sample = ds[0]
print(sample['combined_id'])      # Unique ID in format "patient_number_fov"
print(sample['patient_id'])       # Original patient ID (e.g., "S1.2") 
print(sample['fov'])              # FOV number
print(sample['hsi_cube'].shape)   # HSI cube data
print(sample['metadata'])         # Associated metadata

# Get by combined ID (e.g., "1.2_3" for patient S1.2, FOV 3)
sample_by_id = ds.get_sample_by_combined_id("1.2_3")

# Get all FOVs for a specific patient
patient_samples = ds.get_samples_by_patient_id("S1.2")

# Get a specific patient and FOV
specific_sample = ds.get_sample_by_patient_and_fov("S1.2", "3")
```

For interactive analysis, see [notebooks/example_analysis.ipynb](notebooks/example_analysis.ipynb).

## Testing

Run unit tests with pytest:

```bash
pytest --maxfail=1 --disable-warnings -q
```

## Contributing

Contributions, bug reports, and feature requests are welcome via GitHub issues and pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
