import os
import numpy as np

# Constants for HSI data directory and metadata CSV path, loaded from environment
HSI_DATA_DIR = os.environ.get("HSI_DATA_DIR")
METADATA_CSV_PATH = os.environ.get("METADATA_CSV_PATH")
WAVELENGTHS = np.linspace(385, 1015, 127)

if not HSI_DATA_DIR or not METADATA_CSV_PATH:
    raise RuntimeError(
        "Environment variables HSI_DATA_DIR and METADATA_CSV_PATH must be set. "
        "You can set them in a .env file (gitignored) or export in your shell before running the code."
    )
