import os
import numpy as np
import pandas as pd
import enum
from typing import List, Dict, Set, Optional, Any, ClassVar

# Constants for HSI data directory and metadata CSV path, loaded from environment
HSI_DATA_DIR = os.environ.get("HSI_DATA_DIR")
METADATA_CSV_PATH = os.environ.get("METADATA_CSV_PATH")
WAVELENGTHS = np.linspace(385, 1015, 127)


if not HSI_DATA_DIR or not METADATA_CSV_PATH:
    raise RuntimeError(
        "Environment variables HSI_DATA_DIR and METADATA_CSV_PATH must be set. "
        "You can set them in a .env file (gitignored) or export in your shell before running the code."
    )


# Normalize patient IDs for consistent usage (same as in dataset.py)
def _normalize_id(sample_id):
    if isinstance(sample_id, str):
        return sample_id.replace("S.", "S").replace(" ", "").strip()
    return sample_id


# Load metadata CSV file and create patient ID mappings
try:
    _metadata_df = pd.read_csv(METADATA_CSV_PATH)
    _metadata_df["normalized_id"] = _metadata_df["id"].apply(_normalize_id)
    _all_patient_ids = _metadata_df["normalized_id"].dropna().unique().tolist()

    # Group patient IDs by tumor type
    _patient_ids_by_type = {}
    for tumor_type in _metadata_df["type_of_tumor"].dropna().unique():
        filtered_ids = (
            _metadata_df[_metadata_df["type_of_tumor"] == tumor_type]["normalized_id"]
            .dropna()
            .tolist()
        )
        _patient_ids_by_type[tumor_type] = filtered_ids

    print(f"Loaded {len(_all_patient_ids)} patient IDs from metadata CSV")
except Exception as e:
    print(f"Warning: Could not load patient IDs: {e}")
    _all_patient_ids = []
    _patient_ids_by_type = {}


# Create an enum-like class with patient IDs as class attributes for autocomplete
class ALL_PATIENT_IDS:
    """
    A class containing all normalized patient IDs from the metadata CSV as attributes.
    This provides autocomplete support for patient IDs in IDEs.

    Usage:
        # Access specific patient IDs (with autocomplete)
        patient_id = ALL_PATIENT_IDS.S1_2  # Returns "S1.2"

        # Get all patient IDs as a list
        all_ids = ALL_PATIENT_IDS.get_all()

        # Get glioma patient IDs
        glioma_ids = ALL_PATIENT_IDS.get_by_type("Glioma")
    """

    # Class variables to store collections of IDs
    _all_ids: ClassVar[List[str]] = _all_patient_ids
    _by_type: ClassVar[Dict[str, List[str]]] = _patient_ids_by_type

    @classmethod
    def get_all(cls) -> List[str]:
        """Returns a list of all patient IDs."""
        return sorted(cls._all_ids)

    @classmethod
    def get_by_type(cls, tumor_type: str) -> List[str]:
        """Returns a list of patient IDs filtered by tumor type."""
        return sorted(cls._by_type.get(tumor_type, []))

    @classmethod
    def get_types(cls) -> List[str]:
        """Returns a list of all tumor types."""
        return sorted(list(cls._by_type.keys()))


# Dynamically add each patient ID as a class attribute for autocomplete
for patient_id in _all_patient_ids:
    if isinstance(patient_id, str):
        # Convert dots to underscores to create valid attribute names
        # For example: "S1.2" becomes ALL_PATIENT_IDS.S1_2
        attr_name = patient_id.replace(".", "_")
        setattr(ALL_PATIENT_IDS, attr_name, patient_id)
