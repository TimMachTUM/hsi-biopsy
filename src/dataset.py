import pandas as pd
import os
import re
from scipy.io import loadmat
import numpy as np
import typing
import h5py
import torch
from torch.utils.data import Dataset
from .constants import HSI_DATA_DIR, METADATA_CSV_PATH


class HSIDataset(Dataset):
    """
    A dataset class to load and manage HSI biopsy data.

    It reads hyperspectral cubes from .mat files located in a specified directory
    and associates them with metadata from a provided CSV file.
    Each sample is a single HSI cube, making it memory-efficient for training.
    """

    def __init__(
        self,
        hsi_data_dir: str = HSI_DATA_DIR,
        metadata_csv_path: str = METADATA_CSV_PATH,
    ):
        """
        Initializes the HSIDataset.

        Parameters
        ----------
        hsi_data_dir : str
            Path to the directory containing HSI .mat files.
        metadata_csv_path : str
            Path to the processed metadata CSV file.
        """
        self.hsi_data_dir = hsi_data_dir
        try:
            self.metadata_df = pd.read_csv(metadata_csv_path)
        except FileNotFoundError:
            print(f"Error: Metadata CSV file not found at {metadata_csv_path}")
            self.metadata_df = pd.DataFrame()  # Empty DataFrame
        except Exception as e:
            print(f"Error reading metadata CSV {metadata_csv_path}: {e}")
            self.metadata_df = pd.DataFrame()

        if not self.metadata_df.empty and "id" in self.metadata_df.columns:
            self.metadata_df["normalized_id"] = self.metadata_df["id"].apply(
                self._normalize_id_csv
            )
            self.metadata_df.set_index(
                "normalized_id", inplace=True, drop=False
            )  # Keep 'id' column too
        else:
            print(
                "Warning: Metadata DataFrame is empty or 'id' column is missing. No metadata will be loaded."
            )
            # Ensure 'normalized_id' column exists even if empty for consistent access
            if "normalized_id" not in self.metadata_df.columns:
                self.metadata_df["normalized_id"] = pd.Series(dtype="str")

        # Each sample is a single HSI cube now, rather than a patient with multiple FOVs
        self.cube_samples = self._find_cube_samples()
        self.cube_samples.sort(key=lambda x: x["combined_id"])

        # Dictionary for faster sample lookups by combined_id
        self.sample_map = {s["combined_id"]: i for i, s in enumerate(self.cube_samples)}

        # ensure proper Dataset behavior
        super().__init__()

    def _normalize_id_csv(self, sample_id: any) -> typing.Union[str, any]:
        """Normalizes sample IDs from the CSV for consistent matching."""
        if isinstance(sample_id, str):
            return sample_id.replace("S.", "S").replace(" ", "").strip()
        return sample_id

    def _normalize_id_filename(self, filename_id_part: str) -> str:
        """Normalizes sample IDs extracted from filenames."""
        return (
            filename_id_part.strip()
        )  # Basic strip, regex usually gives clean S<num>.<num>

    def _extract_patient_number(self, sample_id: str) -> str:
        """
        Extracts the patient number from a sample ID (e.g., 'S1.2' -> '1.2').
        """
        if sample_id.startswith("S"):
            return sample_id[1:]
        return sample_id

    def _find_cube_samples(self) -> list:
        """
        Scans the HSI data directory and treats each HSI cube as a separate sample.
        Each sample gets a unique combined_id of format "patient_number_fov_number".
        """
        cube_samples = []
        # Regex: HyperProbe1.1_Biopsy_ (S<digits>.<digits>) (_FOV(<digits>))? .mat
        pattern = re.compile(r"^HyperProbe1\.1_Biopsy_(S\d+\.\d+)(?:_FOV(\d+))?(?:_BIS)?\.mat$")

        if not os.path.isdir(self.hsi_data_dir):
            print(f"Error: HSI data directory not found at {self.hsi_data_dir}")
            return []

        for filename in os.listdir(self.hsi_data_dir):
            match = pattern.match(filename)
            if match:
                raw_sample_id_part = match.group(1)  # e.g., "S1.2"
                fov_number_str = match.group(2)  # e.g., "1" or None

                # Default FOV is "1" if not specified
                fov_key = fov_number_str if fov_number_str else "1"

                normalized_file_id = self._normalize_id_filename(raw_sample_id_part)
                patient_number = self._extract_patient_number(normalized_file_id)

                # Create a combined ID like "1.2_3" for patient S1.2 FOV 3
                combined_id = f"{patient_number}_{fov_key}"

                if (
                    not self.metadata_df.empty
                    and "normalized_id" in self.metadata_df.index.names
                    and normalized_file_id in self.metadata_df.index
                ):
                    file_path = os.path.join(self.hsi_data_dir, filename)

                    # Ensure metadata is a dict, even if multiple rows match (should not happen with set_index)
                    meta_entry = self.metadata_df.loc[normalized_file_id]
                    if isinstance(
                        meta_entry, pd.DataFrame
                    ):  # if somehow index is not unique
                        metadata = meta_entry.iloc[0].to_dict()
                        print(
                            f"Warning: Multiple metadata entries for {normalized_file_id}, using first."
                        )
                    else:
                        metadata = meta_entry.to_dict()

                    # Each sample is a single HSI cube
                    cube_samples.append(
                        {
                            "combined_id": combined_id,  # Unique ID for each cube (patient_fov)
                            "patient_id": normalized_file_id,  # Original patient ID (e.g., S1.2)
                            "fov": fov_key,  # FOV number
                            "file_path": file_path,  # Path to the .mat file
                            "metadata": metadata,  # Associated metadata
                        }
                    )
                else:
                    print(
                        f"Warning: Metadata not found for sample ID '{normalized_file_id}' from file '{filename}' (normalized from '{raw_sample_id_part}')"
                    )

        return cube_samples

    def __len__(self) -> int:
        """Returns the number of individual HSI cubes in the dataset."""
        return len(self.cube_samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single HSI cube sample by its index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        dict
            A dictionary containing:
            - combined_id: Unique identifier of format "patient_number_fov_number"
            - patient_id: The patient identifier (e.g., "S1.2")
            - fov: The field of view number
            - hsi_cube: The hyperspectral cube data
            - metadata: Associated metadata from the CSV
        """
        # Validate index
        if idx < 0 or idx >= len(self.cube_samples):
            raise IndexError("Sample index out of range.")

        sample_info = self.cube_samples[idx]

        # Lazy loading of the HSI cube data
        hsi_cube = self._load_hsi_cube(sample_info["file_path"])

        return {
            "combined_id": sample_info["combined_id"],
            "patient_id": sample_info["patient_id"],
            "fov": sample_info["fov"],
            "hsi_cube": hsi_cube,
            "metadata": sample_info["metadata"],
        }

    def _load_hsi_cube(self, file_path: str) -> np.ndarray:
        """
        Loads a single HSI cube from a .mat file.

        Parameters
        ----------
        file_path : str
            Path to the .mat file containing the HSI cube.

        Returns
        -------
        np.ndarray or None
            The HSI cube data as a numpy array, or None if loading fails.
        """
        try:
            with h5py.File(file_path, "r") as f:
                dset = f["Ref_hyper"]
                return dset[...]
        except Exception as e:
            print(f"Error loading HSI data from {file_path}: {e}")
            return None

    def get_sample_by_combined_id(self, combined_id: str) -> typing.Union[dict, None]:
        """
        Retrieves a sample by its combined ID (patient_number_fov).

        Parameters
        ----------
        combined_id : str
            The combined ID in the format "patient_number_fov_number" (e.g., "1.2_3").

        Returns
        -------
        dict or None
            The sample data or None if not found.
        """
        if combined_id in self.sample_map:
            return self.__getitem__(self.sample_map[combined_id])

        print(f"Sample with combined ID '{combined_id}' not found.")
        return None

    def get_samples_by_patient_id(self, patient_id: str) -> typing.List[dict]:
        """
        Retrieves all samples (FOVs) for a specific patient ID.

        Parameters
        ----------
        patient_id : str
            The patient ID (e.g., "S1.2").

        Returns
        -------
        list of dict
            A list of all samples for the given patient, empty if none found.
        """
        normalized_id = self._normalize_id_csv(patient_id)

        # Strip "S" prefix if present for matching with combined_id format
        if normalized_id.startswith("S"):
            patient_number = normalized_id[1:]
        else:
            patient_number = normalized_id

        samples = []
        for idx, sample in enumerate(self.cube_samples):
            if sample["combined_id"].startswith(f"{patient_number}_"):
                samples.append(self.__getitem__(idx))

        if not samples:
            print(
                f"No samples found for patient ID '{patient_id}' (normalized: '{normalized_id}')."
            )

        return samples

    def get_sample_by_patient_and_fov(
        self, patient_id: str, fov: str
    ) -> typing.Union[dict, None]:
        """
        Retrieves a specific sample by patient ID and FOV number.

        Parameters
        ----------
        patient_id : str
            The patient ID (e.g., "S1.2").
        fov : str
            The FOV number as a string.

        Returns
        -------
        dict or None
            The sample data or None if not found.
        """
        normalized_id = self._normalize_id_csv(patient_id)

        # Strip "S" prefix if present for matching with combined_id format
        if normalized_id.startswith("S"):
            patient_number = normalized_id[1:]
        else:
            patient_number = normalized_id

        combined_id = f"{patient_number}_{fov}"
        return self.get_sample_by_combined_id(combined_id)
