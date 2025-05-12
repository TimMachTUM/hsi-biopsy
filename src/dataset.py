import pandas as pd
import os
import re
from scipy.io import loadmat
import numpy as np
import typing  # Added import for typing.Union


class HSIDataset:
    """
    A dataset class to load and manage HSI biopsy data.

    It reads hyperspectral cubes from .mat files located in a specified directory
    and associates them with metadata from a provided CSV file.
    Handles samples that may have multiple Fields of View (FOVs).
    """

    def __init__(self, hsi_data_dir: str, metadata_csv_path: str):
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

        self.samples = self._find_samples()

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

    def _find_samples(self) -> list:
        """
        Scans the HSI data directory, parses filenames, and links to metadata.
        Groups files by sample ID, accommodating multiple FOVs per sample.
        """
        samples_map = {}
        # Regex: HyperProbe1.1_Biopsy_ (S<digits>.<digits>) (_FOV(<digits>))? .mat
        pattern = re.compile(r"^HyperProbe1\.1_Biopsy_(S\d+\.\d+)(?:_FOV(\d+))?\.mat$")

        if not os.path.isdir(self.hsi_data_dir):
            print(f"Error: HSI data directory not found at {self.hsi_data_dir}")
            return []

        for filename in os.listdir(self.hsi_data_dir):
            match = pattern.match(filename)
            if match:
                raw_sample_id_part = match.group(1)  # e.g., "S1.2"
                fov_number_str = match.group(2)  # e.g., "1" or None

                normalized_file_id = self._normalize_id_filename(raw_sample_id_part)

                if (
                    not self.metadata_df.empty
                    and "normalized_id" in self.metadata_df.index.names
                    and normalized_file_id in self.metadata_df.index
                ):
                    file_path = os.path.join(self.hsi_data_dir, filename)
                    fov_key = (
                        fov_number_str if fov_number_str else "1"
                    )  # Default FOV key

                    if normalized_file_id not in samples_map:
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

                        samples_map[normalized_file_id] = {
                            "id": metadata.get(
                                "id", normalized_file_id
                            ),  # Use original CSV id if available
                            "files": [],
                            "metadata": metadata,
                        }

                    samples_map[normalized_file_id]["files"].append(
                        {"path": file_path, "fov": fov_key}
                    )
                else:
                    print(
                        f"Warning: Metadata not found for sample ID '{normalized_file_id}' from file '{filename}' (normalized from '{raw_sample_id_part}')"
                    )

        for sample_id_key in samples_map:
            # Sort files by FOV number for consistent order
            samples_map[sample_id_key]["files"].sort(
                key=lambda x: int(x["fov"]) if x["fov"].isdigit() else -1
            )

        return list(samples_map.values())

    def __len__(self) -> int:
        """Returns the number of unique samples found."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a sample by its index.

        Loads HSI data for all FOVs of the sample.
        Returns a dictionary containing sample ID, metadata, and HSI cubes.
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError("Sample index out of range.")

        sample_info = self.samples[idx]
        print(f"Loading sample {idx + 1}/{len(self.samples)}: {sample_info}")
        hsi_cubes = {}

        for file_info in sample_info["files"]:
            try:
                mat_data = loadmat(file_info["path"])
                # Attempt to find the HSI cube data within the .mat file
                cube_key = None
                potential_keys = ["cube", "hsi_data", "image", "data", "hsi"]
                for pk in potential_keys:
                    if (
                        pk in mat_data
                        and isinstance(mat_data[pk], np.ndarray)
                        and mat_data[pk].ndim >= 2
                    ):  # typically 3D
                        cube_key = pk
                        break

                if (
                    not cube_key
                ):  # Fallback: find the largest 3D array or first 3D array
                    for key, value in mat_data.items():
                        if (
                            not key.startswith("__")
                            and isinstance(value, np.ndarray)
                            and value.ndim >= 2
                        ):  # Min 2D
                            if (
                                cube_key is None
                                or (value.ndim > mat_data[cube_key].ndim)
                                or (
                                    value.ndim == mat_data[cube_key].ndim
                                    and np.prod(value.shape)
                                    > np.prod(mat_data[cube_key].shape)
                                )
                            ):
                                cube_key = key

                if cube_key:
                    hsi_cubes[file_info["fov"]] = mat_data[cube_key]
                else:
                    print(
                        f"Warning: Could not find HSI data cube in {file_info['path']}"
                    )
                    hsi_cubes[file_info["fov"]] = None
            except Exception as e:
                print(f"Error loading HSI data from {file_info['path']}: {e}")
                hsi_cubes[file_info["fov"]] = None

        return {
            "id": sample_info["id"],
            "hsi_cubes": hsi_cubes,
            "metadata": sample_info["metadata"],
        }

    def get_sample_by_id(self, sample_id_query: str) -> typing.Union[dict, None]:
        """
        Retrieves a sample by its original ID (as in CSV).
        """
        normalized_query_id = self._normalize_id_csv(sample_id_query)
        for i, sample_data in enumerate(self.samples):
            # 'id' in sample_data should be the original CSV id
            if self._normalize_id_csv(sample_data["id"]) == normalized_query_id:
                return self.__getitem__(i)
        print(
            f"Sample with ID '{sample_id_query}' (normalized: '{normalized_query_id}') not found."
        )
        return None
