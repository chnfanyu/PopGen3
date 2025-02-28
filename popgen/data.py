import os
import pandas as pd
import numpy as np
import logging
from .config import ConfigError


class DB:
    """Handles all PopGen input data files and maintains necessary mappings and datasets."""

    def __init__(self, config):
        self.config = config
        self.sample = {}
        self.geo_marginals = {}
        self.region_marginals = {}
        self.geo = {}
        self.geo_ids = None
        self.region_ids = None
        self.sample_geo_ids = None
        self._inputs_config = self.config.project.inputs
        self.location = os.path.abspath(self.config.project.location)

    def load_data(self):
        """Loads all required data while handling missing data gracefully."""
        logging.info("Loading database data...")

        # Load geo correspondence mapping
        geo_corr_mapping_config = getattr(self._inputs_config.location, "geo_corr_mapping", None)
        self.geo = self.get_data(geo_corr_mapping_config) if geo_corr_mapping_config else {}

        # Load sample data
        sample_config = getattr(self._inputs_config.location, "sample", None)
        self.sample = self.get_data(sample_config) if sample_config else {}

        # Load marginals
        marginals_config = getattr(self._inputs_config.location, "marginals", None)
        self.geo_marginals = self.get_data(getattr(marginals_config, "geo", None),
                                           header=[0, 1]) if marginals_config and hasattr(marginals_config,
                                                                                          "geo") else {}
        self.region_marginals = self.get_data(getattr(marginals_config, "region", None),
                                              header=[0, 1]) if marginals_config and hasattr(marginals_config,
                                                                                             "region") else {}

        try:
            self._enumerate_geo_ids()
        except Exception as e:
            logging.warning(f"_enumerate_geo_ids failed due to {e}. Defaulting to empty lists.")
            self.geo_ids = []
            self.region_ids = []

    def get_data(self, config, header=0):
        """Loads data while handling missing files gracefully."""
        config_dict = config.return_dict() if config else {}
        data_dict = {}

        for item, filename in config_dict.items():
            if filename is None:
                logging.warning(f"{item} has no filename specified. Skipping.")
                continue

            full_location = os.path.join(self.location, filename)
            if os.path.exists(full_location):
                try:
                    data_dict[item] = pd.read_csv(full_location, index_col=0, header=header)
                    if data_dict[item].index.name:
                        data_dict[item].loc[:, data_dict[item].index.name] = data_dict[item].index.values
                except Exception as e:
                    logging.warning(f"Failed to load {filename} due to {e}. Skipping {item}.")
            else:
                logging.warning(f"{filename} not found. Skipping {item}.")

        return data_dict

    def _enumerate_geo_ids(self):
        """Ensures proper initialization of geo and region IDs."""
        self.geo_ids_all = []
        self.region_ids_all = []

        try:
            if "geo_to_sample" in self.geo:
                self.geo_ids_all = self.geo["geo_to_sample"].index.tolist()

            if "region_to_geo" in self.geo:
                self.region_ids_all = np.unique(self.geo["region_to_geo"].index.values).tolist()
            elif "region_to_sample" in self.geo:
                self.region_ids_all = np.unique(self.geo["region_to_sample"].index.values).tolist()
        except Exception as e:
            logging.warning(f"Failed to enumerate geo IDs due to {e}. Defaulting to empty lists.")
            self.geo_ids_all = []
            self.region_ids_all = []

    def get_geo_ids_for_region(self, region_id):
        """Retrieves geo IDs corresponding to a given region ID."""
        geo_name = self._inputs_config.column_names.geo
        geo_list = self.geo["region_to_geo"].loc[region_id, geo_name]
        return [int(geo_list)] if isinstance(geo_list, (int, np.integer)) else list(geo_list)

    def enumerate_geo_ids_for_scenario(self, scenario_config):
        """Ensures geo enumeration is handled even when missing data is encountered."""
        try:
            self.region_ids = getattr(scenario_config.geos_to_synthesize.region, "ids", self.region_ids_all)
            self.geo_ids = []

            if "region_to_geo" in self.geo:
                for region_id in self.region_ids:
                    geo_list = self.get_geo_ids_for_region(region_id)
                    if geo_list:
                        self.geo_ids += geo_list
            else:
                print("⚠️ No region_to_geo mapping. Using only region data.")
                self.geo_ids = self.region_ids if self.region_ids else []
        except ConfigError as e:
            print(f"⚠️ KeyError: {e}. Defaulting to all geo and region IDs.")
            self.geo_ids = self.geo_ids_all if self.geo_ids_all else []
            self.region_ids = self.region_ids_all if self.region_ids_all else []

    def return_variables_cats(self, entity, variable_names):
        """Returns unique categories for each variable in an entity dataset."""
        return {var: self.return_variable_cats(entity, var) for var in variable_names}

    def return_variable_cats(self, entity, variable_name):
        """Returns unique values for a specific variable in an entity dataset."""
        return np.unique(self.sample[entity][variable_name].values).tolist()

    def check_data(self):
        """Runs data consistency checks."""
        self.check_sample_marginals_consistency()
        self.check_marginals()

    def check(self):
        """Placeholder for additional data consistency checks."""
        pass
