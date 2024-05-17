import os
import pandas as pd
import numpy as np
from config import ConfigError

class DB(object):
    """Handles all PopGen input data files stored as Pandas dataframes."""
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
        geo_corr_mapping_config = self._inputs_config.location.geo_corr_mapping
        self.geo = self.get_data(geo_corr_mapping_config)
        sample_config = self._inputs_config.location.sample
        self.sample = self.get_data(sample_config)
        geo_marginals_config = self._inputs_config.location.marginals.geo
        self.geo_marginals = self.get_data(geo_marginals_config, header=[0, 1])
        region_marginals_config = self._inputs_config.location.marginals.region
        self.region_marginals = self.get_data(region_marginals_config, header=[0, 1])
        self._enumerate_geo_ids()

    def get_data(self, config, header=0):
        config_dict = config.return_dict()
        data_dict = {}
        for item in config_dict:
            filename = config_dict[item]
            full_location = os.path.join(self.location, filename)
            data_dict[item] = pd.read_csv(full_location, index_col=0, header=header)
            data_dict[item].loc[:, data_dict[item].index.name] = data_dict[item].index.values
        return data_dict

    def _enumerate_geo_ids(self):
        geo_to_sample = self.geo["geo_to_sample"]
        self.geo_ids_all = geo_to_sample.index.tolist()
        region_to_geo = self.geo["region_to_geo"]
        self.region_ids_all = np.unique(region_to_geo.index.values).tolist()

    def get_geo_ids_for_region(self, region_id):
        geo_name = self._inputs_config.column_names.geo
        return self.geo["region_to_geo"].loc[region_id, geo_name].copy().tolist()

    def enumerate_geo_ids_for_scenario(self, scenario_config):
        try:
            self.region_ids = scenario_config.geos_to_synthesize.region.ids
            self.geo_ids = []
            for region_id in self.region_ids:
                self.geo_ids += self.get_geo_ids_for_region(region_id)
        except ConfigError as e:
            print("KeyError", e)
            self.geo_ids = self.geo_ids_all
            self.region_ids = self.region_ids_all

    def return_variables_cats(self, entity, variable_names):
        return {variable_name: self.return_variable_cats(entity, variable_name) for variable_name in variable_names}

    def return_variable_cats(self, entity, variable_name):
        return np.unique(self.sample[entity][variable_name].values).tolist()

    def check_data(self):
        self.check_sample_marginals_consistency()
        self.check_marginals()

    def check(self):
        pass
