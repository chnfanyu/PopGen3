import logging
import os
import time
import yaml
import sys
import pandas as pd



from .config import Config
from .data import DB
from .ipf import Run_IPF
from .reweighting import Run_Reweighting
from .draw import Draw_Population
from .output import Syn_Population



class Project:
    """Primary class to set up and run PopGen projects."""

    def __init__(self, config_loc):
        self.config_loc = config_loc
        self._config = None
        self.db = None

    def load_project(self):
        """Loads the project configuration and initializes necessary components."""
        try:
            self._load_config()
            self._populate_project_properties()
            self._load_data()
        except Exception as e:
            logging.error(f"Failed to load project: {e}", exc_info=True)
            raise

    def _load_config(self):
        """Loads configuration from the YAML file."""
        try:
            with open(self.config_loc, "r") as config_f:
                config_dict = yaml.safe_load(config_f)
            logging.info("Configuration loaded successfully.")
            self._config = Config(config_dict)
        except FileNotFoundError:
            logging.critical(f"Configuration file {self.config_loc} not found.")
            raise
        except yaml.YAMLError as e:
            logging.critical("Error parsing YAML configuration file.", exc_info=True)
            raise

    def _populate_project_properties(self):
        """Extracts basic project properties from configuration."""
        self.name = self._config.project.name
        self.location = os.path.abspath(self._config.project.location)
        logging.info(f"Project initialized: {self.name} at {self.location}")

    def _load_data(self):
        """Loads the project database."""
        self.db = DB(self._config)
        self.db.load_data()
        logging.info("Database loaded successfully.")

    def run_scenarios(self):
        """Runs all configured scenarios."""
        scenarios_config = self._config.project.scenario
        synthesize = getattr(self._config.project, 'synthesize', True)

        for scenario_config in scenarios_config:
            if "parameters" not in scenario_config._data or scenario_config._data["parameters"] is None:
                logging.warning("'parameters' missing in scenario. Using default values.")
                scenario_config._data["parameters"] = Config.DEFAULT_PARAMETERS

            logging.info(f"Running scenario: {scenario_config.description}")

            apply_cross_level = getattr(scenario_config, "apply_cross_level", True)
            if not apply_cross_level:
                self.process_geo_region_mappings(scenario_config)

            try:
                scenario_obj = Scenario(
                    self.location,
                    self._config.project.inputs.entities,
                    self._config.project.inputs.housing_entities,
                    self._config.project.inputs.person_entities,
                    self._config.project.inputs.column_names,
                    scenario_config,
                    self.db,
                    synthesize
                )
                scenario_obj.run_scenario()
            except Exception as e:
                logging.error(f"Error running scenario: {scenario_config.description}", exc_info=True)

    def process_geo_region_mappings(self, scenario_config):
        """
        Handles the transition from geo to region mappings when apply_cross_level is False.
        Ensures region_to_sample is created if missing, updates geo_to_sample accordingly,
        and ensures region_to_geo is properly structured.
        """
        region_to_sample_exists = "region_to_sample" in self.db.geo and not self.db.geo["region_to_sample"].empty
        geo_to_sample_exists = "geo_to_sample" in self.db.geo and not self.db.geo["geo_to_sample"].empty
        region_to_geo_exists = "region_to_geo" in self.db.geo and not self.db.geo["region_to_geo"].empty

        if region_to_sample_exists:
            self.db.geo["geo_to_sample"] = self.db.geo["region_to_sample"]
        elif geo_to_sample_exists and region_to_geo_exists:
            geo_to_sample = self.db.geo["geo_to_sample"].reset_index(drop=True)
            region_to_geo = self.db.geo["region_to_geo"].reset_index(drop=True)

            region_to_sample = region_to_geo.merge(geo_to_sample, on="geo", how="left")
            region_to_sample = region_to_sample[['region', 'sample_geo']].drop_duplicates()
            region_to_sample = region_to_sample.sort_values(by=['region', 'sample_geo']).reset_index(drop=True)

            self.db.geo["region_to_sample"] = region_to_sample
            self.db.geo["geo_to_sample"] = region_to_sample
        else:
            self.db.geo["geo_to_sample"] = pd.DataFrame(columns=["region", "sample_geo"])
            self.db.geo["region_to_sample"] = pd.DataFrame(columns=["region", "sample_geo"])

        self._config.project.inputs.column_names["geo"] = self._config.project.inputs.column_names["region"]

        region_col_name = self._config.project.inputs.column_names["region"]
        if "region_to_geo" not in self.db.geo or self.db.geo["region_to_geo"].empty:
            self.db.geo["region_to_geo"] = pd.DataFrame(columns=[region_col_name])

        region_to_geo = pd.DataFrame({region_col_name: self.db.region_ids_all})
        region_to_geo[region_col_name + "_copy"] = region_to_geo[region_col_name]
        region_to_geo["extra_column"] = region_to_geo[region_col_name]
        region_to_geo.set_index(region_col_name, inplace=True)
        region_to_geo.columns = [region_col_name, region_col_name]
        region_to_geo = region_to_geo.loc[:, ~region_to_geo.columns.duplicated()]
        self.db.geo['region_to_geo'] = region_to_geo

        self.db.geo_ids = self.db.region_ids
        self.db.geo_marginals = self.db.region_marginals
        scenario_config.control_variables.geo = scenario_config.control_variables.region



class Scenario:
    """Class to manage and execute a scenario."""

    def __init__(self, location, entities, housing_entities, person_entities,
                 column_names_config, scenario_config, db, synthesize):
        self.location = location
        self.entities = entities
        self.housing_entities = housing_entities
        self.person_entities = person_entities
        self.column_names_config = column_names_config
        self.scenario_config = scenario_config
        self.db = db
        self.synthesize = synthesize
        self.start_time = time.time()

    def run_scenario(self):
        """Executes the scenario."""
        try:
            self._get_geo_ids()
            self._run_ipf()
            self._run_weighting()

            if self.synthesize:
                self._draw_sample()
                self._report_results()
            else:
                self._output_weights_only()
        except Exception as e:
            logging.error(f"Scenario execution failed: {self.scenario_config.description}", exc_info=True)

    def _get_geo_ids(self):
        """Enumerates geographical IDs for the scenario."""
        self.db.enumerate_geo_ids_for_scenario(self.scenario_config)
        logging.info("Geographical IDs enumerated successfully.")

    def _run_ipf(self):
        """Runs iterative proportional fitting (IPF)."""
        self.run_ipf_obj = Run_IPF(
            self.entities, self.housing_entities,
            self.column_names_config, self.scenario_config, self.db
        )
        self.run_ipf_obj.run_ipf()
        logging.info(f"IPF completed in: {time.time() - self.start_time:.4f} seconds")

    def _run_weighting(self):
        """Runs the reweighting process."""
        self.run_reweighting_obj = Run_Reweighting(
            self.entities, self.column_names_config, self.scenario_config, self.db
        )
        self.run_reweighting_obj.create_ds()
        self.run_reweighting_obj.run_reweighting(
            self.run_ipf_obj.region_constraints,
            self.run_ipf_obj.geo_constraints if self.run_ipf_obj.geo_constraints is not None else None
        )
        logging.info(f"Reweighting completed in: {time.time() - self.start_time:.4f} seconds")

    def _draw_sample(self):
        """Draws the synthetic population sample."""
        self.draw_population_obj = Draw_Population(
            self.scenario_config, self.db.geo_ids,
            self.run_reweighting_obj.geo_row_idx,
            self.run_ipf_obj.geo_frequencies,
            self.run_ipf_obj.geo_constraints,
            self.run_reweighting_obj.geo_stacked,
            self.run_reweighting_obj.region_sample_weights
        )
        self.draw_population_obj.draw_population()
        logging.info(f"Drawing completed in: {time.time() - self.start_time:.4f} seconds")

    def _report_results(self):
        self.syn_pop_obj = Syn_Population(
            self.location, self.db, self.column_names_config,
            self.scenario_config, self.run_ipf_obj,
            self.run_reweighting_obj, self.draw_population_obj,
            self.entities, self.housing_entities, self.person_entities
        )

        self.syn_pop_obj.add_records()
        self.syn_pop_obj.prepare_data()
        logging.debug(f"Scenario config type: {type(self.scenario_config)}")

        self.syn_pop_obj.export_outputs()
        logging.info(f"Results generated in: {time.time() - self.start_time:.4f} seconds")

    def _output_weights_only(self):
        weights_output_path = os.path.join(self.location, 'weights_output.csv')
        self.run_reweighting_obj.region_sample_weights.to_csv(weights_output_path, float_format='%.10f', index=False)
        logging.info(f"Sample weights saved to {weights_output_path}")


def popgen_run(project_config):
    """Entry point for running PopGen project."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    start_time = time.time()
    p_obj = Project(project_config)
    p_obj.load_project()
    p_obj.run_scenarios()
    logging.info(f"Total execution time: {time.time() - start_time:.4f} seconds")


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)

#
# def main():
#     config_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'configuration_arizona.yaml')
#     os.chdir(os.path.join(os.path.dirname(__file__), '..', 'data'))
#     popgen_run(config_path)
#
# # configuration_arizona
# # configuration_queen_creek4
#
#
# if __name__ == "__main__":
#     main()
