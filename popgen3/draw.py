import pandas as pd
import numpy as np
from scipy import stats

class DrawPopulation(object):
    def __init__(self, scenario_config, geo_ids, geo_row_idx, geo_frequencies, geo_constraints, geo_stacked, region_sample_weights):
        self.scenario_config = scenario_config
        self.geo_ids = geo_ids
        self.geo_row_idx = geo_row_idx
        self.geo_frequencies = geo_frequencies
        self.geo_constraints = geo_constraints
        self.geo_stacked = geo_stacked
        self.region_sample_weights = region_sample_weights

        self.iterations = self.scenario_config.parameters.draws.iterations
        self.seed = self.scenario_config.parameters.draws.seed
        self.pvalue_tolerance = self.scenario_config.parameters.draws.pvalue_tolerance
        self.geo_id_rows_syn_dict = {}
        self.performance_columns = ["p_value", "iterations", "chi_sq_stat"]
        self.draws_performance = pd.DataFrame(index=self.geo_ids, columns=self.performance_columns)

    def draw_population(self):
        np.random.seed(self.seed)
        for geo_id in self.geo_ids:
            geo_sample_weights = self.region_sample_weights.loc[:, geo_id]
            geo_cumulative_weights = self._return_cumulative_probability(geo_sample_weights)
            geo_id_frequencies = self.geo_frequencies.loc[geo_id, :]
            geo_id_constraints = self.geo_constraints.loc[geo_id, :]

            if geo_id_frequencies.sum() == 0:
                continue

            p_value_max = -1
            for iter in range(self.iterations):
                seed = self.seed + iter
                geo_id_rows_syn = self._pick_households(geo_id_frequencies, geo_cumulative_weights)
                stat, p_value = self._measure_match(geo_id_rows_syn, geo_id_constraints)
                if p_value > self.pvalue_tolerance:
                    self._update_draw_performance(geo_id, p_value, iter, stat)
                    break
                elif p_value > p_value_max:
                    p_value_max, geo_id_rows_syn_max, iter_max, stat_max = p_value, geo_id_rows_syn, iter, stat
                    self._update_draw_performance(geo_id, p_value_max, iter_max, stat_max)

            self.geo_id_rows_syn_dict[geo_id] = geo_id_rows_syn_max

    def _return_cumulative_probability(self, geo_sample_weights):
        geo_cumulative_weights = {}
        for column in self.geo_frequencies.columns.values:
            rows = self.geo_row_idx[column]
            weights = geo_sample_weights.take(rows)
            geo_cumulative_weights[column] = (weights / weights.sum()).cumsum()
        return geo_cumulative_weights

    def _pick_households(self, geo_id_frequencies, geo_cumulative_weights):
        last = 0
        rand_numbers = np.random.random(int(geo_id_frequencies.sum()))
        list_rows_syn_subpop = []
        for column in self.geo_frequencies.columns.values:
            rows = self.geo_row_idx[column]
            column_frequency = int(geo_id_frequencies[column])
            column_bins = np.searchsorted(geo_cumulative_weights[column], rand_numbers[last:last + column_frequency], side="right")
            last += column_frequency
            list_rows_syn_subpop.append(rows.take(column_bins))
        return np.sort(np.concatenate(list_rows_syn_subpop))

    def _measure_match(self, geo_id_rows_syn, geo_id_constraints):
        geo_id_synthetic = self.geo_stacked.take(geo_id_rows_syn).sum()
        geo_id_synthetic = pd.DataFrame(geo_id_synthetic, columns=["synthetic_count"])
        geo_id_synthetic = geo_id_synthetic.join(geo_id_constraints, how="inner")
        return stats.chisquare(geo_id_synthetic["synthetic_count"], geo_id_synthetic["constraint"])

    def _update_draw_performance(self, geo_id, p_value, iter, stat):
        self.draws_performance.loc[geo_id, self.performance_columns] = (p_value, iter, stat)
