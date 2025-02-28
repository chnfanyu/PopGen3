import yaml
import logging

#logging.basicConfig(level=logging.INFO)


class ConfigError(Exception):
    pass


def wrap_config_value(value):
    """Wrap YAML elements as Config objects to allow attribute access.

    Example:
    If YAML defines:
        attribute1:
            attribute2: 'Value'

    Then, x.attribute1.attribute2 can be used to access "Value".
    """
    if isinstance(value, str):
        return value
    try:
        return value + 0
    except TypeError:
        pass

    return Config(value)


class Config:
    """Config class for handling YAML configuration as attribute-accessible objects."""

    DEFAULT_PARAMETERS = {
        "ipf": {
            "tolerance": 0.0001,
            "iterations": 250,
            "zero_marginal_correction": 0.00001,
            "rounding_procedure": "bucket",
            "archive_performance_frequency": 1
        },
        "reweighting": {
            "procedure": "ipu",
            "tolerance": 0.0001,
            "inner_iterations": 1,
            "outer_iterations": 50,
            "archive_performance_frequency": 1
        },
        "draws": {
            "pvalue_tolerance": 0.9999,
            "iterations": 25,
            "seed": 0
        }
    }

    def __init__(self, data):
        self._data = data or {}

    def __setitem__(self, key, value):
        self._data[key] = value

    def __setattr__(self, key, value):
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __getattr__(self, key):
        value = self.return_value(key)
        return wrap_config_value(value) if value is not None else None

    def __getitem__(self, key):
        value = self.return_value(key)
        return wrap_config_value(value)

    def __getstate__(self):
        """Ensure `yaml.dump()` works correctly with Config objects."""
        return self.__dict__

    def write_to_file(self, filepath):
        with open(filepath, 'w') as file:
            yaml.dump(self._data, file, default_flow_style=False)

    def return_value(self, key):
        """Retrieve value from config data safely."""
        try:
            return self._data[key]
        except KeyError:
            logging.warning(f"Key '{key}' not found in configuration.")
            return None  # Return None instead of raising an error

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return repr(self._data)

    def return_list(self):
        """Return a list of top-level keys in the configuration."""
        data_list = []
        for i in self._data:
            data_list.append(i)
        return data_list

    def return_dict(self):
        """Convert nested Config objects to pure dictionaries."""

        def convert(value):
            if isinstance(value, Config):
                return value.return_dict()
            elif isinstance(value, list):
                return [convert(v) for v in value]
            elif isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            return value

        return convert(self._data)

    def write_to_open(self, filepath):
        with open(filepath, 'w') as outfile:
            yaml.dump(self._data, outfile, default_flow_style=False)
