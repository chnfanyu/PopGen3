import yaml

class ConfigError(Exception):
    pass

def wrap_config_value(value):
    """Wrap YAML elements as Config objects for attribute access."""
    if isinstance(value, str):
        return value

    try:
        return value + 0
    except TypeError:
        pass

    return Config(value)

class Config(object):
    """Class to access YAML elements used to specify the PopGen project."""
    def __init__(self, data):
        self._data = data

    def __getattr__(self, key):
        value = self.return_value(key)
        return wrap_config_value(value)

    def __getitem__(self, key):
        value = self.return_value(key)
        return wrap_config_value(value)

    def write_to_file(self, filepath):
        with open(filepath, 'w') as file:
            yaml.dump(self._data, file, default_flow_style=False)

    def return_value(self, key):
        try:
            return self._data[key]
        except KeyError:
            print(f"Warning: Key '{key}' not found in configuration. Using default value.")
            return None

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return self._data.__repr__()

    def return_list(self):
        return list(self._data)

    def return_dict(self):
        return self._data

    def write_to_open(self, filepath):
        with open(filepath, 'w') as outfile:
            outfile.write(yaml.dump(self._data, default_flow_style=False))

if __name__ == "__main__":
    yaml_f = open("configuration_arizona.yaml", "r")
    config_dict = yaml.safe_load(yaml_f)
    config_obj = Config(config_dict)
    print(config_obj.project.name)
    print(config_obj["project"]["name"])
