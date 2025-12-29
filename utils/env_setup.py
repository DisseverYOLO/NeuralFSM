import os
import yaml

def set_env_variables_from_yaml(yaml_file):
    if not os.path.isfile(yaml_file):
        return
    with open(yaml_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file) or {}
        _set_env_variables(config)

def _set_env_variables(config, prefix=''):
    for key, value in config.items():
        if isinstance(value, dict):
            _set_env_variables(value, prefix + key.upper() + '_')
        else:
            env_var = prefix + key.upper()
            os.environ[env_var] = str(value)
            # Do not print secret values.
            print(f"Set environment variable {env_var}")

if __name__ == "__main__":
    # Optional: load local config.yaml if present (config.yaml should be gitignored).
    set_env_variables_from_yaml('config.yaml')
