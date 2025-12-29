import os
import yaml

def set_env_variables_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
        _set_env_variables(config)

def _set_env_variables(config, prefix=''):
    for key, value in config.items():
        if isinstance(value, dict):
            _set_env_variables(value, prefix + key.upper() + '_')
        else:
            env_var = prefix + key.upper()
            os.environ[env_var] = str(value)
            print(f'Set environment variable {env_var}={value}')

# 使用示例
set_env_variables_from_yaml('config.yaml')
