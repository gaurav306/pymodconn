import ruamel.yaml
import os
import json
from jsonschema import validate, ValidationError
import jsonschema
import pkg_resources

yaml = ruamel.yaml.YAML()
DEFAULT_CONFIG_FILENAME = "configs\default_config.yaml"

def read_write_yaml(filename, mode, data_yaml):
	# Read the yaml file
	if mode == 'r':
		with open(filename) as file:
			yaml_configs = yaml.load(file)
		return yaml_configs

	# Write to the yaml file
	if mode == 'w':
		with open(filename, 'w') as yamlfile:
			yaml.dump(data_yaml, yamlfile)


def get_configs(config_filename):
	
	# Check if the config file exists
	if not os.path.exists(config_filename) or config_filename == None:
		print(f"Config file '{config_filename}' not found. The default config file will be saved in current directory as '{config_filename}'. After editing the config file, please run the scrip again with .")
		# Load the default config file
		default_config_path = pkg_resources.resource_filename('pymodconn', DEFAULT_CONFIG_FILENAME)
		configs = read_write_yaml(default_config_path, 'r', None)
		# Write the default config to a user config file
		read_write_yaml(config_filename, 'w', configs)
		print(f"Default config file saved as '{config_filename}' in current directory.")
		print(f"Initiating system exit. \nPlease run the script again with '{config_filename}' in the current directory.")
		raise SystemExit(0)
	else:
		configs = read_write_yaml(config_filename, 'r', None)
	
	# Returning the configs
	return configs




