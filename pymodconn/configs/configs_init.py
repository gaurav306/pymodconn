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


def assert_check_configs(configs):
	assert configs['optimizer'] == 'Adam' or configs['optimizer'] == 'SGD', 'Adam must be either Adam or SGD'
	assert configs['model_type_prob'] == 'prob' or configs[
		'model_type_prob'] == 'nonprob', 'model_type_prob must be either prob or nonprob'
	assert configs['loss_prob'] == 'nonparametric' or configs[
		'loss_prob'] == 'parametric', 'loss_prob must be either nonparametric or parametric'
	#assert configs['control_future_cells'] == 1 or configs['control_future_cells'] == 6, 'control_future_cells must be either 1 or 6'
	assert configs['all_layers_neurons'] % 8 == 0 and configs[
		'all_layers_neurons'] >= 8, 'all_layers_neurons must be divisible by 8 and greater than or equal to 8'
	assert configs['mha_head'] % 8 == 0 and configs[
		'mha_head'] >= 8, 'mha_head must be divisible by 8 and greater than or equal to 8'
	assert configs['rnn_type'] in ['LSTM', 'GRU', 'RNN'], 'rnn_type must be either LSTM, GRU or RNN'
	assert configs['input_enc_rnn_depth'] <= 5, 'max depth of RNN units is 5'


def get_configs(config_filename):
	
	# Check if the config file exists
	if not os.path.exists(config_filename):
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
	
	# Validate the config file
	schema_path = pkg_resources.resource_filename('pymodconn', 'configs/schema_validation.json')
	validate_config(configs, schema_path)
	assert_check_configs(configs)

	# Returning the configs
	return configs



def validate_config(config, schema_path):
	with open(schema_path, 'r') as f:
		schema = json.load(f)
	try:
		jsonschema.validate(config, schema)
		print("Configuration is valid")
	except jsonschema.exceptions.ValidationError as e:
		print(f"Config file {config} failed schema validation with errors:")
		print(e)
		raise ValueError("Config file failed schema validation.")


