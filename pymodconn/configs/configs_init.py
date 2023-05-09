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


def assert_check_edit_configs(configs):
	'''
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

	all_attn1 = configs['IFSELF_enc_MHA'] + configs['IFATTENTION']
	all_attn2 = configs['IFSELF_dec_MHA'] + configs['IFATTENTION']
	all_attn3 = configs['IFCROSS_MHA'] + configs['IFATTENTION']

	assert all_attn1 == 1 and all_attn2 == 1 and all_attn3 == 1, 'IFSELF_MHA, IFCASUAL_MHA, IFCROSS_MHA and IFATTENTION must be 1, i.e, only one of them can be 1 at a time'
	'''
	if configs['IF_SIMPLE_MODEL']['IF'] == 1:

		for enc_dec in ['encoder', 'decoder']:
			for block in ['TCN_input', 'RNN_block_input', 'self_MHA_block', 'cross_MHA_block', 'TCN_output', 'RNN_block_output']:
				
				all_try = ['IF_NONE_GLUADDNORM_ADDNORM_block',
	       					'IF_NONE_GLUADDNORM_ADDNORM_deep',
						    'IF_NONE_GLUADDNORM_ADDNORM_TCN',
						    'IF_GRN_block',
						    'IF_RNN',
						    'IF_MHA',
						    'IF_TCN',
						    'rnn_depth',
						    'rnn_type',
						    'IF_birectionalRNN',
							'MHA_head',
							'MHA_depth',
							'kernel_size',
							'nb_stacks',
							'dilations']
				
				all_except = ['IF_ALL_NONE_GLUADDNORM_ADDNORM',
							  'IF_ALL_NONE_GLUADDNORM_ADDNORM',
							  'IF_ALL_NONE_GLUADDNORM_ADDNORM',
							  'IF_ALL_GRN',
							  'IF_ALL_RNN',
							  'IF_ALL_MHA',
							  'IF_ALL_TCN',
							  'ALL_RNN_DEPTH',
							  'ALL_RNN_TYPE',
							  'ALL_RNN_BIDIRECTIONAL',
							  'ALL_MHA_HEAD',
							  'ALL_MHA_DEPTH',
							  'ALL_KERNEL_SIZE',
							  'ALL_NB_STACKS',
							  'ALL_DILATIONS']
		  		
				for i in range(len(all_try)):
					try:
						x = configs[enc_dec][block][all_try[i]]
						if x == 1:
							configs[enc_dec][block][all_try[i]] = configs['IF_SIMPLE_MODEL'][all_except[i]]
							
					except:
						continue

				all_try1 = ['IF_SELF_CROSS_MHA',
							'SELF_CROSS_MHA_depth']
				
				all_except1 = ['IF_ALL_MHA',
							   'ALL_MHA_DEPTH']
				
				for i in range(len(all_try1)):
					try:
						x = configs[enc_dec][all_try1[i]]
						if x == 1:
							configs[enc_dec][all_try1[i]] = configs['IF_SIMPLE_MODEL'][all_except1[i]]
					except:
						continue
	return configs


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
	
	# Validate the config file
	schema_path = pkg_resources.resource_filename('pymodconn', 'configs/schema_validation.json')
	validate_config(configs, schema_path)
	configs = assert_check_edit_configs(configs)

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


