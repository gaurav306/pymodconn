import ruamel.yaml
yaml = ruamel.yaml.YAML()

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
	configs = read_write_yaml(config_filename, 'r', None)
	assert_check_configs(configs)
	return configs
