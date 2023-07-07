import tensorflow as tf
from pymodconn.utils_layers import positional_encoding
from pymodconn.TCN_addnorm_class_function import TCN_addnorm_class
from pymodconn.RNN_block_class_function import RNN_block_class
from pymodconn.MHA_block_class_function import MHA_block_class

K = tf.keras.backend


class Encoder_class():
	def __init__(self, cfg, enc_or_dec_number):
		self.cfg = cfg
		self.enc_or_dec_number = enc_or_dec_number

		self.n_past = cfg['n_past']
		self.n_future = cfg['n_future']
		self.known_past_features = cfg['known_past_features']
		self.unknown_future_features = cfg['unknown_future_features']
		self.known_future_features = cfg['known_future_features']

		self.all_layers_neurons = cfg['all_layers_neurons']
		self.all_layers_dropout = cfg['all_layers_dropout']

	def __call__(self, input, init_states=None):
		
		# first Dense layer to make number of features equal to all_layers_neurons
		encoder_input = tf.keras.layers.Dense(
			self.all_layers_neurons)(input)

		# dropout layer with one fifth of all_layers_dropout
		encoder_input = tf.keras.layers.Dropout(
			self.all_layers_dropout/5)(encoder_input)
		output_cell = encoder_input

		# TCN layer with addnorm and GLU at input of encoder aka TCN_encoder_input 
		# Settings in config file under cfg['encoder']['TCN_' + self.location]
		input_cell = output_cell
		output_cell = TCN_addnorm_class(self.cfg, 
				  						'encoder', 
										'input',
										self.enc_or_dec_number)(input_cell)

		# RNN layer with addnorm and GLU at input of encoder
		input_cell = output_cell
		output_cell, output_states = RNN_block_class(self.cfg,
					       							'encoder',
													'input',
													self.enc_or_dec_number)(input_cell, init_states=init_states)

		# MHA layer with addnorm and GLU at input of encoder
		input_cell = output_cell
		for i in range(self.cfg['encoder']['self_MHA_block']['MHA_depth']):
			output_cell = MHA_block_class(self.cfg,
									   'encoder',
										self.enc_or_dec_number, 
										'self',
										str(i+1))(input_cell, input_cell)
			
			input_cell = output_cell

		output = output_cell
		return output, output_states

