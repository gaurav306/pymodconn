import tensorflow as tf
from pymodconn.utils_layers import positional_encoding
from pymodconn.utils_layers import STATES_MANIPULATION_BLOCK
from pymodconn.TCN_addnorm_class_function import TCN_addnorm_class
from pymodconn.RNN_block_class_function import RNN_block_class
from pymodconn.MHA_block_class_function import MHA_block_class
from pymodconn.cit_block import CIT_block

K = tf.keras.backend


class Decoder_class():
	def __init__(self, cfg, enc_or_dec_number):
		self.cfg = cfg
		self.enc_or_dec_number = enc_or_dec_number

		self.n_past = self.cfg['n_past']
		self.n_future = self.cfg['n_future']
		self.known_past_features = self.cfg['known_past_features']
		self.unknown_future_features = self.cfg['unknown_future_features']
		self.known_future_features = self.cfg['known_future_features']

		self.all_layers_neurons = self.cfg['all_layers_neurons']
		self.all_layers_dropout = self.cfg['all_layers_dropout']

		self.merge_states_units = int(self.all_layers_neurons/self.cfg['decoder']['RNN_block_output']['rnn_depth'])
		self.merge_states_units = 8 * int(self.merge_states_units/8)		

	def __call__(self, input, input_vk, encoder_states=None):
		# input_vk is the input of the known past data from encoder class
		
		# first Dense layer to make number of features equal to all_layers_neurons
		encoder_input = tf.keras.layers.Dense(
			self.all_layers_neurons)(input)

		# dropout layer with one fifth of all_layers_dropout
		encoder_input = tf.keras.layers.Dropout(
			self.all_layers_dropout/5)(encoder_input)
		output_cell = encoder_input

		# TCN layer with addnorm and GLU at input of decoder aka TCN_decoder_input 
		# Settings in config file under cfg['decoder']['TCN_' + self.location]
		input_cell = output_cell
		output_cell = TCN_addnorm_class(self.cfg, 
				  						'decoder', 
										'input',
										self.enc_or_dec_number)(input_cell)

		# RNN layer with addnorm and GLU at input of decoder
		input_cell = output_cell
		output_cell, decoder_input_states = RNN_block_class(self.cfg,
						      								'decoder',
															"input",
															self.enc_or_dec_number)(input_cell, init_states=encoder_states)

		# Contextual Information Transfer block (CIT_block) layer
		input_cell = output_cell
		cit_block = CIT_block(self.cfg, self.enc_or_dec_number)
		output_cell = cit_block(input_cell, input_vk)
		
		# TCN layer with addnorm and GLU at output of decoder aka TCN_decoder_output
		input_cell = output_cell
		output_cell = TCN_addnorm_class(self.cfg, 
				  						'decoder', 
										'output',
										self.enc_or_dec_number)(input_cell)
		# STATES_MANIPULATION_BLOCK layer to merge encoder_input RNN states and decoder_input RNN states
		input_cell = output_cell
		merged_states =  STATES_MANIPULATION_BLOCK(self.merge_states_units, 
					     						   self.cfg['decoder']['MERGE_STATES_METHOD'])(encoder_states, decoder_input_states)
		# RNN layer with addnorm and GLU at output of decoder
		output_cell, _  = RNN_block_class(self.cfg,
				      						'decoder',
											"output",
											self.enc_or_dec_number)(input_cell, init_states = merged_states)
		
		# decoder_future output
		output_cell = tf.keras.layers.TimeDistributed(
			tf.keras.layers.Dense(self.unknown_future_features))(output_cell)

		return output_cell

