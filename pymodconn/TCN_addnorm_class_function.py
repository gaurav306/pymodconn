import tensorflow as tf
from tcn import TCN
from pymodconn.utils_layers import GLU_with_ADDNORM

K = tf.keras.backend

class TCN_addnorm_class():
	def __init__(self, cfg, enc_or_dec, location):
		self.cfg = cfg
		self.enc_or_dec = enc_or_dec
		self.location = location
		assert self.location in ['encoder_input', 'decoder_input', 'decoder_output'], 'location must be one of encoder_input, decoder_input, decoder_output'
		self.all_layers_neurons = self.cfg['all_layers_neurons']
		self.all_layers_dropout = self.cfg['all_layers_dropout']

		self.IF_TCN = self.cfg[self.enc_or_dec]['TCN_' + self.location]['IF_TCN_' + self.location]
		self.kernel_size = self.cfg[self.enc_or_dec]['TCN_' + self.location]['kernel_size_' + self.location]
		self.nb_stacks = self.cfg[self.enc_or_dec]['TCN_' + self.location]['nb_stacks_' + self.location]
		self.dilations = self.cfg[self.enc_or_dec]['TCN_' + self.location]['dilations_' + self.location]
		
	def __call__(self, input_cell):
		if self.IF_TCN:
			tcn_block = TCN(nb_filters = self.all_layers_neurons, 
		   					kernel_size = self.kernel_size, 
							nb_stacks = self.nb_stacks,
							dilations = self.dilations,
							return_sequences = True, 
							dropout_rate = 0.05,  # ----> similar to recurrent_dropout in LSTM
							use_layer_norm = True,
							name='TCN_' + self.location)
			
			output_cell = tcn_block(input_cell)
			output_cell = GLU_with_ADDNORM(            
								output_layer_size=self.all_layers_neurons,
								dropout_rate=self.all_layers_dropout,
								use_time_distributed=False,
								activation=None)(input_cell, output_cell)

			return output_cell
		else:
			return input_cell
