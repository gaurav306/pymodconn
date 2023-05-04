import tensorflow as tf
from pymodconn.keras_tcn import TCN
from pymodconn.utils_layers import GLU_with_ADDNORM, ADD_NORM

K = tf.keras.backend

class TCN_addnorm_class():
	def __init__(self, cfg, enc_or_dec, input_or_output, enc_or_dec_number):
		self.cfg = cfg
		self.enc_or_dec = enc_or_dec
		self.input_or_output = input_or_output
		self.enc_or_dec_number = enc_or_dec_number
		assert self.input_or_output in ['input', 'output'], 'input_or_output must be one of input, output'
		self.all_layers_neurons = self.cfg['all_layers_neurons']
		self.all_layers_dropout = self.cfg['all_layers_dropout']

		self.IF_TCN 					= self.cfg[self.enc_or_dec]['TCN_' + self.input_or_output]['IF_TCN']
		self.IF_NONE_GLUADDNORM_ADDNORM = self.cfg[self.enc_or_dec]['TCN_' + self.input_or_output]['IF_NONE_GLUADDNORM_ADDNORM_TCN']
		self.kernel_size 				= self.cfg[self.enc_or_dec]['TCN_' + self.input_or_output]['kernel_size']
		self.nb_stacks 					= self.cfg[self.enc_or_dec]['TCN_' + self.input_or_output]['nb_stacks']
		self.dilations 					= self.cfg[self.enc_or_dec]['TCN_' + self.input_or_output]['dilations']
		
	def __call__(self, input_cell):
		if self.IF_TCN:
			tcn_block = TCN(nb_filters = self.all_layers_neurons, 
		   					kernel_size = self.kernel_size, 
							nb_stacks = self.nb_stacks,
							dilations = self.dilations,
							return_sequences = True, 
							dropout_rate = 0.05,  # ----> similar to recurrent_dropout in LSTM
							use_layer_norm = True,
							name='TCN_' + '_' + self.enc_or_dec + '_' + self.input_or_output + '_' + str(self.enc_or_dec_number))
			
			output_cell = tcn_block(input_cell)

			if self.IF_NONE_GLUADDNORM_ADDNORM == 1:
				output_cell = GLU_with_ADDNORM(            
									output_layer_size=self.all_layers_neurons,
									dropout_rate=self.all_layers_dropout,
									use_time_distributed=False,
									activation=None)(input_cell, output_cell)
				
			elif self.IF_NONE_GLUADDNORM_ADDNORM == 2:
				output_cell = ADD_NORM()(input_cell, output_cell)

			return output_cell
		else:
			return input_cell
