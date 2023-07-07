import tensorflow as tf
from pymodconn.utils_layers import GLU_with_ADDNORM
from pymodconn.utils_layers import linear_layer
from pymodconn.utils_layers import ADD_NORM
from pymodconn.utils_layers import GRN_layer

K = tf.keras.backend


class RNN_block_class():
	def __init__(self, cfg, enc_or_dec, input_or_output, num):
		self.cfg = cfg
		self.enc_or_dec = enc_or_dec
		self.input_or_output = input_or_output
		assert self.input_or_output in ['input', 'output'], 'input_or_output must be one of input, output'
		self.num = num
		
		self.all_layers_neurons = cfg['all_layers_neurons']
		self.all_layers_dropout = cfg['all_layers_dropout']

		self.IF_RNN 					= self.cfg[self.enc_or_dec]['RNN_block_' + self.input_or_output]['IF_RNN']
		self.IF_NONE_GLUADDNORM_ADDNORM = self.cfg[self.enc_or_dec]['RNN_block_' + self.input_or_output]['IF_NONE_GLUADDNORM_ADDNORM_block']
		self.IF_GRN 					= self.cfg[self.enc_or_dec]['RNN_block_' + self.input_or_output]['IF_GRN_block']

	def __call__(self, input_cell, init_states=None):
		if self.IF_RNN:
			rnn = rnn_unit(self.cfg,
		  				self.enc_or_dec,
						input_or_output=self.input_or_output,
						num=self.num)
			
			rnn_outputs1 = rnn(input_cell,
								init_states=init_states)
			
			output_cell = rnn_outputs1[0]
			
			rnn_outputs1_allstates = rnn_outputs1[1:]

			if self.IF_NONE_GLUADDNORM_ADDNORM == 0:
				output_cell = linear_layer(self.all_layers_neurons)(output_cell)

			elif self.IF_NONE_GLUADDNORM_ADDNORM == 1:
				output_cell = GLU_with_ADDNORM(            
									output_layer_size=self.all_layers_neurons,
									dropout_rate=self.all_layers_dropout,
									use_time_distributed=False,
									activation=None)(input_cell, output_cell)
				
			elif self.IF_NONE_GLUADDNORM_ADDNORM == 2:
				output_cell = linear_layer(self.all_layers_neurons)(output_cell)
				output_cell = ADD_NORM()(input_cell, output_cell)

			output_states = rnn_outputs1_allstates
			
			if self.IF_GRN:
				output_cell = GRN_layer(
									hidden_layer_size=self.all_layers_neurons,
									output_size=self.all_layers_neurons,
									dropout_rate=self.all_layers_dropout,
									use_time_distributed=True,
									activation_layer_type='elu')(output_cell)
		else:
			output_cell = input_cell
			output_states = None
		
		return output_cell, output_states



class rnn_unit():
	def __init__(self, cfg, enc_or_dec, input_or_output, num): 
		
		self.cfg = cfg
		self.enc_or_dec = enc_or_dec
		self.input_or_output = input_or_output
		self.num = num
		
		self.all_layers_neurons = self.cfg['all_layers_neurons']
		self.all_layers_dropout = self.cfg['all_layers_dropout']

		self.rnn_depth 					= self.cfg[self.enc_or_dec]['RNN_block_' + self.input_or_output]['rnn_depth']
		self.IF_NONE_GLUADDNORM_ADDNORM = self.cfg[self.enc_or_dec]['RNN_block_' + self.input_or_output]['IF_NONE_GLUADDNORM_ADDNORM_deep']
		self.rnn_type 					= self.cfg[self.enc_or_dec]['RNN_block_' + self.input_or_output]['rnn_type']
		self.IF_birectionalRNN 			= self.cfg[self.enc_or_dec]['RNN_block_' + self.input_or_output]['IF_birectionalRNN']

		self.all_layers_neurons_rnn = int(self.all_layers_neurons/self.rnn_depth)
		self.all_layers_neurons_rnn = 8 * int(self.all_layers_neurons_rnn/8)
		assert self.all_layers_neurons_rnn > 0, 'all_layers_neurons_rnn should be > 0...change all_layers_neurons or rnn_depth in config file'


	def __call__(self, input_cell, init_states=None):
		if self.rnn_depth == 1:
			return self.single_rnn_layer(x_input = input_cell, 
										 init_states = init_states, 
										 mid_layer = False, 
										 layername_prefix = 'Only_')
		else:
			x = self.single_rnn_layer(x_input = input_cell, 
			     					  init_states = init_states,
									  mid_layer = True, 
									  layername_prefix = 'First_')
						
			if self.IF_NONE_GLUADDNORM_ADDNORM == 1:
				x = GLU_with_ADDNORM(            #---------------> fix this
									output_layer_size=self.all_layers_neurons,
									dropout_rate=self.all_layers_dropout,
									use_time_distributed=False,
									activation=None)(input_cell, x)
			
			elif self.IF_NONE_GLUADDNORM_ADDNORM == 2:
				input_cell = linear_layer(self.all_layers_neurons_rnn * 2)(input_cell)
				x = ADD_NORM()(input_cell, x)
			
			for i in range(0, self.rnn_depth-2):
				x = self.single_rnn_layer(
					x_input=x, init_states=init_states, mid_layer=True, layername_prefix='Mid_%s_' % (i+1))
				
				if self.IF_NONE_GLUADDNORM_ADDNORM == 1:
					x = GLU_with_ADDNORM(            #---------------> fix this
										output_layer_size=self.all_layers_neurons,
										dropout_rate=self.all_layers_dropout,
										use_time_distributed=False,
										activation=None)(input_cell, x)
				
				elif self.IF_NONE_GLUADDNORM_ADDNORM == 2:
					input_cell = linear_layer(self.all_layers_neurons_rnn * 2)(input_cell)
					x = ADD_NORM()(input_cell, x)


			return self.single_rnn_layer(x_input = x, 
										 init_states = init_states, 
										 mid_layer = False, 
										 layername_prefix = 'Last_')

	def single_rnn_layer(self, x_input, init_states, mid_layer=False, layername_prefix=None):
		if self.rnn_type == "LSTM":
			RNN_type = tf.keras.layers.LSTM
		elif self.rnn_type == "GRU":
			RNN_type = tf.keras.layers.GRU
		elif self.rnn_type == "RNN":
			RNN_type = tf.keras.layers.SimpleRNN

		self.rnn_location = self.enc_or_dec + '_' + self.input_or_output
		if self.rnn_location == "encoder_input":
			self.init_state = None
		elif self.rnn_location == "decoder_input" or self.rnn_location == "decoder_output":
			self.init_state = init_states

		if mid_layer:
			ret_seq = True
			ret_state = False
		else:
			ret_seq = True
			ret_state = True

		if self.IF_birectionalRNN:
			self.layername = layername_prefix + self.rnn_location + \
				'_' + str(self.num) + '_bi' + self.rnn_type
		else:
			self.layername = layername_prefix + self.rnn_location + \
				'_' + str(self.num) + '_' + self.rnn_type

		if self.IF_birectionalRNN:
			x_output = tf.keras.layers.Bidirectional(RNN_type(
				self.all_layers_neurons_rnn,
				dropout=self.all_layers_dropout,
				return_sequences=ret_seq,
				return_state=ret_state,
				name=self.layername))(x_input, initial_state=self.init_state)
		else:
			x_output = RNN_type(
				self.all_layers_neurons_rnn,
				dropout=self.all_layers_dropout,
				return_sequences=ret_seq,
				return_state=ret_state,
				name=self.layername)(x_input, initial_state=self.init_state)
		return x_output

