import tensorflow as tf

from pymodconn.utils_layers import *

K = tf.keras.backend


class Encoder_class():
	def __init__(self, cfg, enc_or_dec_number):
		self.cfg = cfg
		self.enc_or_dec_number = enc_or_dec_number

		self.IF_RNN = cfg['IFRNN_input']
		self.IF_MHA = cfg['IFSELF_MHA']
		self.MHA_DEPTH = cfg['ENCODER_MHA_DEPTH']
		self.IF_MASK = 0
	
		self.future_data_col = cfg['future_data_col']
		self.n_past = cfg['n_past']
		self.n_features_input = cfg['n_features_input']
		self.all_layers_neurons = cfg['all_layers_neurons']
		self.all_layers_dropout = cfg['all_layers_dropout']
		self.mha_head = cfg['mha_head']
		self.n_future = cfg['n_future']
		self.n_features_output = cfg['n_features_output']
		self.MHA_RNN = cfg['MHA_RNN']

	def __call__(self, input, init_states=None):
		
		encoder_input = tf.keras.layers.Dense(
			self.all_layers_neurons)(input)

		encoder_input = tf.keras.layers.Dropout(
			self.all_layers_dropout/5)(encoder_input)
		output_cell = encoder_input

		if self.MHA_RNN == 0:
			input_cell = output_cell
			rnn_block = RNN_block_class(self.cfg['IFRNN_input'],
										self.cfg,
										'encoder_input',
										self.enc_or_dec_number)
			
			output_cell, output_states = rnn_block(input_cell, init_states=init_states)

		input_cell = output_cell
		for i in range(self.MHA_DEPTH):
			self_mha = MHA_block_class(self.IF_MHA,
								True,
								self.cfg,
								'encoder',
								self.enc_or_dec_number, 
								'self',
								str(i+1))
			
			output_cell = self_mha(input_cell, input_cell)
			input_cell = output_cell

		if self.MHA_RNN == 1:
			input_cell = output_cell
			rnn_block = RNN_block_class(self.cfg['IFRNN_input'],
										self.cfg,
										'encoder_input',
										self.enc_or_dec_number)
			
			output_cell, output_states = rnn_block(input_cell, init_states=init_states)

		output = output_cell
		return output, output_states


class Decoder_class():
	def __init__(self, cfg, enc_or_dec_number):
		self.cfg = cfg
		self.enc_or_dec_number = enc_or_dec_number

		self.MHA_DEPTH = cfg['DECODER_MHA_DEPTH']

		self.future_data_col = cfg['future_data_col']
		self.n_past = cfg['n_past']
		self.n_features_input = cfg['n_features_input']
		self.all_layers_neurons = cfg['all_layers_neurons']
		self.all_layers_dropout = cfg['all_layers_dropout']
		self.mha_head = cfg['mha_head']
		self.n_future = cfg['n_future']
		self.n_features_output = cfg['n_features_output']

		self.merge_states_units = int(
			self.all_layers_neurons/self.cfg['input_enc_rnn_depth'])
		self.merge_states_units = 8 * int(self.merge_states_units/8)		

	def __call__(self, input, input_vk, encoder_states=None):
		
		encoder_input = tf.keras.layers.Dense(
			self.all_layers_neurons)(input)

		encoder_input = tf.keras.layers.Dropout(
			self.all_layers_dropout/5)(encoder_input)
		output_cell = encoder_input

		input_cell = output_cell
		rnn_block_dec_in = RNN_block_class(self.cfg['IFRNN_input'],
									self.cfg,
									"decoder_input",
									self.enc_or_dec_number)
		
		output_cell, decoder_input_states = rnn_block_dec_in(input_cell, init_states=encoder_states)

		input_cell = output_cell

		for i in range(self.MHA_DEPTH):
			casual_mha = MHA_block_class(self.cfg['IFCASUAL_MHA'],
									False,
									self.cfg,
									'decoder',
									self.enc_or_dec_number, 
									'casual',
									str(i+1))        #<------------------here
			output_cell = casual_mha(input_cell, input_cell)
		
			# decoder multi head attention
			skip_connection = output_cell
			input_cell = output_cell
			cross_mha = MHA_block_class(self.cfg['IFCROSS_MHA'],
									True,
									self.cfg,
									'decoder',
									self.enc_or_dec_number, 
									'cross',
									str(i+1))        #<------------------here
			output_cell = cross_mha(input_cell, input_vk)
			input_cell = output_cell


		input_cell = output_cell
		rnn_block_dec_out = RNN_block_class(self.cfg['IFRNN_output'],
											self.cfg,
											"decoder_output",
											self.enc_or_dec_number)

		merged_states =  STATES_MANIPULATION_BLOCK(self.merge_states_units, self.cfg['MERGE_STATES_METHOD'])(encoder_states, decoder_input_states)

		output_cell, output_states = rnn_block_dec_out(input_cell, init_states = merged_states)
		
		if self.cfg['IF_NONE_GLUADDNORM_ADDNORM'] == 1:
			output_cell, _ = GLU_with_ADDNORM(            #---------------> fix this
								output_layer_size=self.all_layers_neurons,
								dropout_rate=self.all_layers_dropout,
								use_time_distributed=False,
								activation=None)(skip_connection, output_cell)
		elif self.cfg['IF_NONE_GLUADDNORM_ADDNORM'] == 2:
			output_cell = ADD_NORM()(skip_connection, output_cell)

		# decoder_future output
		output_cell = tf.keras.layers.TimeDistributed(
			tf.keras.layers.Dense(self.n_features_output))(output_cell)

		return output_cell



class MHA_block_class():
	def __init__(self, IF_MHA, IF_GRN, cfg, enc_or_dec, enc_or_dec_number, self_or_casual_or_crossMHA, mha_depth_index):
		self.cfg = cfg
		self.enc_or_dec = enc_or_dec
		self.enc_or_dec_number = enc_or_dec_number
		self.IF_MHA = IF_MHA
		self.IF_GRN_mha = IF_GRN
		self.mha_depth_index = mha_depth_index
		self.self_or_casual_or_crossMHA = self_or_casual_or_crossMHA
		
		self.mha_head = self.cfg['mha_head']
		self.n_features_input = self.cfg['n_features_input']
		self.all_layers_neurons = self.cfg['all_layers_neurons']	
		self.all_layers_dropout = self.cfg['all_layers_dropout']
		self.IF_GRN = self.cfg['IF_GRN']
		self.dec_attn_mask = self.cfg['dec_attn_mask']


	def __call__(self, input_q, input_kv):
		if self.IF_MHA:
			
			self.mha_layer_name = self.enc_or_dec + '_' + str(self.enc_or_dec_number) + '_'+ str(self.self_or_casual_or_crossMHA) + 'MHA-' + str(self.mha_depth_index)

			if self.self_or_casual_or_crossMHA == 'casual' and self.dec_attn_mask == 1:
				causal_mask = get_causal_attention_mask()(input_q)
			else:
				causal_mask = None
			
			encoder_mha = tf.keras.layers.MultiHeadAttention(
								num_heads = self.mha_head,
								key_dim = self.n_features_input,
								value_dim = self.n_features_input,
								name=self.mha_layer_name)	

			output_cell = encoder_mha(query=input_q,
										key=input_kv,
										value=input_kv,
										attention_mask=causal_mask,
										training=True)

			if self.cfg['IF_NONE_GLUADDNORM_ADDNORM'] == 1:
				output_cell, _ = GLU_with_ADDNORM(            #---------------> fix this
									output_layer_size=self.all_layers_neurons,
									dropout_rate=self.all_layers_dropout,
									use_time_distributed=False,
									activation=None)(input_q, output_cell)
			elif self.cfg['IF_NONE_GLUADDNORM_ADDNORM'] == 2:
				output_cell = ADD_NORM()(input_q, output_cell)
			
			if self.IF_GRN == True and self.IF_GRN_mha == True:
				output_cell, _ = GRN_layer(
								hidden_layer_size = self.all_layers_neurons,
								output_size = self.all_layers_neurons,
								dropout_rate = self.all_layers_dropout,
								use_time_distributed = True,
								activation_layer_type = 'elu')(output_cell)					
		else:
			
			output_cell = input_q
		
		return output_cell



class RNN_block_class():
	def __init__(self, IF_RNN, cfg, location, num):
		self.IF_RNN = IF_RNN
		self.cfg = cfg
		self.location = location
		self.num = num
		self.n_features_input = self.cfg['n_features_input']
		self.all_layers_neurons = self.cfg['all_layers_neurons']	
		self.all_layers_dropout = self.cfg['all_layers_dropout']
		self.IF_GRN = self.cfg['IF_GRN']
	
	def __call__(self, input_cell, init_states=None):
		if self.IF_RNN:
			rnn = rnn_unit(self.cfg,
						rnn_location=self.location,
						num=self.num)
			
			rnn_outputs1 = rnn(input_cell,
								init_states=init_states)
			
			output_cell = rnn_outputs1[0]
			
			rnn_outputs1_allstates = rnn_outputs1[1:]

			if self.cfg['IF_NONE_GLUADDNORM_ADDNORM'] == 0:
				output_cell = linear_layer(self.all_layers_neurons)(output_cell)
			elif self.cfg['IF_NONE_GLUADDNORM_ADDNORM'] == 1:
				output_cell, _ = GLU_with_ADDNORM(            #---------------> fix this
									output_layer_size=self.all_layers_neurons,
									dropout_rate=self.all_layers_dropout,
									use_time_distributed=False,
									activation=None)(input_cell, output_cell)
			elif self.cfg['IF_NONE_GLUADDNORM_ADDNORM'] == 2:
				output_cell = linear_layer(self.all_layers_neurons)(output_cell)
				output_cell = ADD_NORM()(input_cell, output_cell)

			output_states = rnn_outputs1_allstates
			
			if self.cfg['IF_GRN']:
				output_cell, _ = GRN_layer(
									hidden_layer_size=self.all_layers_neurons,
									output_size=self.all_layers_neurons,
									dropout_rate=self.all_layers_dropout,
									use_time_distributed=True,
									activation_layer_type='elu')(output_cell)
		else:
			output_cell = input_cell
			output_states = None
		
		return output_cell, output_states
