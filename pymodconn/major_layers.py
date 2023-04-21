import tensorflow as tf

from pymodconn.utils_layers import *

K = tf.keras.backend


class Input_encoder_MHA_RNN():
	def __init__(self, cfg, location, num):
		self.cfg = cfg
		self.location = location
		self.num = num
		if self.location == "encoder_past":
			self.IF_FFN = cfg['IFFFN']
			self.IF_RNN = cfg['IFRNN_input']
			self.IF_MHA = cfg['IFSELF_MHA']
			self.IF_MASK = 0
		elif self.location == "encoder_future":
			self.IF_FFN = cfg['IFFFN']
			self.IF_RNN = cfg['IFRNN_output']
			self.IF_MHA = cfg['IFCASUAL_MHA']
			self.IF_MASK = cfg['dec_attn_mask']
			
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
		
		encoder_past_inputs1 = tf.keras.layers.Dense(
			self.all_layers_neurons)(input)

		encoder_past_inputs1 = tf.keras.layers.Dropout(
			self.all_layers_dropout/5)(encoder_past_inputs1)
		output_cell = encoder_past_inputs1

		if self.MHA_RNN == 0:
			input_cell = output_cell
			# encoder BiLSTM
			if self.IF_RNN:
				
				encoder1 = rnn_unit(self.cfg,
									rnn_location=self.location,
									num=self.num)
				encoder_outputs1 = encoder1(
					input_cell,
					init_states=init_states)
				output_cell = encoder_outputs1[0]
				encoder_outputs1_allstates = encoder_outputs1[1:]

				output_cell, _ = GLU_with_ADDNORM(
									IF_GLU=self.cfg['IF_GLU'],
									IF_ADDNORM=self.cfg['IF_ADDNORM'],
									hidden_layer_size=self.all_layers_neurons,
									dropout_rate=self.all_layers_dropout,
									use_time_distributed=False,
									activation=None)(input_cell, output_cell)

				output_states = encoder_outputs1_allstates
				
				if self.cfg['IF_GLU']:
					#GRN
					output_cell, _ = GRN_layer(
										hidden_layer_size=self.all_layers_neurons,
										output_size=self.all_layers_neurons,
										dropout_rate=self.all_layers_dropout,
										use_time_distributed=True,
										activation_layer='elu')(output_cell)
				
			else:
				output_cell = input_cell
				output_states = None

		input_cell = output_cell
		if self.IF_MHA:
			if self.IF_MASK:
				causal_mask = get_causal_attention_mask()(input_cell)

				encoder_mha = tf.keras.layers.MultiHeadAttention(
					num_heads=self.mha_head,
					key_dim=self.n_features_input,
					value_dim=self.n_features_input,
					name=self.location+'_'+str(self.num) + '_casualMHA')

				output_cell = encoder_mha(query=input_cell,
										  key=input_cell,
										  value=input_cell,
										  attention_mask=causal_mask,
										  training=True)
			else:
				encoder_mha = tf.keras.layers.MultiHeadAttention(
					num_heads=self.mha_head,
					key_dim=self.n_features_input,
					value_dim=self.n_features_input,
					name=self.location+'_'+str(self.num) + '_selfMHA')
				
				output_cell = encoder_mha(query=input_cell,
										  key=input_cell,
										  value=input_cell,
										  training=True)

			output_cell, _ = GLU_with_ADDNORM(
									IF_GLU=self.cfg['IF_GLU'],
									IF_ADDNORM=self.cfg['IF_ADDNORM'],
									hidden_layer_size=self.all_layers_neurons,
									dropout_rate=self.all_layers_dropout,
									use_time_distributed=False,
									activation=None)(input_cell, output_cell)
			if self.cfg['IF_GLU']:
				#GRN
				output_cell, _ = GRN_layer(
									hidden_layer_size=self.all_layers_neurons,
									output_size=self.all_layers_neurons,
									dropout_rate=self.all_layers_dropout,
									use_time_distributed=True,
									activation_layer='elu')(output_cell)

		else:
			output_cell = input_cell

		input_cell = output_cell
		# encoder feed forward network
		if self.IF_FFN and self.IF_MHA:
			output_cell = FeedForward(
				self.all_layers_neurons, self.all_layers_dropout/2)(input_cell)
			
			output_cell, _ = GLU_with_ADDNORM(
							IF_GLU=self.cfg['IF_GLU'],
							IF_ADDNORM=self.cfg['IF_ADDNORM'],
							hidden_layer_size=self.all_layers_neurons,
							dropout_rate=self.all_layers_dropout,
							use_time_distributed=False,
							activation=None)(input_cell, output_cell)
		else:
			output_cell = input_cell

		if self.MHA_RNN == 1:
			input_cell = output_cell
			# encoder BiLSTM
			if self.IF_RNN:
				encoder1 = rnn_unit(self.cfg,
									rnn_location=self.location,
									num=self.num)
				encoder_outputs1 = encoder1(
					input_cell,
					init_states=init_states)
				output_cell = encoder_outputs1[0]
				encoder_outputs1_allstates = encoder_outputs1[1:]

				output_cell, _ = GLU_with_ADDNORM(
							IF_GLU=self.cfg['IF_GLU'],
							IF_ADDNORM=self.cfg['IF_ADDNORM'],
							hidden_layer_size=self.all_layers_neurons,
							dropout_rate=self.all_layers_dropout,
							use_time_distributed=False,
							activation=None)(input_cell, output_cell)

				output_states = encoder_outputs1_allstates
			else:
				output_cell = input_cell
				output_states = None

		output = output_cell
		return output, output_states


class Output_decoder_crossMHA_RNN():
	def __init__(self, cfg, location, num):
		self.cfg = cfg
		self.location = location
		self.num = num
		if self.location == "encoder_past":
			self.IF_FFN = cfg['IFFFN']
			self.IF_RNN = cfg['IFRNN_input']
			self.IF_MHA = cfg['IFSELF_MHA']
			self.IF_MASK = 0
		elif self.location == "encoder_future":
			self.IF_FFN = cfg['IFFFN']
			self.IF_RNN = cfg['IFRNN_output']
			self.IF_MHA = cfg['IFCASUAL_MHA']
			self.IF_MASK = cfg['dec_attn_mask']
		elif self.location == "decoder_future":
			self.IF_FFN = cfg['IFFFN']
			self.IF_RNN = cfg['IFRNN_output']
			self.IF_MHA = cfg['IFCROSS_MHA']
			self.IF_MASK = 0

		self.future_data_col = cfg['future_data_col']
		self.n_past = cfg['n_past']
		self.n_features_input = cfg['n_features_input']
		self.all_layers_neurons = cfg['all_layers_neurons']
		self.all_layers_dropout = cfg['all_layers_dropout']
		self.mha_head = cfg['mha_head']
		self.n_future = cfg['n_future']
		self.n_features_output = cfg['n_features_output']

	def __call__(self, input_qk, input_v, init_states=None):
		
		skip_connection = input_qk

		# decoder multi head attention
		if self.IF_MHA:
			decoder_mha = tf.keras.layers.MultiHeadAttention(
				num_heads=self.mha_head,
				key_dim=self.n_features_input,
				value_dim=self.n_features_input,
				name=self.location+'_'+str(self.num) + '_crossMHA')
			decoder_attn_output = decoder_mha(query=input_qk,
											  key=input_v,
											  value=input_v,
											  training=True)

			decoder_attn_output, _ = GLU_with_ADDNORM(
						IF_GLU=self.cfg['IF_GLU'],
						IF_ADDNORM=self.cfg['IF_ADDNORM'],
						hidden_layer_size=self.all_layers_neurons,
						dropout_rate=self.all_layers_dropout,
						use_time_distributed=False,
						activation=None)(input_qk, decoder_attn_output)

			if self.cfg['IF_GLU']:
				#GRN
				decoder_attn_output, _ = GRN_layer(
									hidden_layer_size=self.all_layers_neurons,
									output_size=self.all_layers_neurons,
									dropout_rate=self.all_layers_dropout,
									use_time_distributed=True,
									activation_layer='elu')(decoder_attn_output)

		else:
			decoder_attn_output = input_qk

		# decoder feed forward network
		if self.IF_FFN and self.IF_MHA:
			decoder_ffn_output = FeedForward(
				self.all_layers_neurons, self.all_layers_dropout/2)(decoder_attn_output)

			decoder_ffn_output, _ = GLU_with_ADDNORM(
						IF_GLU=self.cfg['IF_GLU'],
						IF_ADDNORM=self.cfg['IF_ADDNORM'],
						hidden_layer_size=self.all_layers_neurons,
						dropout_rate=self.all_layers_dropout,
						use_time_distributed=False,
						activation=None)(decoder_attn_output, decoder_ffn_output)			
		else:
			decoder_ffn_output = decoder_attn_output

		# decoder BiLSTM
		if self.IF_RNN:
			decoder1 = rnn_unit(
				self.cfg,
				rnn_location=self.location,
				num=self.num)
			decoder_outputs1 = decoder1(
				decoder_ffn_output,
				init_states=init_states)

			if self.cfg['IF_GLU']:
				#GRN
				decoder_outputs1, _ = GRN_layer(
									hidden_layer_size=self.all_layers_neurons,
									output_size=self.all_layers_neurons,
									dropout_rate=self.all_layers_dropout,
									use_time_distributed=True,
									activation_layer='elu')(decoder_outputs1[0])
			else:
				decoder_outputs1 = decoder_outputs1[0]		

			decoder_outputs1, _ = GLU_with_ADDNORM(
						IF_GLU=self.cfg['IF_GLU'],
						IF_ADDNORM=self.cfg['IF_ADDNORM'],
						hidden_layer_size=self.all_layers_neurons,
						dropout_rate=self.all_layers_dropout,
						use_time_distributed=False,
						activation=None)(decoder_ffn_output, decoder_outputs1)	
		else:
			decoder_outputs1 = decoder_ffn_output

		# skip connection
		decoder_outputs1, _ = GLU_with_ADDNORM(
						IF_GLU=self.cfg['IF_GLU'],
						IF_ADDNORM=self.cfg['IF_ADDNORM'],
						hidden_layer_size=self.all_layers_neurons,
						dropout_rate=self.all_layers_dropout,
						use_time_distributed=False,
						activation=None)(skip_connection, decoder_outputs1)

		# decoder_future output
		decoder_outputs2 = tf.keras.layers.TimeDistributed(
			tf.keras.layers.Dense(self.n_features_output))(decoder_outputs1)

		output = decoder_outputs2

		return output




class rnn_unit():
	def __init__(self, cfg, rnn_location, num):

		self.all_layers_dropout = cfg['all_layers_dropout']
		self.rnn_type = cfg['rnn_type']
		self.input_enc_rnn_depth = cfg['input_enc_rnn_depth']
		self.input_enc_rnn_bi = cfg['input_enc_rnn_bi']
		self.all_layers_neurons = cfg['all_layers_neurons']
		self.all_layers_neurons_rnn = int(
			self.all_layers_neurons/self.input_enc_rnn_depth)
		self.all_layers_neurons_rnn = 8 * int(self.all_layers_neurons_rnn/8)
		self.all_layers_dropout = cfg['all_layers_dropout']
		self.rnn_location = rnn_location
		self.cfg = cfg
		self.num = num

	def __call__(self, input_to_layers, init_states=None):
		if self.input_enc_rnn_depth == 1:
			return self.single_rnn_layer(x_input=input_to_layers, init_states=init_states, mid_layer=False, layername_prefix='Only_')
		else:
			x = self.single_rnn_layer(x_input=input_to_layers, init_states=init_states,
									  mid_layer=True, layername_prefix='First_')  # change
						
			x, _ = GLU_with_ADDNORM(
									IF_GLU=self.cfg['IF_GLU'],
									IF_ADDNORM=self.cfg['IF_ADDNORM'],
									hidden_layer_size=self.all_layers_neurons,
									dropout_rate=self.all_layers_dropout,
									use_time_distributed=False,
									activation=None)(input_to_layers, x)
			
			for i in range(0, self.input_enc_rnn_depth-2):
				x = self.single_rnn_layer(
					x_input=x, init_states=init_states, mid_layer=True, layername_prefix='Mid_%s_' % (i+1))
				
				x, _ = GLU_with_ADDNORM(
									IF_GLU=self.cfg['IF_GLU'],
									IF_ADDNORM=self.cfg['IF_ADDNORM'],
									hidden_layer_size=self.all_layers_neurons,
									dropout_rate=self.all_layers_dropout,
									use_time_distributed=False,
									activation=None)(input_to_layers, x)

			return self.single_rnn_layer(x_input=x, init_states=init_states, mid_layer=False, layername_prefix='Last_')

	def single_rnn_layer(self, x_input, init_states, mid_layer=False, layername_prefix=None):
		if self.rnn_type == "LSTM":
			RNN_type = tf.keras.layers.LSTM
		elif self.rnn_type == "GRU":
			RNN_type = tf.keras.layers.GRU
		elif self.rnn_type == "RNN":
			RNN_type = tf.keras.layers.SimpleRNN

		if self.rnn_location == "encoder_past":
			self.init_state = None
		elif self.rnn_location == "encoder_future" or self.rnn_location == "decoder_future":
			self.init_state = init_states

		if mid_layer:
			ret_seq = True
			ret_state = False
		else:
			ret_seq = True
			ret_state = True

		if self.input_enc_rnn_bi:
			self.layername = layername_prefix + self.rnn_location + \
				'_' + str(self.num) + '_bi' + self.rnn_type
		else:
			self.layername = layername_prefix + self.rnn_location + \
				'_' + str(self.num) + '_' + self.rnn_type

		if self.input_enc_rnn_bi:
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
