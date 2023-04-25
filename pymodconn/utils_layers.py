from keras.utils.layer_utils import count_params
import os
from typing import *

import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import datetime as dt
K = tf.keras.backend

def soft_relu(x):
	"""
	Soft ReLU activation function, used for ensuring the positivity of the standard deviation of the Normal distribution
	when using the parameteric loss function. See Section 3.2.2 in the DeepTCN paper.
	"""
	return tf.math.log(1.0 + tf.math.exp(x))

class get_causal_attention_mask(tf.keras.layers.Layer):
	def __init__(self):
		super().__init__()

	def call(self, inputs):
		input_shape = tf.shape(inputs)
		batch_size, seq_length, num_feats = input_shape[0], input_shape[1], input_shape[2]
		i = tf.range(seq_length)[:, tf.newaxis]
		j = tf.range(seq_length)
		mask = tf.cast(i >= j, dtype="int32")
		mask = tf.reshape(mask, (1, seq_length, seq_length))
		mult = tf.concat(
			[tf.expand_dims(batch_size, -1),
			 tf.constant([1, 1], dtype=tf.int32)],
			axis=0)
		return tf.tile(mask, mult)

class get_causal_attention_mask1(tf.keras.layers.Layer):
	"""Returns causal mask to apply for self-attention layer.

	Args:
		self_attn_inputs: Inputs to self attention layer to determine mask shape
	"""
	def __init__(self):
		super().__init__()
		

	def call(self, self_attn_inputs):
		len_s = tf.shape(input=self_attn_inputs)[1]
		bs = tf.shape(input=self_attn_inputs)[:1]
		mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
		return mask



class linear_layer(tf.keras.layers.Layer):
	def __init__(self, hidden_layer_size, activation=None, use_time_distributed=False, use_bias=True):
		super().__init__()
		self.use_time_distributed = use_time_distributed
		self.activation = activation
		self.use_bias = use_bias
		self.hidden_layer_size = hidden_layer_size
		self.dense_layer = tf.keras.layers.Dense(self.hidden_layer_size, activation=self.activation, use_bias=self.use_bias)
		self.td_layer = tf.keras.layers.TimeDistributed(self.dense_layer)	

	def call(self, x, training=False):

		if self.use_time_distributed:
			x = self.td_layer(x)
		else:
			x = self.dense_layer(x)
		
		return x



class GRN_layer(tf.keras.layers.Layer):
	"""
	Applies the gated residual network (GRN) as defined in the TFT paper
	
	Adapted from 
	https://github.com/greatwhiz/tft_tf2/blob/HEAD/libs/tft_model.py
	"""
	def __init__(self, hidden_layer_size, output_size, dropout_rate=None, use_time_distributed=False, activation_layer_type='elu'):
		super().__init__()
		self.hidden_layer_size = hidden_layer_size
		self.output_size = output_size
		self.dropout_rate = dropout_rate
		self.use_time_distributed = use_time_distributed
		self.activation_layer_type = activation_layer_type
		self.linear_layer1 = linear_layer(self.hidden_layer_size,
										activation=None,
										use_time_distributed=self.use_time_distributed)
		self.linear_layer2 = linear_layer(self.hidden_layer_size,
										activation=None,
										use_time_distributed=self.use_time_distributed)
		
		self.activation_layer = tf.keras.layers.Activation(self.activation_layer_type)
		
		self.gluwithaddnorm = GLU_with_ADDNORM(self.output_size, self.dropout_rate, self.use_time_distributed, None)


	def call(self, x, training=False):
		skip = x

		hidden = self.linear_layer1(x)
		hidden = self.activation_layer(hidden)
		hidden = self.linear_layer2(hidden)

		grn_output, gate = self.gluwithaddnorm(skip, hidden)
		
		return grn_output, gate


class GLU_with_ADDNORM(tf.keras.layers.Layer):
	def __init__(self, output_layer_size, dropout_rate, use_time_distributed=True, activation=None):
		super().__init__()
		self.output_layer_size = output_layer_size
		self.dropout_rate = dropout_rate
		self.use_time_distributed = use_time_distributed
		self.activation = activation
		self.linear_layer1 = linear_layer(self.output_layer_size,
										activation=None,
										use_time_distributed=self.use_time_distributed)
		self.GLU_layer = GLU_layer(output_layer_size = self.output_layer_size,
						  dropout_rate = self.dropout_rate,
						  use_time_distributed = self.use_time_distributed,
						  activation=None)
		self.ADD_NORM_layer = ADD_NORM()
	
	def call(self, skip, x, training=False):
		x, gate = self.GLU_layer(x)
		
		x = self.linear_layer1(x)
		
		x = self.ADD_NORM_layer(skip, x)
		return x, gate

class ADD_NORM(tf.keras.layers.Layer):
	def __init__(self):
		super().__init__()
		self.add_layer = tf.keras.layers.Add()
		self.norm_layer = tf.keras.layers.LayerNormalization()

	def call(self, skip, x, training=False):
		x = [skip, x]
		tmp = self.add_layer(x)
		tmp = self.norm_layer(tmp)
		return tmp

class GLU_layer(tf.keras.layers.Layer):
	def __init__(self, output_layer_size, dropout_rate, use_time_distributed=True, activation=None):
		super().__init__()
		self.output_layer_size = output_layer_size
		self.dropout_rate = dropout_rate
		self.use_time_distributed = use_time_distributed
		self.activation = activation
		self.dr_layer = tf.keras.layers.Dropout(self.dropout_rate)
		self.dense_layer = tf.keras.layers.Dense(self.output_layer_size, activation=self.activation)
		self.dense_sigmoid_layer = tf.keras.layers.Dense(self.output_layer_size, activation='sigmoid')
		self.multiply_layer = tf.keras.layers.Multiply()
		self.td_layer = tf.keras.layers.TimeDistributed(self.dense_layer)
		self.td_sigmoid_layer = tf.keras.layers.TimeDistributed(self.dense_sigmoid_layer)

	def call(self, x, training=False):
		if self.dropout_rate is not None:
			x = self.dr_layer(x, training=training)

		if self.use_time_distributed:
			activation_layer = self.td_layer(x)
			gated_layer = self.td_sigmoid_layer(x)
		else:
			activation_layer = self.dense_layer(x)
			gated_layer = self.dense_sigmoid_layer(x)

		x, gate = self.multiply_layer([activation_layer, gated_layer]), gated_layer

		return x, gate



class STATES_MANIPULATION_BLOCK():
	""" Merge states of two different RNNs
	Concates the states and then applies a dense layer
	"""

	def __init__(self, d1, states_manipulation_method):
		self.states_manipulation_method = states_manipulation_method
		self.conc = tf.keras.layers.Concatenate()
		self.dense = linear_layer(d1, activation=None, use_time_distributed=False, use_bias=True)
		self.add_layer = tf.keras.layers.Add()
		self.norm_layer = tf.keras.layers.LayerNormalization()

	def __call__(self, x1, x2):
		if x1 == None or x2 == None or x1 == [] or x2 == []:
			return None
		elif self.states_manipulation_method == 1:
			return None
		elif self.states_manipulation_method == 2:
			return x1
		elif self.states_manipulation_method == 3:
			return x2
		elif self.states_manipulation_method == 4:
			all_x = []
			len_x = len(x1)
			for i in range(len_x):
				a = x1[i]
				b = x2[i]
				x = self.conc([a, b])
				x = self.dense(x)
				all_x.append(x)
			return all_x
		elif self.states_manipulation_method == 5:
			all_x = []
			len_x = len(x1)
			for i in range(len_x):
				a = x1[i]
				b = x2[i]
				x = self.add_layer([a, b])
				x = self.dense(x)
				all_x.append(x)
			return all_x
		elif self.states_manipulation_method == 6:
			all_x = []
			len_x = len(x1)
			for i in range(len_x):
				a = x1[i]
				b = x2[i]
				x = self.add_layer([a, b])
				x = self.norm_layer(x)
				x = self.dense(x)
				all_x.append(x)
			return all_x
		elif self.states_manipulation_method == 7:
			all_x = []
			len_x = len(x1)
			for i in range(len_x):
				a = x1[i]
				b = x2[i]
				x = self.add_layer([a, b])
				all_x.append(x)
			return all_x
		elif self.states_manipulation_method == 8:
			all_x = []
			len_x = len(x1)
			for i in range(len_x):
				a = x1[i]
				b = x2[i]
				x = self.add_layer([a, b])
				x = self.norm_layer(x)
				all_x.append(x)
			return all_x		
		else:
			print('only 8 methods defined, returning None')
			return None




class MERGE_LIST(tf.keras.layers.Layer):
	""" Takes a list of tensors and concats them
	Concates the states and then applies a dense layer
	"""
	def __init__(self, d1):
		super().__init__()
		self.conc = tf.keras.layers.Concatenate()
		self.dense1 = tf.keras.layers.Dense(d1+2)
		self.dense2 = tf.keras.layers.Dense(d1)

	def call(self, x1, training=False):
		x = self.conc(x1)
		x = self.dense1(x)
		x = self.dense2(x)
		return x



class rnn_unit():
	def __init__(self, cfg, rnn_location, num):  #rnn_location = 'encoder_input' or 'decoder_input' or 'decoder_output'

		self.all_layers_dropout = cfg['all_layers_dropout']
		self.rnn_type = cfg['rnn_type']
		self.input_enc_rnn_depth = cfg['input_enc_rnn_depth']
		self.input_enc_rnn_bi = cfg['input_enc_rnn_bi']
		self.all_layers_neurons = cfg['all_layers_neurons']
		self.all_layers_neurons_rnn = int(
			self.all_layers_neurons/self.input_enc_rnn_depth)
		self.all_layers_neurons_rnn = 8 * int(self.all_layers_neurons_rnn/8)
		assert self.all_layers_neurons_rnn > 0, 'all_layers_neurons_rnn should be > 0...change all_layers_neurons or input_enc_rnn_depth in config file'
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
						
			if self.cfg['IF_NONE_GLUADDNORM_ADDNORM'] == 1:
				x, _ = GLU_with_ADDNORM(            #---------------> fix this
									output_layer_size=self.all_layers_neurons,
									dropout_rate=self.all_layers_dropout,
									use_time_distributed=False,
									activation=None)(input_to_layers, x)
			elif self.cfg['IF_NONE_GLUADDNORM_ADDNORM'] == 2:
				x = ADD_NORM()(input_to_layers, x)
			
			for i in range(0, self.input_enc_rnn_depth-2):
				x = self.single_rnn_layer(
					x_input=x, init_states=init_states, mid_layer=True, layername_prefix='Mid_%s_' % (i+1))
				

				if self.cfg['IF_NONE_GLUADDNORM_ADDNORM'] == 1:
					x, _ = GLU_with_ADDNORM(            #---------------> fix this
										output_layer_size=self.all_layers_neurons,
										dropout_rate=self.all_layers_dropout,
										use_time_distributed=False,
										activation=None)(input_to_layers, x)
				elif self.cfg['IF_NONE_GLUADDNORM_ADDNORM'] == 2:
					x = ADD_NORM()(input_to_layers, x)


			return self.single_rnn_layer(x_input=x, init_states=init_states, mid_layer=False, layername_prefix='Last_')

	def single_rnn_layer(self, x_input, init_states, mid_layer=False, layername_prefix=None):
		if self.rnn_type == "LSTM":
			RNN_type = tf.keras.layers.LSTM
		elif self.rnn_type == "GRU":
			RNN_type = tf.keras.layers.GRU
		elif self.rnn_type == "RNN":
			RNN_type = tf.keras.layers.SimpleRNN

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

