from typing import *

import numpy as np
import tensorflow as tf
K = tf.keras.backend

def soft_relu(x):
	"""
	Soft ReLU activation function, used for ensuring the positivity of the standard deviation of the Normal distribution
	when using the parameteric loss function. See Section 3.2.2 in the DeepTCN paper.
	"""
	return tf.math.log(1.0 + tf.math.exp(x))


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
	
	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'hidden_layer_size': self.hidden_layer_size,
			'activation': self.activation,
			'use_time_distributed': self.use_time_distributed,
			'use_bias': self.use_bias,
		})
		return config


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

		grn_output = self.gluwithaddnorm(skip, hidden)
		
		return grn_output

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'hidden_layer_size': self.hidden_layer_size,
			'output_size': self.output_size,
			'dropout_rate': self.dropout_rate,
			'use_time_distributed': self.use_time_distributed,
			'activation_layer_type': self.activation_layer_type,
		})
		return config

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
		x = self.GLU_layer(x)
		
		x = self.linear_layer1(x)
		
		x = self.ADD_NORM_layer(skip, x)
		return x

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'output_layer_size': self.output_layer_size,
			'dropout_rate': self.dropout_rate,
			'use_time_distributed': self.use_time_distributed,
			'activation': self.activation,
		})
		return config

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

	def get_config(self):
		config = super().get_config().copy()
		return config

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

		x = self.multiply_layer([activation_layer, gated_layer])

		return x

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'output_layer_size': self.output_layer_size,
			'dropout_rate': self.dropout_rate,
			'use_time_distributed': self.use_time_distributed,
			'activation': self.activation,
		})
		return config


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
		self.dense1 = tf.keras.layers.Dense(d1*4)
		self.dense2 = tf.keras.layers.Dense(d1*2)
		self.dense3 = tf.keras.layers.Dense(d1)

	def call(self, x1, training=False):
		x = self.conc(x1)
		x = self.dense1(x)
		x = self.dense2(x)
		x = self.dense3(x)
		return x
	
	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'd1': self.dense1.units,  # Assuming 'd1' corresponds to the units of the first dense layer
		})
		return config


def positional_encoding(position, d_model):
	def get_angles(pos, i, d_model):
		angle_rates = 1 / np.power(1000, (2 * (i // 2)) / np.float32(d_model))
		return pos * angle_rates
	
	angle_rads = get_angles(np.arange(position)[:, np.newaxis],
							 np.arange(d_model)[np.newaxis, :],
							 d_model)
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
	pos_encoding = angle_rads[np.newaxis, ...]
	return tf.cast(pos_encoding, dtype=tf.float32)
