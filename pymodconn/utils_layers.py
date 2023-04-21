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

class get_causal_attention_mask1():
	def __init__(self):
		pass

	def __call__(self, inputs):
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


class get_causal_attention_mask():
	def __init__(self):
		pass

	def __call__(self, self_attn_inputs):
		"""Returns causal mask to apply for self-attention layer.

		Args:
			self_attn_inputs: Inputs to self attention layer to determine mask shape
		"""
		len_s = tf.shape(input=self_attn_inputs)[1]
		bs = tf.shape(input=self_attn_inputs)[:1]
		mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
		return mask




class FeedForward():
	def __init__(self, d1, dropout_rate=0.1):
		self.dense1 = tf.keras.layers.Dense(d1*4, activation='relu')
		self.dense2 = tf.keras.layers.Dense(d1)
		self.dropout = tf.keras.layers.Dropout(dropout_rate)

	def __call__(self, x):
		x = self.dense1(x)
		x = self.dense2(x)
		x = self.dropout(x)
		return x


def linear_layer(size,
				 activation=None,
				 use_time_distributed=False,
				 use_bias=True):
	"""Returns simple Keras linear layer.
	"""
	linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
	if use_time_distributed:
		linear = tf.keras.layers.TimeDistributed(linear)
	return linear


class GRN_layer():
	"""
	Applies the gated residual network (GRN) as defined in the TFT paper
	
	Adapted from 
	https://github.com/greatwhiz/tft_tf2/blob/HEAD/libs/tft_model.py
	"""
	def __init__(self, hidden_layer_size, output_size, dropout_rate=None, use_time_distributed=False, activation_layer='elu'):
		self.hidden_layer_size = hidden_layer_size
		self.output_size = output_size
		self.dropout_rate = dropout_rate
		self.use_time_distributed = use_time_distributed
		self.activation_layer = activation_layer

	def __call__(self, x):
		skip = x

		hidden = linear_layer(
			self.hidden_layer_size,
			activation=None,
			use_time_distributed=self.use_time_distributed)(x)
		hidden = tf.keras.layers.Activation(self.activation_layer)(hidden)
		hidden = linear_layer(
			self.hidden_layer_size,
			activation=None,
			use_time_distributed=self.use_time_distributed)(hidden)

		grn_output, gate = GLU_with_ADDNORM(IF_GLU=True,
											  IF_ADDNORM=True,
											  hidden_layer_size=self.output_size,
											  dropout_rate=self.dropout_rate,
											  use_time_distributed=self.use_time_distributed,
											  activation=None)(skip, hidden)
		
		return grn_output, gate


class GLU_with_ADDNORM():
	def __init__(self, IF_GLU, IF_ADDNORM , hidden_layer_size, dropout_rate, use_time_distributed=True, activation=None):
		self.hidden_layer_size = hidden_layer_size
		self.dropout_rate = dropout_rate
		self.use_time_distributed = use_time_distributed
		self.activation = activation
		self.IF_GLU = IF_GLU
		self.IF_ADDNORM = IF_ADDNORM

	def __call__(self, skip, x):
		if self.IF_GLU:
			x, gate = GLU_layer(hidden_layer_size = self.hidden_layer_size,
						  dropout_rate = self.dropout_rate,
						  use_time_distributed = self.use_time_distributed,
						  activation=None)(x)
		else:
			gate = []
		if self.IF_ADDNORM:
			x = ADD_NORM(hidden_layer_size = self.hidden_layer_size,
						 use_time_distributed = self.use_time_distributed)(skip, x)
		return x, gate

class ADD_NORM():
	def __init__(self, hidden_layer_size, use_time_distributed=True):
		self.hidden_layer_size = hidden_layer_size
		self.use_time_distributed = use_time_distributed

	def __call__(self, skip, x):
		linear = tf.keras.layers.Dense(self.hidden_layer_size)
		if self.use_time_distributed:
			linear = tf.keras.layers.TimeDistributed(linear)
		skip = linear(skip)
		x = [skip, x]
		tmp = tf.keras.layers.Add()(x)
		tmp = tf.keras.layers.LayerNormalization()(tmp)
		return tmp

class GLU_layer():
	def __init__(self, hidden_layer_size, dropout_rate, use_time_distributed=True, activation=None):
		self.hidden_layer_size = hidden_layer_size
		self.dropout_rate = dropout_rate
		self.use_time_distributed = use_time_distributed
		self.activation = activation

	def __call__(self, x):
		if self.dropout_rate is not None:
			x = tf.keras.layers.Dropout(self.dropout_rate)(x)

		if self.use_time_distributed:
			activation_layer = tf.keras.layers.TimeDistributed(
				tf.keras.layers.Dense(self.hidden_layer_size, activation=self.activation))(
				x)
			gated_layer = tf.keras.layers.TimeDistributed(
				tf.keras.layers.Dense(self.hidden_layer_size, activation='sigmoid'))(
				x)
		else:
			activation_layer = tf.keras.layers.Dense(
				self.hidden_layer_size, activation=self.activation)(
				x)
			gated_layer = tf.keras.layers.Dense(
				self.hidden_layer_size, activation='sigmoid')(
				x)

		x, gate = tf.keras.layers.Multiply()([activation_layer, gated_layer]), gated_layer

		return x, gate



class MERGE_STATES():
	""" Merge states of two different RNNs
	Concates the states and then applies a dense layer
	"""

	def __init__(self, d1):
		self.conc = tf.keras.layers.Concatenate()
		self.dense = tf.keras.layers.Dense(d1)

	def __call__(self, x1, x2):
		all_x = []
		len_x = len(x1)
		for i in range(len_x):
			a = x1[i]
			b = x2[i]
			x = self.conc([a, b])
			x = self.dense(x)
			all_x.append(x)
		return all_x


class MERGE_LIST():
	""" Takes a list of tensors and concats them
	Concates the states and then applies a dense layer
	"""

	def __init__(self, d1):
		self.conc = tf.keras.layers.Concatenate()
		self.dense1 = tf.keras.layers.Dense(d1+2)
		self.dense2 = tf.keras.layers.Dense(d1)

	def __call__(self, x1):
		x = self.conc(x1)
		x = self.dense1(x)
		x = self.dense2(x)
		return x

