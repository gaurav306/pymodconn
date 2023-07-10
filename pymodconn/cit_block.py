import tensorflow as tf
from pymodconn.utils_layers import positional_encoding
from pymodconn.MHA_block_class_function import MHA_block_class
from pymodconn.utils_layers import GLU_with_ADDNORM
from pymodconn.utils_layers import ADD_NORM

K = tf.keras.backend


class CIT_block():
	def __init__(self, cfg, enc_or_dec_number):
		self.cfg = cfg
		self.option = self.cfg['decoder']['CIT_option']
		self.enc_or_dec_number = enc_or_dec_number

		self.n_past = self.cfg['n_past']
		self.n_future = self.cfg['n_future']
		self.known_past_features = self.cfg['known_past_features']
		self.unknown_future_features = self.cfg['unknown_future_features']
		self.known_future_features = self.cfg['known_future_features']

		self.all_layers_neurons = self.cfg['all_layers_neurons']
		self.all_layers_dropout = self.cfg['all_layers_dropout']

		
	
	def __call__(self, input_cell, input_enc):
		if self.option == 1:
			input_enc = tf.keras.layers.Reshape((self.n_future, -1))(input_enc)
			for i in range(self.cfg['decoder']['option_1_depth']):
				output_cell = tf.keras.layers.Concatenate()([input_cell, input_enc])
				output_cell = tf.keras.layers.Dense(self.all_layers_neurons)(output_cell)
				
				self.IF_NONE_GLUADDNORM_ADDNORM = self.cfg['decoder']['IF_NONE_GLUADDNORM_ADDNORM_CIT_1']

				if self.IF_NONE_GLUADDNORM_ADDNORM == 1:
					output_cell = GLU_with_ADDNORM(            
										output_layer_size=self.all_layers_neurons,
										dropout_rate=self.all_layers_dropout,
										use_time_distributed=False,
										activation=None)(input_cell, output_cell)
				
				elif self.IF_NONE_GLUADDNORM_ADDNORM == 2:
					output_cell = ADD_NORM()(input_cell, output_cell)
				input_cell = output_cell

		if self.option == 2:
			for i in range(self.cfg['decoder']['option_2_depth']):
				self.attn_type = self.cfg['decoder']['attn_type']
				if self.attn_type == 1:
					attention_block = tf.keras.layers.Attention()
				elif self.attn_type == 2:
					attention_block = tf.keras.layers.AdditiveAttention()
				elif self.attn_type == 3:
					print("Wrong attention type")
				attention_output = attention_block([input_cell, input_enc])
				output_cell = tf.keras.layers.Concatenate()([input_cell, attention_output])
				output_cell = tf.keras.layers.Dense(self.all_layers_neurons)(output_cell)

				self.IF_NONE_GLUADDNORM_ADDNORM = self.cfg['decoder']['IF_NONE_GLUADDNORM_ADDNORM_CIT_2']

				if self.IF_NONE_GLUADDNORM_ADDNORM == 1:
					output_cell = GLU_with_ADDNORM(            
										output_layer_size=self.all_layers_neurons,
										dropout_rate=self.all_layers_dropout,
										use_time_distributed=False,
										activation=None)(input_cell, output_cell)
				
				elif self.IF_NONE_GLUADDNORM_ADDNORM == 2:
					output_cell = ADD_NORM()(input_cell, output_cell)	
				input_cell = output_cell	

		if self.option == 3:
			# MHA layer for decoder, self and cross attention
			if self.cfg['decoder']['IF_SELF_CROSS_MHA'] == 1:
				for i in range(self.cfg['decoder']['SELF_CROSS_MHA_depth']):
					output_cell = MHA_block_class(self.cfg,
												'decoder',
												self.enc_or_dec_number, 
												'self',
												str(i+1))(input_cell, input_cell)        
				
					input_cell = output_cell
					output_cell = MHA_block_class(self.cfg,
												'decoder',
												self.enc_or_dec_number, 
												'cross',
												str(i+1))(input_cell, input_enc)
					input_cell = output_cell
		
		return output_cell

