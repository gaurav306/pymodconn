import tensorflow as tf
from pymodconn.utils_layers import GLU_with_ADDNORM
from pymodconn.utils_layers import ADD_NORM
from pymodconn.utils_layers import GRN_layer

K = tf.keras.backend

class MHA_block_class():
	def __init__(self, cfg, enc_or_dec, enc_or_dec_number, self_or_crossMHA, mha_depth_index):
		self.cfg = cfg
		self.enc_or_dec = enc_or_dec
		assert self.enc_or_dec in ['encoder', 'decoder'], 'enc_or_dec must be one of encoder, decoder'
		self.enc_or_dec_number = enc_or_dec_number
		self.self_or_crossMHA = self_or_crossMHA
		assert self.self_or_crossMHA in ['self', 'cross'], 'self_or_crossMHA must be one of self, cross'
		self.mha_depth_index = mha_depth_index
		
		self.IF_GRN_block 				= self.cfg[self.enc_or_dec][self.self_or_crossMHA +'_MHA_block']['IF_GRN_block']
		self.IF_MHA 					= self.cfg[self.enc_or_dec][self.self_or_crossMHA +'_MHA_block']['IF_MHA']
		self.mha_head 					= self.cfg[self.enc_or_dec][self.self_or_crossMHA +'_MHA_block']['MHA_head']
		self.IF_NONE_GLUADDNORM_ADDNORM = self.cfg[self.enc_or_dec][self.self_or_crossMHA +'_MHA_block']['IF_NONE_GLUADDNORM_ADDNORM_deep']
		
		self.known_past_features 		= self.cfg['known_past_features']
		self.all_layers_neurons 		= self.cfg['all_layers_neurons']
		self.all_layers_dropout 		= self.cfg['all_layers_dropout']

	def __call__(self, input_q, input_kv):
		if self.IF_MHA:
			
			self.mha_layer_name = self.enc_or_dec + '_' + str(self.enc_or_dec_number) + '_'+ str(self.self_or_crossMHA) + 'MHA-' + str(self.mha_depth_index)

			encoder_mha = tf.keras.layers.MultiHeadAttention(
								num_heads = self.mha_head,
								key_dim = self.known_past_features,
								value_dim = self.known_past_features,
								name=self.mha_layer_name)	

			output_cell = encoder_mha(query=input_q,
										key=input_kv,
										value=input_kv,
										training=True)

			if self.IF_NONE_GLUADDNORM_ADDNORM == 1:
				output_cell = GLU_with_ADDNORM(            
									output_layer_size=self.all_layers_neurons,
									dropout_rate=self.all_layers_dropout,
									use_time_distributed=False,
									activation=None)(input_q, output_cell)
			
			elif self.IF_NONE_GLUADDNORM_ADDNORM == 2:
				output_cell = ADD_NORM()(input_q, output_cell)
			
			if (self.IF_GRN_block == 1) and ((self.enc_or_dec == 'encoder' and self.self_or_crossMHA == 'self') or (self.enc_or_dec == 'decoder' and self.self_or_crossMHA == 'cross')):
				output_cell = GRN_layer(
								hidden_layer_size = self.all_layers_neurons,
								output_size = self.all_layers_neurons,
								dropout_rate = self.all_layers_dropout,
								use_time_distributed = True,
								activation_layer_type = 'elu')(output_cell)					
		else:
			output_cell = input_q
		
		return output_cell

