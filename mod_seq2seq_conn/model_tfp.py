import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


import mod_seq2seq_conn.model_tfp_utils as Model_utils

class ModelClass_tfp():
	"""A class for an building and inferencing an lstm model"""

	def __init__(self, configs, current_dt):
		self.configs = configs
		self.current_dt = current_dt

		if configs['model']['if_seed']:
			np.random.seed(configs['model']['seed'])
			tf.random.set_seed(configs['model']['seed'])
		self.epochs = configs['training']['epochs']
		self.batch_size = configs['training']['batch_size']
		self.save_models_dir = configs['model']['save_models_dir']
		self.save_results_dir = configs['model']['save_results_dir']
		self.future_data_col = configs['data']['future_data_col']
		self.n_past = configs['data']['n_past']
		self.n_features_input = configs['data']['n_features_input']
		self.all_layers_neurons = configs['model']['all_layers_neurons']
		self.input_enc_rnn_depth = configs['rnn_units']['input_enc_rnn_depth']
		self.all_layers_dropout = configs['model']['all_layers_dropout']
		self.n_future = configs['data']['n_future']
		self.n_features_output = configs['data']['n_features_output']
		self.optimizer = configs['model']['optimizer']  # new
		self.SGD_lr = configs['model']['SGD']['lr']  # new
		self.SGD_mom = configs['model']['SGD']['momentum']  # new
		self.Adam_lr = configs['model']['Adam']['lr']
		self.Adam_b1 = configs['model']['Adam']['b1']
		self.Adam_b2 = configs['model']['Adam']['b2']
		self.Adam_epsi = configs['model']['Adam']['epsi']
		self.loss_func = configs['model']['loss']
		self.model_type_prob = configs['model']['model_type_prob']
		self.loss_prob = configs['model']['loss_prob']
		self.q = configs['model']['quantiles']
		self.control_future_cells = configs['model']['control_future_cells']

		self.n_features_output_block = 1

		self.n_outputs_lastlayer = 2 if self.loss_prob == 'parametric' else len(self.q)

		self.metrics = configs['model']['metrics']
		self.mha_head = configs['model']['mha_head']

		self.configs = configs
		self.rnn_type = configs['rnn_units']['rnn_type']
		self.dec_attn_mask = configs['model']['dec_attn_mask']

		self.fit_type = configs['training']['fit_type']
		self.seq_len = configs['training']['seq_len']
		self.if_save_model_image = configs['model']['if_model_image']
		self.if_model_summary = configs['model']['if_model_summary']

		self.model_size_GB = 0

		# model structure
		self.IF_GLU 			= configs['model']['IF_GLU']
		self.IF_ADDNORM 		= configs['model']['IF_ADDNORM']
		self.IFFFN 				= configs['model']['IFFFN']
		self.IFRNN1 			= configs['model']['IFRNN_input']
		self.IFRNN2 			= configs['model']['IFRNN_output']
		self.IFSELF_MHA        	=configs['model']['IFSELF_MHA']
		self.IFCASUAL_MHA      	=configs['model']['IFCASUAL_MHA']
		self.IFCROSS_MHA       	=configs['model']['IFCROSS_MHA']


		self.save_training_history_file = os.path.join(
			self.save_results_dir, '%s.csv' % (self.current_dt))
		self.save_training_history = os.path.join(
			self.save_results_dir, '%s_history.png' % (self.current_dt))
		self.save_hf5_name = os.path.join(
			self.save_models_dir, '%s.h5' % (self.current_dt))
		self.save_modelimage_name = os.path.join(
			self.save_models_dir, '%s_modelimage.png' % (self.current_dt))
		self.save_modelsummary_name = os.path.join(
			self.save_models_dir, '%s_modelsummary.txt' % (self.current_dt))

		if not os.path.exists(configs['model']['save_models_dir']): os.makedirs(configs['model']['save_models_dir'])
		
	def build_model(self):
		"""
		Full transformer biLSTM 1 single = input > MHA > biLSTM > output
		here Input goes to MHA and then to biLSTM
		"""
		timer = Model_utils.Timer()
		timer.start()
		# print('[ModelClass] Model e1d1_wFuture_6_full_transformercs Compiling.....')
		self.future_data_col = self.future_data_col - self.control_future_cells + 1

		# input for encoder_past
		encoder_past1_inputs = tf.keras.layers.Input(
			shape=(self.n_past, self.n_features_input), name='encoder_past_inputs')
		encoder_outputs_past_seq, encoder_outputs_past_allstates = Model_utils.Input_encoder_MHA_RNN(
			self.configs, location='encoder_past', num=1)(encoder_past1_inputs, init_states=None)

		self.merge_states_units = int(self.all_layers_neurons/self.input_enc_rnn_depth)
		self.merge_states_units = 8 * int(self.merge_states_units/8)
		#print('input for encoder_future1')
		# input for encoder_future1
		nx = 1
		encoder_future_1_inputs = tf.keras.layers.Input(shape=(
			self.n_future, self.future_data_col), name='encoder_future_%s_inputs' % str(nx))
		encoder_outputs_future_1_seq, encoder_outputs_future_1_allstates = Model_utils.Input_encoder_MHA_RNN(self.configs, location='encoder_future', num=nx)(encoder_future_1_inputs,
                                                                                                                                                        init_states=encoder_outputs_past_allstates)
		encoders_1_allstates = Model_utils.MERGE_STATES(self.merge_states_units)(
			encoder_outputs_past_allstates, encoder_outputs_future_1_allstates)
		decoder_outputs_1 = Model_utils.Output_decoder_crossMHA_RNN(self.configs, location='decoder_future', num=nx)(encoder_outputs_future_1_seq,
                                                                                                               encoder_outputs_past_seq,
                                                                                                               init_states=encoders_1_allstates)
		if self.control_future_cells == 6:
			#print('input for encoder_future2')
			# input for encoder_future2
			nx = 2
			encoder_future_2_inputs = tf.keras.layers.Input(shape=(
				self.n_future, self.future_data_col), name='encoder_future_%s_inputs' % str(nx))
			encoder_outputs_future_2_seq, encoder_outputs_future_2_allstates = Model_utils.Input_encoder_MHA_RNN(self.configs, location='encoder_future', num=nx)(encoder_future_2_inputs,
																																							init_states=encoder_outputs_past_allstates)
			encoders_2_allstates = Model_utils.MERGE_STATES(self.merge_states_units)(
				encoder_outputs_past_allstates, encoder_outputs_future_2_allstates)
			decoder_outputs_2 = Model_utils.Output_decoder_crossMHA_RNN(self.configs, location='decoder_future', num=nx)(encoder_outputs_future_2_seq,
																												encoder_outputs_past_seq,
																												init_states=encoders_2_allstates)
			#print('input for encoder_future3')
			# input for encoder_future3
			nx = 3
			encoder_future_3_inputs = tf.keras.layers.Input(shape=(
				self.n_future, self.future_data_col), name='encoder_future_%s_inputs' % str(nx))
			encoder_outputs_future_3_seq, encoder_outputs_future_3_allstates = Model_utils.Input_encoder_MHA_RNN(self.configs, location='encoder_future', num=nx)(encoder_future_3_inputs,
																																							init_states=encoder_outputs_past_allstates)
			encoders_3_allstates = Model_utils.MERGE_STATES(self.merge_states_units)(
				encoder_outputs_past_allstates, encoder_outputs_future_3_allstates)
			decoder_outputs_3 = Model_utils.Output_decoder_crossMHA_RNN(self.configs, location='decoder_future', num=nx)(encoder_outputs_future_3_seq,
																												encoder_outputs_past_seq,
																												init_states=encoders_3_allstates)
			#print('input for encoder_future4')
			# input for encoder_future4
			nx = 4
			encoder_future_4_inputs = tf.keras.layers.Input(shape=(
				self.n_future, self.future_data_col), name='encoder_future_%s_inputs' % str(nx))
			encoder_outputs_future_4_seq, encoder_outputs_future_4_allstates = Model_utils.Input_encoder_MHA_RNN(self.configs, location='encoder_future', num=nx)(encoder_future_4_inputs,
																																							init_states=encoder_outputs_past_allstates)
			encoders_4_allstates = Model_utils.MERGE_STATES(self.merge_states_units)(
				encoder_outputs_past_allstates, encoder_outputs_future_4_allstates)
			decoder_outputs_4 = Model_utils.Output_decoder_crossMHA_RNN(self.configs, location='decoder_future', num=nx)(encoder_outputs_future_4_seq,
																												encoder_outputs_past_seq,
																												init_states=encoders_4_allstates)
			#print('input for encoder_future5')
			# input for encoder_future5
			nx = 5
			encoder_future_5_inputs = tf.keras.layers.Input(shape=(
				self.n_future, self.future_data_col), name='encoder_future_%s_inputs' % str(nx))
			encoder_outputs_future_5_seq, encoder_outputs_future_5_allstates = Model_utils.Input_encoder_MHA_RNN(self.configs, location='encoder_future', num=nx)(encoder_future_5_inputs,
																																							init_states=encoder_outputs_past_allstates)
			encoders_5_allstates = Model_utils.MERGE_STATES(self.merge_states_units)(
				encoder_outputs_past_allstates, encoder_outputs_future_5_allstates)
			decoder_outputs_5 = Model_utils.Output_decoder_crossMHA_RNN(self.configs, location='decoder_future', num=nx)(encoder_outputs_future_5_seq,
																												encoder_outputs_past_seq,
																												init_states=encoders_5_allstates)
			#print('input for encoder_future6')
			# input for encoder_future6
			nx = 6
			encoder_future_6_inputs = tf.keras.layers.Input(shape=(
				self.n_future, self.future_data_col), name='encoder_future_%s_inputs' % str(nx))
			encoder_outputs_future_6_seq, encoder_outputs_future_6_allstates = Model_utils.Input_encoder_MHA_RNN(self.configs, location='encoder_future', num=nx)(encoder_future_6_inputs,
																																							init_states=encoder_outputs_past_allstates)
			encoders_6_allstates = Model_utils.MERGE_STATES(self.merge_states_units)(
				encoder_outputs_past_allstates, encoder_outputs_future_6_allstates)
			decoder_outputs_6 = Model_utils.Output_decoder_crossMHA_RNN(self.configs, location='decoder_future', num=nx)(encoder_outputs_future_6_seq,
																												encoder_outputs_past_seq,
																												init_states=encoders_6_allstates)
		if self.control_future_cells == 6:
			#print('Final merging')
			decoder_outputs_all = Model_utils.MERGE_LIST(self.n_features_output)(
				[decoder_outputs_1, decoder_outputs_2, decoder_outputs_3, decoder_outputs_4, decoder_outputs_5, decoder_outputs_6])
		if self.control_future_cells == 1:
			decoder_outputs_all = decoder_outputs_1
		
		#print('Probabilistic or non probabilistic changes')
		# If or not using the probabilistic loss, reshape the output to match the shape required by the loss function.
		if self.model_type_prob == 'prob':
			decoder_outputs3 = tf.keras.layers.Dense(
				units=self.n_features_output * self.n_outputs_lastlayer)(decoder_outputs_all)  # this one
			# Reshape the encoder output to match the shape required by the loss function.
			decoder_outputs4 = tf.keras.layers.Reshape(target_shape=(
				self.n_future, self.n_features_output, self.n_outputs_lastlayer))(decoder_outputs3)
			# If using the parametric loss, apply the soft ReLU activation to ensure a positive standard deviation.
			if self.loss_prob == 'parametric':
				decoder_outputs4 = tf.keras.layers.Lambda(function=lambda x: tf.stack(
					[x[:, :, :, 0], Model_utils.soft_relu(x[:, :, :, 1])], axis=-1))(decoder_outputs4)
		elif self.model_type_prob == 'nonprob':
			decoder_outputs4 = tf.keras.layers.Lambda(
				lambda x: tf.clip_by_value(x, -1, 1))(decoder_outputs_all)
		else:
			raise ValueError(
				'model_type_prob should be either prob or nonprob')

		if self.control_future_cells == 6:
			encoder_future_inputs = [encoder_future_1_inputs,
							encoder_future_2_inputs,
							encoder_future_3_inputs,
							encoder_future_4_inputs,
							encoder_future_5_inputs,
							encoder_future_6_inputs]
		if self.control_future_cells == 1:
			encoder_future_inputs = [encoder_future_1_inputs]
		
			
		self.model = Model(
			[encoder_past1_inputs, encoder_future_inputs], decoder_outputs4)

		return Model_utils.Build_utils(
					self.configs, self.current_dt).postbuild_model(self.model)
		# print('[ModelClass_tfp] Model Compiled')

	def load_model(self, filepath):
		print('[ModelClass_tfp] Loading model from file %s' % filepath)
		#self.model = load_model(filepath)
		self.model.load_weights(filepath)
		# https://stackoverflow.com/a/69663259/6510598

	def forget_model(self):
		del self.model
		K.clear_session()
		print("Everything forgoten....maybe")
