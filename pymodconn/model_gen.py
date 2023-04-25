import pymodconn.model_gen_utils as Model_utils
from pymodconn.major_layers import Encoder_class
from pymodconn.major_layers import Decoder_class
from pymodconn.utils_layers import *
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class ModelClass():
    def __init__(self, cfg, current_dt):
        self.cfg = cfg
        self.current_dt = current_dt

        if cfg['if_seed']:
            np.random.seed(cfg['seed'])
            tf.random.set_seed(cfg['seed'])
        self.epochs = cfg['epochs']
        self.batch_size = cfg['batch_size']
        self.save_models_dir = cfg['save_models_dir']
        self.save_results_dir = cfg['save_results_dir']
        self.future_data_col = cfg['future_data_col']
        self.n_past = cfg['n_past']
        self.n_features_input = cfg['n_features_input']
        self.all_layers_neurons = cfg['all_layers_neurons']
        self.input_enc_rnn_depth = cfg['input_enc_rnn_depth']
        self.all_layers_dropout = cfg['all_layers_dropout']
        self.n_future = cfg['n_future']
        self.n_features_output = cfg['n_features_output']
        self.optimizer = cfg['optimizer']  # new
        self.SGD_lr = cfg['SGD']['lr']  # new
        self.SGD_mom = cfg['SGD']['momentum']  # new
        self.Adam_lr = cfg['Adam']['lr']
        self.Adam_b1 = cfg['Adam']['b1']
        self.Adam_b2 = cfg['Adam']['b2']
        self.Adam_epsi = cfg['Adam']['epsi']
        self.loss_func = cfg['loss']
        self.model_type_prob = cfg['model_type_prob']
        self.loss_prob = cfg['loss_prob']
        self.q = cfg['quantiles']
        self.control_future_cells = cfg['control_future_cells']

        self.n_features_output_block = 1

        self.n_outputs_lastlayer = 2 if self.loss_prob == 'parametric' else len(
            self.q)

        self.metrics = cfg['metrics']
        self.mha_head = cfg['mha_head']

        self.cfg = cfg
        self.rnn_type = cfg['rnn_type']
        self.dec_attn_mask = cfg['dec_attn_mask']

        self.fit_type = cfg['fit_type']
        self.seq_len = cfg['seq_len']
        self.if_save_model_image = cfg['if_model_image']
        self.if_model_summary = cfg['if_model_summary']

        self.model_size_GB = 0

        # model structure
        self.IFRNN1 = cfg['IFRNN_input']
        self.IFRNN2 = cfg['IFRNN_output']
        self.IFSELF_MHA = cfg['IFSELF_MHA']
        self.IFCASUAL_MHA = cfg['IFCASUAL_MHA']
        self.IFCROSS_MHA = cfg['IFCROSS_MHA']

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

        if not os.path.exists(cfg['save_models_dir']):
            os.makedirs(cfg['save_models_dir'])

    def build_model(self):
        """
        Full transformer biLSTM 1 single = input > MHA > biLSTM > output
        here Input goes to MHA and then to biLSTM
        """
        timer = Model_utils.Timer()
        timer.start()
        print('[ModelClass] Model Compiling.....')
        self.future_data_col = self.future_data_col - self.control_future_cells + 1

        # input for encoder_past
        encoder_inputs = tf.keras.layers.Input(
            shape=(self.n_past, self.n_features_input), name='encoder_past_inputs')
        
        encoder_outputs_seq, encoder_outputs_allstates = Encoder_class(
            self.cfg, str(1))(encoder_inputs, init_states=None)

        decoder_outputs_list = []
        
        for i in range(1, self.control_future_cells+1):
            locals()[f"decoder_{i}_inputs"] = tf.keras.layers.Input(shape=(self.n_future, self.future_data_col), name=f"decoder_{i}_inputs")
            
            locals()[f"decoder_{i}_outputs"] = Decoder_class(self.cfg, str(i))(locals()[f"decoder_{i}_inputs"],
                                                                                encoder_outputs_seq,
                                                                                encoder_states=encoder_outputs_allstates)
            decoder_outputs_list.append(locals()[f"decoder_{i}_outputs"])
        
        decoder_outputs_all = MERGE_LIST(self.n_features_output)(decoder_outputs_list)

        # print('Probabilistic or non probabilistic changes')
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
                    [x[:, :, :, 0], soft_relu(x[:, :, :, 1])], axis=-1))(decoder_outputs4)
        elif self.model_type_prob == 'nonprob':
            decoder_outputs4 = tf.keras.layers.Lambda(
                lambda x: tf.clip_by_value(x, -1, 1))(decoder_outputs_all)
        else:
            raise ValueError(
                'model_type_prob should be either prob or nonprob')

        decoder_inputs = []
        for i in range(1, self.control_future_cells+1):
            decoder_inputs.append(locals()[f"decoder_{i}_inputs"])

        self.model = Model(
            [encoder_inputs, decoder_inputs], decoder_outputs4)

        Model_utils.Build_utils(
            self.cfg, self.current_dt).postbuild_model(self.model)

    def load_model(self, filepath):
        print('[ModelClass_tfp] Loading model from file %s' % filepath)
        # self.model = load_model(filepath)
        self.model.load_weights(filepath)
        # https://stackoverflow.com/a/69663259/6510598

    def forget_model(self):
        del self.model
        K.clear_session()
        print("Everything forgoten....maybe")
