from keras.utils.layer_utils import count_params
import os
from typing import *

import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import datetime as dt
K = tf.keras.backend


class Build_utils():
    """
    This class is used to compile, make sumamry and save as png.
    """

    def __init__(self, configs, current_dt):
        if configs['model']['if_seed']:
            np.random.seed(configs['model']['seed'])
            tf.random.set_seed(configs['model']['seed'])
        self.epochs = configs['training']['epochs']
        self.batch_size = configs['training']['batch_size']
        self.future_data_col = configs['data']['future_data_col']
        self.n_past = configs['data']['n_past']
        self.n_features_input = configs['data']['n_features_input']
        self.all_layers_neurons = configs['model']['all_layers_neurons']
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

        self.q = np.unique(np.array(self.q))
        if 0.5 not in self.q:
            self.q = np.sort(np.append(0.5, self.q))
        self.n_outputs_lastlayer = 2 if self.loss_prob == 'parametric' else len(
            self.q)

        self.metrics = configs['model']['metrics']
        self.mha_head = configs['model']['mha_head']

        self.configs = configs
        self.rnn_type = configs['rnn_units']['rnn_type']
        self.dec_attn_mask = configs['model']['dec_attn_mask']

        self.model_type = configs['model']['model_type']
        self.fit_type = configs['training']['fit_type']
        self.seq_len = configs['training']['seq_len']
        self.if_save_model_image = configs['model']['if_model_image']
        self.if_model_summary = configs['model']['if_model_summary']

        # model structure
        self.IF_GLU = configs['model']['IF_GLU']
        self.IF_ADDNORM = configs['model']['IF_ADDNORM']
        self.IFFFN = configs['model']['IFFFN']
        self.IFRNN1 = configs['model']['IFRNN_input']
        self.IFRNN2 = configs['model']['IFRNN_output']
        self.IFSELF_MHA = configs['model']['IFSELF_MHA']
        self.IFCASUAL_MHA = configs['model']['IFCASUAL_MHA']
        self.IFCROSS_MHA = configs['model']['IFCROSS_MHA']

        self.save_models_dir = configs['model']['save_models_dir']
        self.save_results_dir = configs['model']['save_results_dir']
        self.save_training_history_file = os.path.join(
            self.save_results_dir, '%s.csv' % (current_dt))
        self.save_training_history = os.path.join(
            self.save_results_dir, '%s_history.png' % (current_dt))
        self.save_hf5_name = os.path.join(
            self.save_models_dir, '%s.h5' % (current_dt))
        self.save_modelimage_name = os.path.join(
            self.save_models_dir, '%s_modelimage.png' % (current_dt))
        self.save_modelsummary_name = os.path.join(
            self.save_models_dir, '%s_modelsummary.txt' % (current_dt))

    def CVRMSE_Q50_prob_nonparametric(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred[:, :, :, 3] - y_true)))/(K.mean(y_true))

    def CVRMSE_Q50_prob_parametric(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred[:, :, :, 0] - y_true)))/(K.mean(y_true))

    def CVRMSE_Q50_nonprob(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))/(K.mean(y_true))

    def postbuild_model(self, model) -> Tuple[float, int]:

        self.model = model
        if self.optimizer == 'Adam':
            optimizer_model = tf.keras.optimizers.Adam(
                learning_rate=self.Adam_lr,
                beta_1=self.Adam_b1,
                beta_2=self.Adam_b2,
                epsilon=self.Adam_epsi)

        if self.optimizer == 'SGD':
            optimizer_model = tf.keras.optimizers.SGD(
                learning_rate=self.SGD_lr,
                momentum=self.SGD_mom)

        if self.model_type_prob == 'prob':

            if self.loss_prob == "parametric":  # maximum likelihood estimation is applied to estimate mean and stddev
                self.model.compile(loss=parametric_loss,
                                   optimizer=optimizer_model,
                                   metrics=[self.CVRMSE_Q50_prob_parametric],
                                   run_eagerly=True)

            elif self.loss_prob == "nonparametric":  # non parametric approach where quantiles are predicted
                self.model.compile(loss=lambda y_true, y_pred: nonparametric_loss(y_true, y_pred, self.q),
                                   optimizer=optimizer_model,
                                   metrics=[
                                       self.CVRMSE_Q50_prob_nonparametric],
                                   run_eagerly=True)

        elif self.model_type_prob == 'nonprob':
            self.metrics = self.metrics.copy()
            self.metrics.append(self.CVRMSE_Q50_nonprob)
            self.model.compile(loss=self.loss_func,
                               optimizer=optimizer_model,
                               metrics=self.metrics,
                               run_eagerly=True)

        if self.if_model_summary:
            self.model.summary()

        self.trainable_count = count_params(model.trainable_weights)

        self.GET_MODEL_SIZE_GB = get_model_memory_usage(
            self.batch_size, self.model)
        print('Trainable parameters in the model : %d' % self.trainable_count)
        with open(self.save_modelsummary_name, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        if self.if_save_model_image:
            print('Saving model as %s' % self.save_modelimage_name)
            plot_model(self.model, to_file=self.save_modelimage_name,
                       show_shapes=True, show_layer_names=True, dpi=600, expand_nested=True)
        print("[ModelClass] Building postmodel DONE!! model can be used as self.model")


def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p)
                             for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p)
                                 for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * \
        (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + \
        internal_model_mem_count
    print('Memory usage for model: {} GB'.format(gbytes))
    return gbytes


"""
Extra layers for the models
"""


def soft_relu(x):
    """
    Soft ReLU activation function, used for ensuring the positivity of the standard deviation of the Normal distribution
    when using the parameteric loss function. See Section 3.2.2 in the DeepTCN paper.
    """
    return tf.math.log(1.0 + tf.math.exp(x))


class get_causal_attention_mask():
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


def glu_with_addnorm(configs, x_old, x):
    IF_GLU = configs['model']['IF_GLU']
    IF_ADDNORM = configs['model']['IF_ADDNORM']
    all_layers_neurons = configs['model']['all_layers_neurons']
    all_layers_dropout = configs['model']['all_layers_dropout']

    if IF_GLU:
        x = GLU_layer(all_layers_neurons,
                      all_layers_dropout,
                      activation=None)(x)
    if IF_ADDNORM:
        x = ADD_NORM(all_layers_neurons)(x_old, x)
    return x


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

        x, _ = tf.keras.layers.Multiply()(
            [activation_layer, gated_layer]), gated_layer

        return x


class ADD_NORM():
    def __init__(self, n_cell):

        self.n_cell = n_cell

    def __call__(self, x_input_old, x_input):
        x_input = tf.keras.layers.Dense(self.n_cell)(x_input)
        x = [x_input, x_input_old]
        tmp = tf.keras.layers.Add()(x)
        tmp = tf.keras.layers.LayerNormalization()(tmp)
        return tmp


class ADD_NORM_simple():
    def __init__(self):
        pass

    def __call__(self, x):
        # x = [x_input, x_input_old]
        tmp = tf.keras.layers.Add()(x)
        tmp = tf.keras.layers.LayerNormalization()(tmp)
        return tmp


class ADD_NORM_list():
    def __init__(self):

        self.add_l = tf.keras.layers.Add()
        self.norm_l = tf.keras.layers.LayerNormalization()

    def __call__(self, x1, x2):
        all_x = []
        len_x = len(x1)
        for i in range(len_x):
            a = x1[i]
            b = x2[i]
            x = self.add_l([a, b])
            x = self.norm_l(x)
            all_x.append(x)
        return all_x


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


class rnn_unit():
    def __init__(self, configs, rnn_location, num):

        self.all_layers_dropout = configs['model']['all_layers_dropout']
        self.rnn_type = configs['rnn_units']['rnn_type']
        self.input_enc_rnn_depth = configs['rnn_units']['input_enc_rnn_depth']
        self.input_enc_rnn_bi = configs['rnn_units']['input_enc_rnn_bi']
        self.all_layers_neurons = configs['model']['all_layers_neurons']
        self.all_layers_neurons_rnn = int(
            self.all_layers_neurons/self.input_enc_rnn_depth)
        self.all_layers_neurons_rnn = 8 * int(self.all_layers_neurons_rnn/8)
        self.all_layers_dropout = configs['model']['all_layers_dropout']
        self.rnn_location = rnn_location
        self.configs = configs
        self.num = num

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

    def __call__(self, input_to_layers, init_states=None):
        if self.input_enc_rnn_depth == 1:
            return self.single_rnn_layer(x_input=input_to_layers, init_states=init_states, mid_layer=False, layername_prefix='Only_')
        else:
            x = self.single_rnn_layer(x_input=input_to_layers, init_states=init_states,
                                      mid_layer=True, layername_prefix='First_')  # change
            x = glu_with_addnorm(self.configs, input_to_layers, x)
            for i in range(0, self.input_enc_rnn_depth-2):
                x = self.single_rnn_layer(
                    x_input=x, init_states=init_states, mid_layer=True, layername_prefix='Mid_%s_' % (i+1))
                x = glu_with_addnorm(self.configs, input_to_layers, x)

            return self.single_rnn_layer(x_input=x, init_states=init_states, mid_layer=False, layername_prefix='Last_')


class Input_encoder_MHA_RNN():
    def __init__(self, configs, location, num):
        self.configs = configs
        self.location = location
        self.num = num
        if self.location == "encoder_past":
            self.IF_FFN = configs['model']['IFFFN']
            self.IF_RNN = configs['model']['IFRNN_input']
            self.IF_MHA = configs['model']['IFSELF_MHA']
            self.IF_MASK = 0
        elif self.location == "encoder_future":
            self.IF_FFN = configs['model']['IFFFN']
            self.IF_RNN = configs['model']['IFRNN_output']
            self.IF_MHA = configs['model']['IFCASUAL_MHA']
            self.IF_MASK = configs['model']['dec_attn_mask']

        self.future_data_col = configs['data']['future_data_col']
        self.n_past = configs['data']['n_past']
        self.n_features_input = configs['data']['n_features_input']
        self.all_layers_neurons = configs['model']['all_layers_neurons']
        self.all_layers_dropout = configs['model']['all_layers_dropout']
        self.mha_head = configs['model']['mha_head']
        self.n_future = configs['data']['n_future']
        self.n_features_output = configs['data']['n_features_output']
        self.MHA_RNN = configs['model']['MHA_RNN']

    def __call__(self, input, init_states=None):
        # encoder multi head attention with or without mask

        encoder_past_inputs1 = tf.keras.layers.Dense(
            self.all_layers_neurons)(input)
        encoder_past_inputs1 = tf.keras.layers.Dropout(
            self.all_layers_dropout/5)(encoder_past_inputs1)
        output_cell = encoder_past_inputs1

        if self.MHA_RNN == 0:
            input_cell = output_cell
            # encoder BiLSTM
            if self.IF_RNN:
                encoder1 = rnn_unit(self.configs,
                                    rnn_location=self.location,
                                    num=self.num)
                encoder_outputs1 = encoder1(
                    input_cell,
                    init_states=init_states)
                output_cell = encoder_outputs1[0]
                encoder_outputs1_allstates = encoder_outputs1[1:]
                output_cell = glu_with_addnorm(
                    self.configs,
                    input_cell,
                    output_cell)

                output_states = encoder_outputs1_allstates
            else:
                output_cell = input_cell
                output_states = None

        input_cell = output_cell
        if self.IF_MHA:
            encoder_mha = tf.keras.layers.MultiHeadAttention(
                num_heads=self.mha_head,
                key_dim=self.n_features_input,
                value_dim=self.n_features_input,
                name=self.location+'_'+str(self.num) + '_selfMHA')
            if self.IF_MASK:
                causal_mask = get_causal_attention_mask()(input_cell)
                output_cell = encoder_mha(query=input_cell,
                                          key=input_cell,
                                          value=input_cell,
                                          attention_mask=causal_mask,
                                          training=True)
            else:
                output_cell = encoder_mha(query=input_cell,
                                          key=input_cell,
                                          value=input_cell,
                                          training=True)
            output_cell = glu_with_addnorm(
                self.configs, input_cell, output_cell)
        else:
            output_cell = input_cell

        input_cell = output_cell
        # encoder feed forward network
        if self.IF_FFN and self.IF_MHA:
            output_cell = FeedForward(
                self.all_layers_neurons, self.all_layers_dropout/2)(input_cell)
            output_cell = glu_with_addnorm(
                self.configs,
                input_cell,
                output_cell)
        else:
            output_cell = input_cell

        if self.MHA_RNN == 1:
            input_cell = output_cell
            # encoder BiLSTM
            if self.IF_RNN:
                encoder1 = rnn_unit(self.configs,
                                    rnn_location=self.location,
                                    num=self.num)
                encoder_outputs1 = encoder1(
                    input_cell,
                    init_states=init_states)
                output_cell = encoder_outputs1[0]
                encoder_outputs1_allstates = encoder_outputs1[1:]
                output_cell = glu_with_addnorm(
                    self.configs,
                    input_cell,
                    output_cell)

                output_states = encoder_outputs1_allstates
            else:
                output_cell = input_cell
                output_states = None

        output = output_cell
        return output, output_states


class Output_decoder_crossMHA_RNN():
    def __init__(self, configs, location, num):
        self.configs = configs
        self.location = location
        self.num = num
        if self.location == "encoder_past":
            self.IF_FFN = configs['model']['IFFFN']
            self.IF_RNN = configs['model']['IFRNN_input']
            self.IF_MHA = configs['model']['IFSELF_MHA']
            self.IF_MASK = 0
        elif self.location == "encoder_future":
            self.IF_FFN = configs['model']['IFFFN']
            self.IF_RNN = configs['model']['IFRNN_output']
            self.IF_MHA = configs['model']['IFCASUAL_MHA']
            self.IF_MASK = configs['model']['dec_attn_mask']
        elif self.location == "decoder_future":
            self.IF_FFN = configs['model']['IFFFN']
            self.IF_RNN = configs['model']['IFRNN_output']
            self.IF_MHA = configs['model']['IFCROSS_MHA']
            self.IF_MASK = 0

        self.future_data_col = configs['data']['future_data_col']
        self.n_past = configs['data']['n_past']
        self.n_features_input = configs['data']['n_features_input']
        self.all_layers_neurons = configs['model']['all_layers_neurons']
        self.all_layers_dropout = configs['model']['all_layers_dropout']
        self.mha_head = configs['model']['mha_head']
        self.n_future = configs['data']['n_future']
        self.n_features_output = configs['data']['n_features_output']

    def __call__(self, input_qk, input_v, init_states=None):
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
            decoder_attn_output = glu_with_addnorm(
                self.configs, input_qk, decoder_attn_output)
        else:
            decoder_attn_output = input_qk

        # decoder feed forward network
        if self.IF_FFN and self.IF_MHA:
            decoder_ffn_output = FeedForward(
                self.all_layers_neurons, self.all_layers_dropout/2)(decoder_attn_output)
            decoder_ffn_output = glu_with_addnorm(
                self.configs, decoder_attn_output, decoder_ffn_output)
        else:
            decoder_ffn_output = decoder_attn_output

        # decoder BiLSTM
        if self.IF_RNN:
            decoder1 = rnn_unit(
                self.configs,
                rnn_location=self.location,
                num=self.num)
            decoder_outputs1 = decoder1(
                decoder_ffn_output,
                init_states=init_states)
            decoder_outputs1 = glu_with_addnorm(
                self.configs, decoder_ffn_output, decoder_outputs1[0])
        else:
            decoder_outputs1 = decoder_ffn_output

        # decoder_future output
        decoder_outputs2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.n_features_output))(decoder_outputs1)

        output = decoder_outputs2

        return output


def nonparametric_loss(y_true, y_pred, q):
    '''
    Nonparametric loss function, see Section 3.2.1 in the DeepTCN paper.

    # Parameters:
    y_true: tf.Tensor.
            Actual values of target time series, a tensor with shape (n_samples, n_forecast, n_targets) where n_samples is
            the batch size, n_forecast is the decoder length and n_targets is the number of target time series.

    y_pred: tf.Tensor.
            Predicted quantiles of target time series, a tensor with shape (n_samples, n_forecast, n_targets, n_quantiles)
            where n_samples is the batch size, n_forecast is the decoder length, n_targets is the number of target time
            series and n_quantiles is the number of quantiles.

    q: tf.Tensor.
            Quantiles, a 1-dimensional tensor with length equal to the number of quantiles.

    # Returns:
    tf.Tensor.
            Loss value, a scalar tensor.
    '''

    y_true = tf.cast(tf.expand_dims(y_true, axis=3), dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    q = tf.cast(tf.reshape(q, shape=(1, len(q))), dtype=tf.float32)
    e = tf.subtract(y_true, y_pred)

    L = tf.multiply(q, tf.maximum(0.0, e)) + \
        tf.multiply(1.0 - q, tf.maximum(0.0, - e))

    return tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(L, axis=-1), axis=-1))


def parametric_loss(y_true, params):
    '''
    Parametric loss function, see Section 3.2.2 in the DeepTCN paper.

    # Parameters:
    y_true: tf.Tensor.
            Actual values of target time series, a tensor with shape (n_samples, n_forecast, n_targets) where n_samples is
            the batch size, n_forecast is the decoder length and n_targets is the number of target time series.

    params: tf.Tensor.
            Predicted means and standard deviations of target time series, a tensor with shape (n_samples, n_forecast,
            n_targets, 2) where n_samples is the batch size, n_forecast is the decoder length and n_targets is the
            number of target time series.

    # Returns:
    tf.Tensor.
            Loss value, a scalar tensor.
    '''

    y_true = tf.cast(y_true, dtype=tf.float32)
    params = tf.cast(params, dtype=tf.float32)

    mu = params[:, :, :, 0]
    sigma = params[:, :, :, 1]

    L = 0.5 * tf.math.log(2 * np.pi) + tf.math.log(sigma) + \
        tf.math.divide(tf.math.pow(y_true - mu, 2), 2 * tf.math.pow(sigma, 2))

    return tf.experimental.numpy.nanmean(tf.experimental.numpy.nanmean(tf.experimental.numpy.nansum(L, axis=-1), axis=-1))


class Timer():
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        timetaken = end_dt - self.start_dt
        print('Time taken: %s' % (timetaken))
        # print("Current memory (MBs)",get_memory_info('GPU:0')['current'] / 1048576)
        print("")
        return timetaken
