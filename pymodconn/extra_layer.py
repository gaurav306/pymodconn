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

