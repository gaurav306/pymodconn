from keras.utils.layer_utils import count_params
import os
from typing import *

import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import datetime as dt
K = tf.keras.backend


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

class Build_utils():
    """
    This class is used to compile, make sumamry and save as png.
    """

    def __init__(self, cfg, current_dt):
        if cfg['if_seed']:
            np.random.seed(cfg['seed'])
            tf.random.set_seed(cfg['seed'])
        self.epochs = cfg['epochs']
        self.batch_size = cfg['batch_size']
        self.future_data_col = cfg['future_data_col']
        self.n_past = cfg['n_past']
        self.n_features_input = cfg['n_features_input']
        self.all_layers_neurons = cfg['all_layers_neurons']
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

        self.q = np.unique(np.array(self.q))
        if 0.5 not in self.q:
            self.q = np.sort(np.append(0.5, self.q))
        self.n_outputs_lastlayer = 2 if self.loss_prob == 'parametric' else len(
            self.q)

        self.metrics = cfg['metrics']
        self.mha_head = cfg['mha_head']

        self.cfg = cfg
        self.rnn_type = cfg['rnn_type']
        self.dec_attn_mask = cfg['dec_attn_mask']

        self.model_type = cfg['model_type']
        self.fit_type = cfg['fit_type']
        self.seq_len = cfg['seq_len']
        self.if_save_model_image = cfg['if_model_image']
        self.if_model_summary = cfg['if_model_summary']

        # model structure
        self.IFRNN1 = cfg['IFRNN_input']
        self.IFRNN2 = cfg['IFRNN_output']
        self.IFSELF_MHA = cfg['IFSELF_MHA']
        self.IFCASUAL_MHA = cfg['IFCASUAL_MHA']
        self.IFCROSS_MHA = cfg['IFCROSS_MHA']

        self.save_models_dir = cfg['save_models_dir']
        self.save_results_dir = cfg['save_results_dir']
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
        '''
        self.GET_MODEL_SIZE_GB = get_model_memory_usage(
            self.batch_size, self.model)
        print('Trainable parameters in the model : %d' % self.trainable_count)
        '''
        with open(self.save_modelsummary_name, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'), 
                               line_length = 250, 
                               expand_nested=True,
                               show_trainable=True)
        '''
        with open(self.save_modelsummary_name, 'a') as f:
            f.write('_' * 25 + '\n')
            f.write('Model size in GB : %f' % self.GET_MODEL_SIZE_GB)
        '''
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


