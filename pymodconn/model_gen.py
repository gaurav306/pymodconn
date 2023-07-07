import pymodconn.model_gen_utils as Model_utils
from pymodconn.Encoder_class_layer import Encoder_class
from pymodconn.Decoder_class_layer import Decoder_class
from pymodconn.utils_layers import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.models import Model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class Model_Gen():
    def __init__(self, cfg, current_dt):
        self.cfg = cfg
        self.current_dt = current_dt

        if cfg['if_seed']:
            np.random.seed(cfg['seed'])
            tf.random.set_seed(cfg['seed'])
            tf.keras.utils.set_random_seed(cfg['seed'])
            tf.config.experimental.enable_op_determinism()
        
        self.n_past = cfg['n_past']
        self.n_future = cfg['n_future']
        self.known_past_features = cfg['known_past_features']
        self.unknown_future_features = cfg['unknown_future_features']
        self.known_future_features = cfg['known_future_features']

        self.all_layers_neurons = cfg['all_layers_neurons']
        self.all_layers_dropout = cfg['all_layers_dropout']
        
        self.model_type_prob = cfg['model_type_prob']
        self.loss_prob = cfg['loss_prob']
        self.q = cfg['quantiles']
        self.n_outputs_lastlayer = 2 if self.loss_prob == 'parametric' else len(self.q)

        self.save_models_dir = cfg['save_models_dir']
        if not os.path.exists(cfg['save_models_dir']):
            os.makedirs(cfg['save_models_dir'])

    def build_model(self):
        """
        The build_model() method is responsible for constructing the architecture of your deep learning model according to the 
        specifications outlined in the provided configuration file. The model is built using the Keras API of TensorFlow, 
        which allows for the flexible construction of a variety of neural network models.

        This method first creates an input layer for the encoder, encoder_inputs, and passes this to an instance of Encoder_class. 
        The encoder processes the input data and returns two outputs: encoder_outputs_seq and encoder_outputs_allstates. 
        These outputs represent the sequence of hidden states and the final state of the encoder, respectively.

        Next, the method enters a loop where it creates a number of decoders based on the control_future_cells parameter, 
        which can range from 1 to 6. For each iteration, it creates a new input layer decoder_{i}_inputs and passes it, 
        along with the encoder outputs, to an instance of Decoder_class. The output of each decoder is added to decoder_outputs_list.

        After all decoders have been processed, the method concatenates their outputs using a custom MERGE_LIST layer.

        The method then determines whether the model uses a probabilistic or non-probabilistic forecast approach based 
        on the model_type_prob parameter. Depending on the approach, it applies different operations to the concatenated 
        decoder outputs to produce the final model output, decoder_outputs4.

        Finally, it creates a Keras Model instance with the encoder inputs and decoder inputs as inputs and decoder_outputs4 
        as the output. The Build_utils method is called to post-process the model, where it selects the appropriate 
        loss function, optimizer, and metrics based on the user input from the configuration file.
        """
        timer = Model_utils.Timer()
        timer.start()
        print('[pymodconn] Model Compiling.....')

        # input for encoder_past
        encoder_inputs = tf.keras.layers.Input(
            shape=(self.n_past, self.known_past_features), name='encoder_past_inputs')
        
        encoder_outputs_seq, encoder_outputs_allstates = Encoder_class(
            self.cfg, str(1))(encoder_inputs, init_states=None)

        '''
        decoder_outputs_list = []
        
        for i in range(1, self.control_future_cells+1):
            locals()[f"decoder_{i}_inputs"] = tf.keras.layers.Input(shape=(self.n_future, self.known_future_features), name=f"decoder_{i}_inputs")
            locals()[f"decoder_{i}_outputs"] = Decoder_class(self.cfg, str(i))(locals()[f"decoder_{i}_inputs"],
                                                                                encoder_outputs_seq,
                                                                                encoder_states=encoder_outputs_allstates)
            decoder_outputs_list.append(locals()[f"decoder_{i}_outputs"])
        
        decoder_outputs_all = MERGE_LIST(self.unknown_future_features)(decoder_outputs_list)
        '''
        decoder_inputs = tf.keras.layers.Input(
            shape=(self.n_future, self.known_future_features), name=f"decoder_inputs")

        decoder_outputs = Decoder_class(self.cfg, '1')(decoder_inputs,
                                                        encoder_outputs_seq,
                                                        encoder_states=encoder_outputs_allstates)        

        # If or not using the probabilistic loss, reshape the output to match the shape required by the loss function.
        if self.model_type_prob == 'prob':
            decoder_outputs3 = tf.keras.layers.Dense(
                units=self.unknown_future_features * self.n_outputs_lastlayer)(decoder_outputs)  # this one
            # Reshape the encoder output to match the shape required by the loss function.
            decoder_outputs4 = tf.keras.layers.Reshape(target_shape=(
                self.n_future, self.unknown_future_features, self.n_outputs_lastlayer))(decoder_outputs3)
            # If using the parametric loss, apply the soft ReLU activation to ensure a positive standard deviation.
            if self.loss_prob == 'parametric':
                decoder_outputs4 = tf.keras.layers.Lambda(function=lambda x: tf.stack(
                    [x[:, :, :, 0], soft_relu(x[:, :, :, 1])], axis=-1))(decoder_outputs4)
        elif self.model_type_prob == 'nonprob':
            decoder_outputs4 = tf.keras.layers.Lambda(
                lambda x: tf.clip_by_value(x, -1, 1))(decoder_outputs)
        else:
            raise ValueError(
                'model_type_prob should be either prob or nonprob')

        self.model = Model(
            [encoder_inputs, decoder_inputs], decoder_outputs4)

        Model_utils.Build_utils(
            self.cfg, self.current_dt).postbuild_model(self.model)

    def load_model(self, filepath):
        print('[pymodconn_tfp] Loading model from file %s' % filepath)
        # self.model = load_model(filepath)
        self.model.load_weights(filepath)
        # https://stackoverflow.com/a/69663259/6510598

    def forget_model(self):
        del self.model
        K.clear_session()
        print("Everything forgoten....maybe")
