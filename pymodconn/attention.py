'''
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, Permute, Dot, Activation
from tensorflow.keras.models import Model

# Define input and output dimensions
input_dim = 1
output_dim = 1

# Define the input sequence
encoder_inputs = Input(shape=(672, input_dim))

# Define the encoder GRU
encoder_gru = GRU(units=256, return_sequences=True, return_state=True)
encoder_outputs, state_h = encoder_gru(encoder_inputs)

# Define the decoder input sequence
decoder_inputs = Input(shape=(96, input_dim))

# Define the first decoder GRU
decoder_gru1 = GRU(units=256, return_sequences=True, return_state=True)
decoder_outputs1, state_h1 = decoder_gru1(decoder_inputs, initial_state=state_h)

# Define the second decoder GRU with teacher forcing
decoder_gru2 = GRU(units=256, return_sequences=True, return_state=True)
decoder_outputs2, state_h2 = decoder_gru2(decoder_outputs1, initial_state=state_h)

# Define the attention mechanism
attention = Dot(axes=[2, 2])([decoder_outputs2, encoder_outputs])
attention = Activation('softmax')(attention)
attention = Dot(axes=[2, 1])([attention, encoder_outputs])
attention = Permute((2, 1))(attention)

# Concatenate the attention output and the decoder output
decoder_combined_context = Concatenate(axis=-1)([decoder_outputs2, attention])

# Define the output dense layer
output_layer = Dense(units=output_dim)

# Connect the decoder GRU and the output dense layer
outputs = output_layer(decoder_combined_context)

# Define the Seq2Seq model
model = Model([encoder_inputs, decoder_inputs], outputs)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Print the model summary
model.summary()



'''



import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, Layer
from tensorflow.keras.models import Model

# Constants
past_steps = 672
future_known_steps = 96
future_unknown_steps = 96
num_features = 10  # Number of input features

class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# Encoder
encoder_inputs = Input(shape=(past_steps, num_features), name='encoder_inputs')
encoder_gru = GRU(128, return_sequences=True, return_state=True, name='encoder_gru')
encoder_outputs, encoder_state = encoder_gru(encoder_inputs)

# Decoder with future known data
decoder_known_inputs = Input(shape=(future_known_steps, num_features), name='decoder_known_inputs')
decoder_gru1 = GRU(128, return_sequences=True, return_state=True, name='decoder_gru1')
decoder_outputs1, decoder_state1 = decoder_gru1(decoder_known_inputs, initial_state=encoder_state)

# Concatenate encoder_state and decoder_state1
combined_state = Concatenate(axis=-1, name='state_concat_layer')([encoder_state, decoder_state1])

# Second Decoder with future unknown data
decoder_gru2 = GRU(256, return_sequences=True, return_state=True, name='decoder_gru2')  # Note the doubled units due to the concatenated states
decoder_outputs2, decoder_state2 = decoder_gru2(decoder_outputs1, initial_state=combined_state)

# Bahdanau attention
attention_layer = BahdanauAttention(128)
context_vector, attention_weights = attention_layer(decoder_state2, encoder_outputs)

# Concatenate context vector and decoder_outputs2
decoder_input3 = Concatenate(axis=-1, name='concat_layer')([context_vector, decoder_outputs2])

# Output layer
output_layer = Dense(num_features, activation='linear', name='output_layer')
outputs = output_layer(decoder_input3)

# Model
model = Model(inputs=[encoder_inputs, decoder_known_inputs], outputs=outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()




import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, Layer
from tensorflow.keras.models import Model

# Constants
past_steps = 672
future_known_steps = 96
future_unknown_steps = 96
num_features = 10  # Number of input features

class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# Encoder
encoder_inputs = Input(shape=(past_steps, num_features), name='encoder_inputs')
encoder_gru = GRU(128, return_sequences=True, return_state=True, name='encoder_gru')
encoder_outputs, encoder_state = encoder_gru(encoder_inputs)

# Decoder with future known data
decoder_known_inputs = Input(shape=(future_known_steps, num_features), name='decoder_known_inputs')
decoder_gru1 = GRU(128, return_sequences=True, return_state=True, name='decoder_gru1')
decoder_outputs1, decoder_state1 = decoder_gru1(decoder_known_inputs, initial_state=encoder_state)

# Concatenate encoder_state and decoder_state1
combined_state = Concatenate(axis=-1, name='state_concat_layer')([encoder_state, decoder_state1])

# Second Decoder with future unknown data
decoder_gru2 = GRU(256, return_sequences=True, return_state=True, name='decoder_gru2')  # Note the doubled units due to the concatenated states
decoder_outputs2, decoder_state2 = decoder_gru2(decoder_outputs1, initial_state=combined_state)

# Bahdanau attention
attention_layer = BahdanauAttention(128)
context_vector, attention_weights = attention_layer(decoder_outputs1, encoder_outputs)  # Modified input from decoder_state2 to decoder_outputs1

# Concatenate context vector and decoder_outputs2
decoder_input3 = Concatenate(axis=-1, name='concat_layer')([context_vector, decoder_outputs2])

# Output layer
output_layer = Dense(num_features, activation='linear', name='output_layer')
outputs = output_layer(decoder_input3)

# Model
model = Model(inputs=[encoder_inputs, decoder_known_inputs], outputs=outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()













import tensorflow as tf
# pylint: disable=E0611,E0401
from tensorflow.keras import layers, models, Input
from tcn import TCN

past_steps = 672
future_known_steps = 96
n_features = 3  # Number of features in the multivariate input

# Encoder
encoder_inputs = Input(shape=(past_steps, n_features))
encoder_tcn = TCN(return_sequences=True)
encoder_outputs = encoder_tcn(encoder_inputs)

# Future known data
future_known_inputs = Input(shape=(future_known_steps, n_features))
future_known_tcn = TCN(return_sequences=True)
future_known_outputs = future_known_tcn(future_known_inputs)

# Attention
query_value_attention_seq = layers.AdditiveAttention()([future_known_outputs, encoder_outputs])
query_value_attention_seq = layers.GlobalAveragePooling1D()(query_value_attention_seq)

# Concatenate attended output and future_known_outputs
concatenated_outputs = layers.Concatenate()([query_value_attention_seq, layers.GlobalAveragePooling1D()(future_known_outputs)])

# Expand dimensions for the third TCN input
concatenated_outputs_expanded = layers.Reshape((1, -1))(concatenated_outputs)

# Third TCN (decoder)
decoder_tcn = TCN(return_sequences=True)
decoder_outputs = decoder_tcn(concatenated_outputs_expanded)

# Final output
final_output = layers.GlobalAveragePooling1D()(decoder_outputs)
final_output = layers.Dense(future_known_steps)(final_output)

# Build and compile the model
model = models.Model(inputs=[encoder_inputs, future_known_inputs], outputs=final_output)
model.compile(optimizer='adam', loss='mse')
model.summary()








import tensorflow as tf
from tensorflow.keras import layers, models

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :],
                             d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

# Constants
past_steps = 672
future_known_steps = 96
num_features = 10
d_model = 128

# Inputs
past_data_input = layers.Input(shape=(past_steps, num_features), name='past_data')
future_known_data_input = layers.Input(shape=(future_known_steps, num_features), name='future_known_data')

# Positional encodings
past_data_pos_encoding = positional_encoding(past_steps, d_model)
future_known_data_pos_encoding = positional_encoding(future_known_steps, d_model)

past_data = layers.Add()([past_data_input, past_data_pos_encoding[:past_steps]])
future_known_data = layers.Add()([future_known_data_input, future_known_data_pos_encoding[:future_known_steps]])

# Transformer architecture (simplified for brevity)
# ... (Add the rest of the Transformer architecture, including multi-head attention, feedforward layers, and residual connections)

# Model
model = models.Model(inputs=[past_data_input, future_known_data_input], outputs=outputs)
