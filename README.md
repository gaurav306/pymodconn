# _pymodconn_ : A Python package for developing modular sequence to sequence control-oriented deep neural networks

![alt text](https://github.com/gaurav306/pymodconn/blob/master/Readme_images/FIG1.png)

A control-oriented deep neural network (CONN) is a deep neural network designed to predict the future behaviour and response of complex dynamic systems, taking into account *Known past data* and *Known future data*. 
- *Known past data* includes historical information such as weather data, time information, system dynamics, and control inputs, which are used to train the CONN to understand the relationships between these variables and their effects on the system. 
- *Known future data* encompasses forecasted weather data, time information, schedules, and most importantly targeted control inputs, representing the expected conditions and desired actions that the system will be subject to in the future. 

The CONN leverages this information to predict future, i.e., *Unknown future data*, referring to the system dynamics or states that are not yet known. This prediction enables the control system to anticipate changes in the environment and adapt its actions, accordingly, ensuring optimal performance and robustness in spite of uncertainties.
# Contents

* [Installation](#installation)
* [Requirements](#requirements)
* [Usage](#usage)
* [Configuration file details](#configuration-file-details)
* [Full implementation example - Model developement and training using Keras](#full-implementation-example---model-developement-and-training-using-keras)


# Installation
*pymodconn* can be installed using [pip](https://pip.pypa.io/en/stable/), a tool for installing Python packages. To do it, run the following command:
```
pip install pymodconn
```

# Requirements
*pymodconn* requires Python >= 3.9.16 or later to run. Note that you should have also the following packages installed in your system:
- jsonschema==4.17.3
- numpy==1.24.3
- PyYAML==6.0
- ruamel.base==1.0.0
- tensorflow==2.12.0

# Usage
## Step 1: Loading the configurations as a dictionary
The users can design their CONN using a text-based configuration file. The details of this configuration file is discussed in next section ([Configuration file details](#configuration-file-details)). The configuration file can be loaded as a dictionary for use using [get_configs()](https://github.com/gaurav306/pymodconn/blob/c1daed967501524311dbbe085b986b7f3c356d45/pymodconn/configs/configs_init.py#L24C7-L24C7). User can furthur change contents of configuration dictionary before using it to develop Keras model
```python
from pymodconn.configs.configs_init import get_configs

MODEL_CONFIG_FILENAME = 'default_config.yaml'
configs = get_configs(MODEL_CONFIG_FILENAME)
configs['n_past'] = 25 #Number of past observations timesteps to be considered for prediction
configs['n_future'] = 10 #Number of future predictions timesteps to be made
configs['known_past_features'] = 2 #Number of features in the past observation window data
configs['known_future_features'] = 3 #Number of features in the future prediction window data
configs['unknown_future_features'] = 2 #Number of features in the future prediction window data 
```
## Step 2: Develop Keras model using configurations dictionary
To differentiate model details the user can choose to use a unique indentation. For example this unique identifier string (ident) is created by appending the current time string to a predefined text. This identifier is used to differentiate between multiple runs or cases and for generating unique filenames when saving the model and prediction results.

Finally to develop the Keras model, the [Model_Gen(configs, ident)](https://github.com/gaurav306/pymodconn/blob/c1daed967501524311dbbe085b986b7f3c356d45/pymodconn/model_gen.py#L14C11-L14C11) class is instantiated with the loaded configurations and the current run's identifier. The [build_model()](https://github.com/gaurav306/pymodconn/blob/c1daed967501524311dbbe085b986b7f3c356d45/pymodconn/model_gen.py#L43) method is then called to create the Keras model object.

The model returned from **build_model()** can be used similarly to a Keras-based model. The user can run *keras.model.fit()*, *keras.model.train_on_batch()*, *keras.model.evaluate()*, or *keras.model.save()* as needed. A  [full implementation example](#full-implementation-example---model-developement-and-training-using-keras) is shown later.

```python
from pymodconn import Model_Gen

# 'ident' is a string used to ensure unique file and prediction case names
ident = 'test_'
current_run_dt = ident + str(dt.datetime.now().strftime('%d.%m-%H.%M.%S'))

# Initialize and build the model using
model_class = Model_Gen(configs, ident)
model_class.build_model()
model_class.model.fit(....)
model_class.model.evaluate(....)
model_class.model.save(....)
```

# Configuration file details

As discussed in previous section, users can design their CONN using a text-based .yaml file. YAML ("YAML Ain't Markup Language") is a human-readable data serialization format that is commonly used for configuration files due to its simplicity and readability. It facilitates implementation with various network architectures, promotes modularity and enables the reuse of predefined models, thus improving efficiency and reducing coding errors. The approach simplifies hyperparameter tuning, a critical aspect of neural network performance, by isolating these parameters in a configuration file.

This file can be generated by providing config_filename = None in the get_configs(config_filename) function or obtained from the package's GitHub repository [default_config.yaml](https://github.com/gaurav306/pymodconn/blob/master/pymodconn/configs/default_config.yaml).

The first option creates a template of the configuration file in the current directory of the Python program, which users can modify with the desired parameters. With the latter approach, users can download a template configuration file, along with an assortment of config files corresponding to various predefined model architectures. The users can use config files for predefined model architectures as starting point.

To avoid errors and ensure correct operation, it is recommended that users do not alter the key names or the data types of the values in the key-value pairs within the configuration file. Maintaining this structure is essential for the Python package to function correctly. Following section gives more details of YAML configuration file lin-by-line. 

## ALL BASIC SETTINGS INDEPENDENT OF MODEL ARCHITECTURE
```YAML
if_model_image: 1
if_model_summary: 0
if_seed: 0
seed: 500

model_type_prob: prob
loss_prob: nonparametric
loss: mean_squared_error
quantiles: [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]
metrics: [mape, acc]
optimizer: Adam
Adam:
  lr: 0.001
  b1: 0.9
  b2: 0.999
  epsi: 1.0e-08
SGD:
  lr: 0.001
  momentum: 0.9
save_models_dir: saved_models/

batch_size: 32

n_past: 672
n_future: 96
known_past_features: 31
unknown_future_features: 1
known_future_features: 15

all_layers_neurons: 128
all_layers_dropout: 0.4
```
- **if_model_image:** When set to 1, the package saves a visual representation of your model architecture as an image file. A 0 disables this feature.
- **if_model_summary:** Setting this to 1 enables the package to print a summary of your model architecture. Set it to 0 to disable this feature.
- **if_seed:** If this value is 1, a specific seed for random number generation is used, ensuring the reproducibility of your model. Set it to 0 to disable this.
- **seed:** If 'if_seed' is set to 1, this value specifies the seed for random number generation. Change it as needed.
- **model_type_prob:** Determines the type of model outputs. 'prob' signifies that probabilistic (interval) predictions will be generated.
- **loss_prob:** This field allows you to set the type of loss function for probabilistic prediction. 'nonparametric' and 'parametric' are the options, derived from DeepTCN.
- **loss:** This parameter sets the loss function for your model. Here, 'mean_squared_error' is used.
- **quantiles:** If using a quantile loss function for probabilistic forecasts, specify the quantiles you wish to estimate here.
- **metrics:** Here you specify the metrics that will be used to evaluate the model's performance.
- **optimizer:** This parameter allows you to set the optimizer for your model. In this example, the 'Adam' optimizer is used.
- **Adam:** This section lets you specify the hyperparameters for the Adam optimizer: learning rate (lr), beta 1 (b1), beta 2 (b2), and epsilon (epsi).
- **SGD:** Here you can specify the hyperparameters for the SGD optimizer if used: learning rate (lr) and momentum.
- **save_models_dir:** This is the directory where your trained models will be saved.
- **batch_size:** This parameter sets the number of samples to work through before the model’s internal parameters are updated.
- **n_past, n_future, known_past_features, unknown_future_features, known_future_features:** These parameters specify the number of past and future time steps, and the number of known past, known future and unknown future features, respectively.
- **all_layers_neurons:** This field specifies the number of neurons for all layers.
- **all_layers_dropout:** This parameter sets the dropout rate for all layers, a regularization technique to prevent overfitting.

![alt text](https://github.com/gaurav306/pymodconn/blob/master/Readme_images/FIG2.png)
The CONN structure includes an Encoder block processing known past data, and one or more Decoder blocks handling known future data. The Encoder block generates a sequence of encoded states reflecting the temporal structure of past data, and a final state summarizing this data. Each Decoder block receives these outputs; the encoded states serve as a temporal context, while the final encoder state initializes the Decoder block.


## ENCODER SETTINGS
This section of config file is used to list settings used to structure the *Encoder* of the model. 

```YAML
# DETAILED MODEL SETTINGS
# ENCODER SETTINGS
encoder:
  TCN_input:
    IF_TCN: 1
    IF_NONE_GLUADDNORM_ADDNORM_TCN: 1         #0: None, 1: GLUADDNORM, 2: ADDNORM
    kernel_size: 3
    nb_stacks: 2
    dilations: [1, 2, 4, 8, 16, 32]
  
  RNN_block_input:
    # settings for RNN block outside the block class
    IF_RNN: 1
    IF_NONE_GLUADDNORM_ADDNORM_block: 1         #0: None, 1: GLUADDNORM, 2: ADDNORM
    IF_GRN_block: 1                             #0: no GRN, 1: GRN
    # settings for RNN units inside the blocks
    rnn_depth: 3
    rnn_type: GRU                               # GRU, LSTM, SimpleRNN
    IF_birectionalRNN: 1                        # 0: no bidirectional, 1: bidirectional
    IF_NONE_GLUADDNORM_ADDNORM_deep: 1          #0: None, 1: GLUADDNORM, 2: ADDNORM

  self_MHA_block:
    # settings for self MHA block outside the block class
    MHA_depth: 3
    # settings for self MHA units inside the blocks
    IF_MHA: 1
    IF_GRN_block: 1                             #0: no GRN, 1: GRN
    MHA_head: 8
    IF_NONE_GLUADDNORM_ADDNORM_deep: 1          #0: None, 1: GLUADDNORM, 2: ADDNORM

```
![alt text](https://github.com/gaurav306/pymodconn/blob/master/Readme_images/FIG3.png)
- **encoder:**
- **TCN_input:** 
    - **IF_TCN:** When set to 1, a Temporal Convolutional Network (TCN) is used in the input. A 0 disables this feature.
    - **IF_NONE_GLUADDNORM_ADDNORM_TCN:** Determines the type of residual connection for the TCN layer. Use 0 for Nothing, 1 for an Add and Normalize residual connection with a GLU gate, and 2 for an Add and Normalize residual connection without a GLU gate.
    - **kernel_size:** Specifies the size of the kernel used in the TCN layer.
    - **nb_stacks:** Specifies the number of TCN stacks to be used.
    - **dilations:** Lists the dilation factors for the TCN layers.
- **RNN_block_input:**
    - **IF_RNN:** When set to 1, a Recurrent Neural Network (RNN) is used in the input. A 0 disables this feature.
    - **IF_NONE_GLUADDNORM_ADDNORM_block:** Determines the type of residual connection for the RNN block. Use 0 for Nothing, 1 for an Add and Normalize residual connection with a GLU gate, and 2 for an Add and Normalize residual connection without a GLU gate.
    - **IF_GRN_block:** Sets whether Gated Recurrent Unit (GRN) is used in the block. Set it to 0 for no GRN, and 1 for GRN.
    - **rnn_depth:** Specifies the depth of the RNN units.
    - **rnn_type:** Sets the type of RNN units used. Options are GRU, LSTM, and SimpleRNN.
    - **IF_birectionalRNN:** Specifies whether the RNN units are bidirectional. Set it to 0 for no bidirectionality, and 1 for bidirectional.
    - **IF_NONE_GLUADDNORM_ADDNORM_deep:** Determines the type of residual connection for the deeper layers. Use 0 for Nothing, 1 for an Add and Normalize residual connection with a GLU gate, and 2 for an Add and Normalize residual connection without a GLU gate.
- **self_MHA_block:**
    - **MHA_depth:** Specifies the depth of the multi-head attention (MHA) units.
    - **IF_MHA:** When set to 1, a Multi-Head Attention (MHA) unit is used in the input. A 0 disables this feature.
    - **IF_GRN_block:** Sets whether Gated Recurrent Unit (GRN) is used in the block. Set it to 0 for no GRN, and 1 for GRN.
    - **MHA_head:** Specifies the number of heads in the MHA unit.
    - **IF_NONE_GLUADDNORM_ADDNORM_deep:** Determines the type of residual connection for the deeper layers. Use 0 for Nothing, 1 for an Add and Normalize residual connection with a GLU gate, and 2 for an Add and Normalize residual connection without a GLU gate.

## DECODER SETTINGS
This section of config file is used to list settings used to structure the *Decoder* of the model. 

```YAML
# DECODER SETTINGS
decoder:

  # TCN block_decoder input settings
  TCN_input:
    IF_TCN: 1
    IF_NONE_GLUADDNORM_ADDNORM_TCN: 1         #0: None, 1: GLUADDNORM, 2: ADDNORM
    kernel_size: 3
    nb_stacks: 2
    dilations: [1, 2, 4, 8, 16, 32]

  # # RNN block_decoder input settings
  RNN_block_input:
    # settings for RNN block outside the block class
    IF_RNN: 1
    IF_NONE_GLUADDNORM_ADDNORM_block: 1         #0: None, 1: GLUADDNORM, 2: ADDNORM
    IF_GRN_block: 0                             #0: no GRN, 1: GRN
    # settings for RNN units inside the blocks
    rnn_depth: 3
    rnn_type: GRU                               # GRU, LSTM, SimpleRNN
    IF_birectionalRNN: 1                        # 0: no bidirectional, 1: bidirectional
    IF_NONE_GLUADDNORM_ADDNORM_deep: 1          #0: None, 1: GLUADDNORM, 2: ADDNORM
  
  CIT_option: 1                                 #1: Concatenation, 2: Luong or Bahdanau, 3: Self+cross MHA
  # Contextual Information Transfer - option 1 settings 
  IF_NONE_GLUADDNORM_ADDNORM_CIT_1: 1           #0: None, 1: GLUADDNORM, 2: ADDNORM
  option_1_depth : 3

  # Contextual Information Transfer - option 2 settings
  IF_NONE_GLUADDNORM_ADDNORM_CIT_2: 1           #0: None, 1: GLUADDNORM, 2: ADDNORM
  attn_type: 2                                  # 1 is Luong attention, 2 is Bahdanau attention 
  option_2_depth : 3
  
  # Contextual Information Transfer - option 3 settings
  IF_SELF_CROSS_MHA: 1
  SELF_CROSS_MHA_depth: 3

  self_MHA_block:
    # settings for self MHA units inside the blocks
    IF_MHA: 1
    IF_GRN_block: 1                             #0: no GRN, 1: GRN
    MHA_head: 8
    IF_NONE_GLUADDNORM_ADDNORM_deep: 1          #0: None, 1: GLUADDNORM, 2: ADDNORM

  cross_MHA_block:
    # settings for cross MHA units inside the blocks
    IF_MHA: 1
    IF_GRN_block: 1                             #0: no GRN, 1: GRN
    MHA_head: 8
    IF_NONE_GLUADDNORM_ADDNORM_deep: 1          #0: None, 1: GLUADDNORM, 2: ADDNORM


  # TCN block_decoder output settings
  TCN_output:
    IF_TCN: 1
    IF_NONE_GLUADDNORM_ADDNORM_TCN: 1         #0: None, 1: GLUADDNORM, 2: ADDNORM
    kernel_size: 3
    nb_stacks: 2
    dilations: [1, 2, 4, 8, 16, 32]

  # RNN block_decoder output settings
  RNN_block_output:
    # settings for RNN block outside the block class
    IF_RNN: 1
    IF_NONE_GLUADDNORM_ADDNORM_block: 1          #0: None, 1: GLUADDNORM, 2: ADDNORM
    IF_GRN_block: 1                              #0: no GRN, 1: GRN
    # settings for RNN units inside the blocks
    rnn_depth: 3
    rnn_type: GRU                                # GRU, LSTM, SimpleRNN
    IF_birectionalRNN: 1                         # 0: no bidirectional, 1: bidirectional
    IF_NONE_GLUADDNORM_ADDNORM_deep: 1           #0: None, 1: GLUADDNORM, 2: ADDNORM

  MERGE_STATES_METHOD: 4
  #-----about merging states from Encoder_states(A) 
  # and Decoder_input_states(B) for Decoder_output init_states
  # there are 8 options:
  # 1: None
  # 2: A - Dense layer
  # 3: B - Dense layer
  # 4: A+B - Concat -> Dense layer 
  # 5: A+B - Add -> Dense layer
  # 6: A+B - Add_Norm -> Dense layer
  # 7: A+B - Add
  # 8: A+B - Add_Norm
```
![alt text](https://github.com/gaurav306/pymodconn/blob/master/Readme_images/FIG4.png)
- **DECODER SETTINGS**
  - **decoder:**
    - **TCN_input:** (Refer to the previously provided TCN settings description)
    - **RNN_block_input:** (Refer to the previously provided RNN settings description)
    - **CIT_option:** Defines the method for Contextual Information Transfer. Options include: 1 for Concatenation, 2 for Luong or Bahdanau attention, and 3 for Self+cross MHA.
    - **IF_NONE_GLUADDNORM_ADDNORM_CIT_1:** Determines the type of residual connection for the first CIT option. Use 0 for Nothing, 1 for an Add and Normalize residual connection with a GLU gate, and 2 for an Add and Normalize residual connection without a GLU gate.
    - **option_1_depth:** Specifies the depth of the first CIT option.
    - **IF_NONE_GLUADDNORM_ADDNORM_CIT_2:** Determines the type of residual connection for the second CIT option. Use 0 for Nothing, 1 for an Add and Normalize residual connection with a GLU gate, and 2 for an Add and Normalize residual connection without a GLU gate.
    - **attn_type:** Specifies the type of attention mechanism to use. Option 1 is for Luong attention, and Option 2 is for Bahdanau attention.
    - **option_2_depth:** Specifies the depth of the second CIT option.
    - **IF_SELF_CROSS_MHA:** When set to 1, enables Self+cross MHA for the third CIT option. A 0 disables this feature.
    - **SELF_CROSS_MHA_depth:** Specifies the depth of the Self+cross MHA for the third CIT option.
    - **self_MHA_block:** (Refer to the previously provided self MHA block settings description)
    - **cross_MHA_block:** (Refer to the previously provided self MHA block settings description. For this setting, it applies to cross MHA units.)
    - **TCN_output:** (Refer to the previously provided TCN settings description)
    - **RNN_block_output:** (Refer to the previously provided RNN settings description)
    - **MERGE_STATES_METHOD:** Specifies the method for merging states from Encoder_states(A) and Decoder_input_states(B) for Decoder_output init_states. Options include:
      - 1: None
      - 2: A - Dense layer
      - 3: B - Dense layer
      - 4: A+B - Concat -> Dense layer 
      - 5: A+B - Add -> Dense layer
      - 6: A+B - Add_Norm -> Dense layer
      - 7: A+B - Add
      - 8: A+B - Add_Norm

<div style="border:2px solid red; padding: 10px;">
    <strong style="color: red;">To avoid errors and ensure correct operation, it is recommended that users do not alter the key names or the data types of the values in the key-value pairs within the configuration file. Maintaining this structure is essential for the Python package to function correctly.</strong>
</div>



# Full implementation example - Model developement and training using Keras

Within the same directory as the default_config.yaml file, three additional configuration files are accessible. These files represent CONNs built on three recognized time series prediction mechanisms: RNNs - [CONN_based_on_biLSTMs_with_Bahdanau_attention.yaml](https://github.com/gaurav306/pymodconn/blob/master/pymodconn/configs/CONN_based_on_biLSTMs_with_Bahdanau_attention.yaml), MHA - [CONN_based_on_MHA.yaml](https://github.com/gaurav306/pymodconn/blob/master/pymodconn/configs/CONN_based_on_MHA.yaml), and TCN - [CONN_based_on_TCN_with_Bahdanau_attention.yaml](https://github.com/gaurav306/pymodconn/blob/master/pymodconn/configs/CONN_based_on_TCN_with_Bahdanau_attention.yaml). 
An implementation example is presented below which develops and tests the three different example model configuration.
```python
import multiprocessing
import numpy as np
import sys
from pymodconn import Model_Gen
from pymodconn.configs.configs_init import get_configs
import datetime as dt

class MultiprocessingWindow():
	"""
	 Wrapper class to spawn a multiprocessing window for training and testing 
	 to avoid the issue of 'Tensorflow not releasing GPU memory'
	"""
	def __init__(self, function_to_run, its_args):
		self.function_to_run = function_to_run
		self.its_args = its_args

	def __call__(self):
		multiprocessing.set_start_method('spawn', force=True)
		print("Multiprocessing window spawned")
		p = multiprocessing.Process(target=self.function_to_run, args=self.its_args)
		p.start()
		p.join()

def train_test(MODEL_CONFIG_FILENAME):
	# Load the configurations for the model
	configs = get_configs(MODEL_CONFIG_FILENAME)
	configs['if_model_image'] 				= 0
	configs['n_past'] 						= 25	#Number of past observations timesteps to be considered for prediction
	configs['n_future'] 					= 10	#Number of future predictions timesteps to be made
	configs['known_past_features'] 			= 2 	#Number of features in the past observation window data
	configs['known_future_features'] 		= 3 	#Number of features in the future prediction window data
	configs['unknown_future_features'] 		= 2		#Number of features in the future prediction window data to be predicted
	
	# Generate random time series data for training and evaluation (sequence-to-sequence)
	num_samples 		= 1000
	x_known_past 		= np.random.random((num_samples, configs['n_past'], configs['known_past_features']))
	x_known_future 		= np.random.random((num_samples, configs['n_future'], configs['known_future_features']))
	y_unknown_future 	= np.random.random((num_samples, configs['n_future'], configs['unknown_future_features']))

	# Split the data into training and testing sets using a basic Python function
	train_test_split_percentage 					= 0.8
	split_index 									= int(train_test_split_percentage * num_samples)
	x_train_known_past, x_test_known_past 			= x_known_past[:split_index], x_known_past[split_index:]
	x_train_known_future, x_test_known_future 		= x_known_future[:split_index], x_known_future[split_index:]
	y_train_unknown_future, y_test_unknown_future 	= y_unknown_future[:split_index], y_unknown_future[split_index:]

	# 'ident' is a string used to ensure unique file and prediction case names
	ident 				= 'test_'
	current_run_dt 		= ident + str(dt.datetime.now().strftime('%d.%m-%H.%M.%S'))

	# Initialize and build the model
	model_class = Model_Gen(configs, ident)
	model_class.build_model()

	# Note: Model compilation happens inside Model_Gen.build_model() and is dependent on the user's choice.
	# Note: User can also compile model again using different optimizer and loss function

	# Train the model
	model_class.model.fit([x_train_known_past, x_train_known_future], y_train_unknown_future, batch_size=128, epochs=3, validation_split=0.2)

	# Evaluate the model
	test_loss, test_accuracy = model_class.model.evaluate([x_test_known_past, x_test_known_future], y_test_unknown_future)
	print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

	# Save the model with a unique filename based on ‘current_run_dt'. Can use both keras.model.save_weights() or keras.model.save()
	model_class.model.save(f'{current_run_dt}_random_time_series_model.h5')

if __name__ == '__main__':
	MultiprocessingWindow(train_test, (['CONN_based_on_bi_directional_LSTMs_with_Bahdanau_attention.yaml']))()
	MultiprocessingWindow(train_test, (['CONN_based_on_multi_head_attention.yaml']))()
	MultiprocessingWindow(train_test, (['CONN_based_on_temporal_convolutional_network_with_Bahdanau_attention.yaml']))()
```
Figure below illustrates the schematics of the Keras models generated from these three configuration files.
![alt text](https://github.com/gaurav306/pymodconn/blob/master/Readme_images/FIG7.png)
# License
MIT License


**Free Software, Hell Yeah!**

   [pymodconn\configs\default_config.yaml]: <https://github.com/gaurav306/pymodconn/blob/master/pymodconn/configs/default_config.yaml>