import multiprocessing
import numpy as np
import sys
#sys.path.append("..")
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