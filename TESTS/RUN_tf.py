import multiprocessing
import numpy as np
import sys
sys.path.append("..")
from pymodconn import Model_Gen
from pymodconn.configs.configs_init import get_configs
import datetime as dt

# Wrapper class to spawn a multiprocessing window for training and testing 
# to avoid the issue of 'Tensorflow not releasing GPU memory'

class MultiprocessingWindow():
    def __init__(self, function_to_run, its_args):
        self.function_to_run = function_to_run
        self.its_args = its_args

    def __call__(self):
        try:
            multiprocessing.set_start_method('spawn', force=True)
            print("Multiprocessing window spawned")
        except RuntimeError:
            pass
        p = multiprocessing.Process(
            target=self.function_to_run, args=self.its_args)
        p.start()
        p.join()

def train_test():
    # Generate random time series data for training and evaluation (sequence-to-sequence)
    num_samples = 1000
    past_observation_window = 25
    future_prediction_window = 10
    num_features_known_past = 2
    num_features_known_future = 3
    num_features_unknown_future = 2

    x_known_past = np.random.random(
        (num_samples, past_observation_window, num_features_known_past))
    x_known_future = np.random.random(
        (num_samples, future_prediction_window, num_features_known_future))
    x_unknown_future = np.random.random(
        (num_samples, future_prediction_window, num_features_unknown_future))

    # Split the data into training and testing sets using a basic Python function
    train_test_split_percentage = 0.8
    split_index = int(train_test_split_percentage * num_samples)
    x_train_known_past, x_test_known_past = x_known_past[:
                                                         split_index], x_known_past[split_index:]
    x_train_known_future, x_test_known_future = x_known_future[
        :split_index], x_known_future[split_index:]
    y_train_unknown_future, y_test_unknown_future = x_unknown_future[
        :split_index], x_unknown_future[split_index:]

    # Load the configurations for the model
    configs = get_configs(
        'CONN_based_on_bi_directional_LSTMs_with_Bahdanau_attention.yaml')

    configs['n_past'] = past_observation_window
    configs['n_future'] = future_prediction_window
    configs['known_past_features'] = num_features_known_past
    configs['known_future_features'] = num_features_known_future
    configs['unknown_future_features'] = num_features_unknown_future

    # 'ident' is a string used to ensure unique file and prediction case names
    ident = 'test_'
    current_run_dt = ident + str(dt.datetime.now().strftime('%H.%M.%S'))

    # Initialize and build the model using your library
    model_class = Model_Gen(configs, ident)
    model_class.build_model()
    model = model_class.model

    # Note: Model compilation happens inside Model_Gen.build_model() and is dependent on the user's choice.
    # Note: User can also compile model again using different optimizer and loss function

    # Train the model
    history = model.fit(
        [x_train_known_past, x_train_known_future],
        y_train_unknown_future,
        batch_size=128,
        epochs=5,
        validation_split=0.2)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(
        [x_test_known_past, x_test_known_future], y_test_unknown_future)
    print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

    # Save the model with a unique filename based on ‘current_run_dt'
    # model.save_weights(f'{ident}_random_time_series_model.h5')
    model.save(f'{current_run_dt}_random_time_series_model.h5')

if __name__ == '__main__':
    MultiprocessingWindow(train_test, ())()
