# pymodconn

## _pymodconn_ : A Python package for developing modular sequence to sequence control oriented neural networks

![alt text](https://github.com/gaurav306/pymodconn/blob/master/Readme_images/Picture1.jpg)

A Control-Oriented Neural Network (CONN) is an artificial neural network designed to predict the future behavior and response of highly complex dynamic systems, taking into account past known data and future known data.

1. Past known data includes historical information such as weather data, system dynamics, and control inputs, which is used to train the neural network to understand the relationships between these variables and their effects on the system.
2. Future known data encompasses forecasted weather data and targeted control inputs, representing the expected conditions and desired actions that the system will be subject to in the future.

The neural network leverages this information to predict future unknown data, referring to the system dynamics or states that are not yet known. This prediction enables the control system to anticipate changes in the environment and adapt its actions accordingly, ensuring optimal performance and robustness in the face of uncertainties.

CONN is particularly useful in applications where systems are affected by external factors (such as weather) and require precise control over their dynamics, such as energy management systems, water resource management, and other infrastructure systems that must be optimized to ensure reliability and efficiency.

## Instructions

### 1. Install:
```
pip install pymodconn
```
### 2. Usage:
Download congifuration file from [pymodconn\configs\default_config.yaml]

```python
import numpy as np
from pymodconn.configs_init import get_configs
from pymodconn import Model_Gen
import datetime as dt

# Generate random time series data for training and evaluation (sequence-to-sequence)
num_samples                 = 1000
past_observation_window     = 25      
future_prediction_window    = 10      
num_features_known_past     = 5
num_features_known_future   = 3
num_features_unknown_future = 2

# Known_past = Known past system dynamics + Known past control inputs to the system
# Known_future = Control inputs to the system in future
# Unknown_future = System dynamics of system in future with known future control inputs
x_known_past      = np.random.random((num_samples, past_observation_window, num_features_known_past))
x_known_future    = np.random.random((num_samples, future_prediction_window, num_features_known_future))
x_unknown_future  = np.random.random((num_samples, output_sequence_length, num_features_unknown_future))

# Split the data into training and testing sets using a basic Python function
train_test_split_percentage                     = 0.8
split_index                                     = int(train_test_split_percentage * num_samples)
x_train_known_past, x_test_known_past           = x_known_past[:split_index], x_known_past[split_index:]
x_train_known_future, x_test_known_future       = x_known_future[:split_index], x_known_future[split_index:]
y_train_unknown_future, y_test_unknown_future   = x_unknown_future[:split_index], x_unknown_future[split_index:]

# Load the configurations for the model
configs = get_configs('config_model.yaml')

# 'ident' is a string used to ensure unique file and prediction case names
ident = 'test_'
ident = ident + str(dt.datetime.now().strftime('%H.%M.%S'))

# Initialize and build the model using your library
model_class = Model_Gen(configs, ident)
model_class.build_model()
model = model_class.model

# Note: Model compilation happens inside Model_Gen.build_model() and is dependent on the user's choice.
# For point-based forecasts, users can decide not to compile the model inside build_model() to have more
# control over the available compile options (e.g., loss function, metrics, and learning rate schedulers).
# For probabilistic forecasts, model.compile() occurs inside build_model() since custom loss functions are used.

# Train the model
history = model.fit(
    [x_train_known_past, x_train_known_future],
    y_train_unknown_future,
    batch_size=32,          # for example
    epochs=10,              # for example
    validation_split=0.2    # for example
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate([x_test_known_past, x_test_known_future], y_test_unknown_future)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

# Save the model with a unique filename based on 'current_run_dt'
model.save(f'{current_run_dt}_random_time_series_model.h5')
```

![alt text](https://github.com/gaurav306/pymodconn/blob/master/Readme_images/Picture2.jpg)
![alt text](https://github.com/gaurav306/pymodconn/blob/master/Readme_images/Picture3.jpg)
![alt text](https://github.com/gaurav306/pymodconn/blob/master/Readme_images/Picture4.jpg)
![alt text](https://github.com/gaurav306/pymodconn/blob/master/Readme_images/Picture5.jpg)
![alt text](https://github.com/gaurav306/pymodconn/blob/master/Readme_images/Picture6.jpg)


## License

MIT License

**Free Software, Hell Yeah!**

   [pymodconn\configs\default_config.yaml]: <https://github.com/gaurav306/pymodconn/blob/master/pymodconn/configs/default_config.yaml>