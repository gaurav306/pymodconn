# pymodconn

## pymodconn = A Python package for developing modular sequence to sequence control oriented neural networks

![alt text](https://github.com/gaurav306/pymodconn/blob/master/Readme_images/Picture1.png)

A Control-Oriented Neural Network (CONN) is an artificial neural network designed to predict the future behavior and response of highly complex thermodynamic systems, taking into account past known data and future known data.

1. Past known data includes historical information such as weather data, system dynamics, and control inputs, which is used to train the neural network to understand the relationships between these variables and their effects on the system.
2. Future known data encompasses forecasted weather data and targeted control inputs, representing the expected conditions and desired actions that the system will be subject to in the future.

The neural network leverages this information to predict future unknown data, referring to the system dynamics or states that are not yet known. This prediction enables the control system to anticipate changes in the environment and adapt its actions accordingly, ensuring optimal performance and robustness in the face of uncertainties.

CONN is particularly useful in applications where systems are affected by external factors (such as weather) and require precise control over their dynamics, such as energy management systems, water resource management, and other infrastructure systems that must be optimized to ensure reliability and efficiency.

## Instructions

1. Install:

```
pip install pymodconn
```

2. Usage:
Download congifuration file from pymodconn\configs\default_config.yaml

```python
import numpy as np
from pymodconn.configs_init import get_configs
from pymodconn import Model_Gen
import datetime as dt

# Generate random time series data for training and evaluation (sequence-to-sequence)
num_samples = 1000
input_sequence_length = 10
output_sequence_length = 10
num_features = 3

# Known_past = Known past system dynamics + Known past control inputs to the system
# Known_future = Control inputs to the system in future
# Unknown_future = System dynamics of system in future with known future control inputs
x_train_known_past = np.random.random((num_samples, input_sequence_length, num_features))
x_train_known_future = np.random.random((num_samples, input_sequence_length, num_features))
y_train_unknown_future = np.random.random((num_samples, output_sequence_length, num_features))

x_test_known_past = np.random.random((num_samples, input_sequence_length, num_features))
x_test_known_future = np.random.random((num_samples, input_sequence_length, num_features))
y_test_unknown_future = np.random.random((num_samples, output_sequence_length, num_features))

# Load the configurations for the model
configs = get_configs('config_model.yaml')

# 'ident' is a string used to differentiate between different runs or cases
ident = 'test_'

# 'current_run_dt' is a timestamped string to ensure unique file and prediction case names
current_run_dt = ident + str(dt.datetime.now().strftime('%H.%M.%S.%f')[:-3])

# Initialize and build the model using your library
model_class = Model_Gen(configs, current_run_dt)
model_class.build_model()
model = model_class.model

# Note: Model compilation happens inside Model_Gen.build_model()

# Train the model
history = model.fit(
    [x_train_known_past, x_train_known_future],
    y_train_unknown_future,
    batch_size=32,
    epochs=10,
    validation_split=0.2
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate([x_test_known_past, x_test_known_future], y_test_unknown_future)

# Print the results
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

# Save the model with a unique filename based on 'current_run_dt'
model.save(f'{current_run_dt}_random_time_series_model.h5')
```


![alt text](https://github.com/gaurav306/pymodconn/blob/master/Readme_images/Picture2.png)
![alt text](https://github.com/gaurav306/pymodconn/blob/master/Readme_images/Picture3.png)
![alt text](https://github.com/gaurav306/pymodconn/blob/master/Readme_images/Picture-4.png)
