# pymodconn

pymodconn = A Python package for developing modular sequence to sequence control oriented neural networks

## Instructions

1. Install:

```
pip install pymodconn
```

2. Usage:
Download congifuration file from pymodconn\configs\default_config.yaml

```python
from pymodconn.configs_init import get_configs
from pymodconn import Model_Gen
import datetime as dt

configs = get_configs('config_model.yaml')
ident = 'test_'
current_run_dt = ident + str(dt.datetime.now().strftime('%H.%M.%S.%f')[:-3])

model_class = Model_Gen(configs_data, current_run_dt)
model_class.build_model()
print('model_class.model.inputs: ',model_class.model.inputs)
print('model_class.model.outputs: ',model_class.model.outputs)
```