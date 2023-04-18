# pymodconn

pymodconn = A Python library for developing modular sequence to sequence control oriented neural networks

## Instructions

1. Install:

```
pip install pymodconn
```

2. Usage:
Download congifuration file from tests_usage\ in the github repository https://github.com/gaurav306/pymodconn

```python
from pymodconn.configs_init import get_configs
from pymodconn.model_gen import ModelClass

configs = get_configs('config_model.yaml')
model_class = ModelClass(configs_data, time_dt)
model_class.build_model()
print('model_class.model.inputs: ',model_class.model.inputs)
print('model_class.model.outputs: ',model_class.model.outputs)
```

## Credits
packaging instructions from https://towardsdatascience.com/how-to-package-your-python-code-df5a7739ab2e