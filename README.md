# mod_seq2seq_conn

mod_seq2seq_conn = modular sequence to sequence control oriented neural networks

## Instructions

1. Install:

```
pip install mod_seq2seq_conn
```

2. Usage:
Download congifuration file from tests_usage\ in the github repository https://github.com/gaurav306/mod_seq2seq_conn

```python
from mod_seq2seq_conn.configs_init import get_configs
from mod_seq2seq_conn.model_tfp import ModelClass_tfp

configs = get_configs('config_model.yaml')
model = ModelClass_tfp(configs_data, time_dt)
```



packaging instructions from https://towardsdatascience.com/how-to-package-your-python-code-df5a7739ab2e