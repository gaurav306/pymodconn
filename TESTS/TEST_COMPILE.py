import datetime as dt
import sys
#sys.path.append("..")
import shutil
import tensorflow as tf

from pymodconn.configs.configs_init import get_configs
from pymodconn import Model_Gen

# takes argument as time_dt for all file saves and configuration json data
def runmain(time_dt, configs_data):
	tf.keras.backend.clear_session()
	model_class = Model_Gen(configs_data, time_dt)
	model_class.build_model()

#shutil.copy('new_config.yaml', '..\pymodconn\configs\default_config.yaml')

configs = get_configs('CONN_based_on_temporal_convolutional_network_with_Bahdanau_attention.yaml')
ident = 'TCN_'
current_run_dt = ident + str(dt.datetime.now().strftime('%H.%M.%S.%f')[:-3])
runmain(current_run_dt, configs)

configs = get_configs('CONN_based_on_multi_head_attention.yaml')
ident = 'MHA_'
current_run_dt = ident + str(dt.datetime.now().strftime('%H.%M.%S.%f')[:-3])
runmain(current_run_dt, configs)

configs = get_configs('CONN_based_on_bi_directional_LSTMs_with_Bahdanau_attention.yaml')
ident = 'RNN_'
current_run_dt = ident + str(dt.datetime.now().strftime('%H.%M.%S.%f')[:-3])
runmain(current_run_dt, configs)

