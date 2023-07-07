import datetime as dt
import sys
sys.path.append("..")
import tensorflow as tf
print('tf.__version__: ',tf.__version__)
import shutil

from pymodconn.configs.configs_init import get_configs
from pymodconn import Model_Gen

# takes argument as time_dt for all file saves and configuration json data
def runmain(time_dt, configs_data):
	tf.keras.backend.clear_session()
	model_class = Model_Gen(configs_data, time_dt)
	model_class.build_model()

shutil.copy('new_config.yaml', '..\pymodconn\configs\default_config.yaml')

configs = get_configs('new_config.yaml')
ident = 'CIT_1_'
current_run_dt = ident + str(dt.datetime.now().strftime('%H.%M.%S.%f')[:-3])
runmain(current_run_dt, configs)
'''
for i in [1,2,3]:
	configs = get_configs('new_config.yaml')
	configs['IF_SIMPLE_MODEL']['CIT_option'] = i
	ident = 'CIT_%s_all_' % i
	current_run_dt = ident + str(dt.datetime.now().strftime('%H.%M.%S.%f')[:-3])
	runmain(current_run_dt, configs)
'''