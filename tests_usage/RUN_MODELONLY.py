import datetime as dt
import sys
sys.path.append("..")
import tensorflow as tf
print('tf.__version__: ',tf.__version__)
import shutil

from pymodconn.configs.configs_init import get_configs
from pymodconn.configs.create_json_schema import create_schema_from_yaml
from pymodconn.model_gen import ModelClass

# takes argument as time_dt for all file saves and configuration json data
def runmain(time_dt, configs_data):
	tf.keras.backend.clear_session()
	model_class = ModelClass(configs_data, time_dt)
	model_class.build_model()
	#print('model_class.model.inputs: ',model_class.model.inputs)
	#print('model_class.model.outputs: ',model_class.model.outputs)

create_schema_from_yaml('config_model_test2.yaml', '..\pymodconn\configs\schema_validation.json')
shutil.copy('config_model_test2.yaml', '..\pymodconn\configs\default_config.yaml')

configs = get_configs('config_model_test2.yaml')



ident = 'Decoders-%s_MHAfirst-%s_RNN-%s_Depth-%s_bi-%s_NNs-%s_' % (
		configs['control_future_cells'],	
		configs['MHA_RNN'],
		configs['rnn_type'],
		configs['input_enc_rnn_depth'],
		configs['input_enc_rnn_bi'],
		configs['all_layers_neurons'])

ident = 'test_'
current_run_dt = ident + str(dt.datetime.now().strftime('%H.%M.%S.%f')[:-3])
runmain(current_run_dt, configs)
'''


for i in [1,4,6]:
	configs['MERGE_STATES_METHOD'] = i
	ident = 'noMHA_MERGE_%s_grn_glu_' % i
	current_run_dt = ident + str(dt.datetime.now().strftime('%H.%M.%S.%f')[:-3])
	runmain(current_run_dt, configs)
'''