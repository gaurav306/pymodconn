import datetime as dt
import sys
sys.path.append("..")
import tensorflow as tf

from pymodconn.configs_init import get_configs
from pymodconn.model_gen import ModelClass

# takes argument as time_dt for all file saves and configuration json data
def runmain(time_dt, configs_data):
	tf.keras.backend.clear_session()
	model_class = ModelClass(configs_data, time_dt)
	model_class.build_model()
	print('model_class.model.inputs: ',model_class.model.inputs)
	print('model_class.model.outputs: ',model_class.model.outputs)


configs = get_configs('config_model.yaml')

ident = 'Decoders-%s_MHAfirst-%s_RNN-%s_Depth-%s_bi-%s_NNs-%s_' % (
		configs['control_future_cells'],	
		configs['MHA_RNN'],
		configs['rnn_type'],
		configs['input_enc_rnn_depth'],
		configs['input_enc_rnn_bi'],
		configs['all_layers_neurons'])

ident = 't2_'
current_run_dt = ident + str(dt.datetime.now().strftime('%H.%M.%S.%f')[:-3])
runmain(current_run_dt, configs)