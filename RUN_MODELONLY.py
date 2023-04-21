import datetime as dt
import sys
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
configs['if_model_image'] = 1
configs['if_model_summary'] = 0

configs['IF_GLU'] = 0
configs['IF_ADDNORM'] = 0
configs['IFFFN'] = 0
configs['IFSELF_MHA'] = 1
configs['IFCASUAL_MHA'] = 1
configs['IFCROSS_MHA'] = 1
configs['rnn_type'] = 'GRU'
configs['input_enc_rnn_depth'] = 5
configs['input_enc_rnn_bi'] = 1
configs['control_future_cells'] = 6
configs['all_layers_neurons'] = 64

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