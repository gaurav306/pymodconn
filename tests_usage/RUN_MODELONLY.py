import datetime as dt
import sys
import tensorflow as tf

from mod_seq2seq_conn.configs_init import get_configs
from mod_seq2seq_conn.model_tfp import ModelClass_tfp

#MODELT = int(sys.argv[1])

# takes argument as time_dt for all file saves and configuration json data
def runmain(time_dt, configs_data):
	tf.keras.backend.clear_session()
	model_case = ModelClass_tfp(configs_data, time_dt)
	return model_case.build_model()


configs = get_configs('config_model.yaml')
configs['model']['if_model_image'] = 1
configs['model']['if_model_summary'] = 1

configs['model']['IF_GLU'] = 1
configs['model']['IF_ADDNORM'] = 1
configs['model']['IFFFN'] = 1
configs['model']['IFSELF_MHA'] = 1
configs['model']['IFCASUAL_MHA'] = 1
configs['model']['IFCROSS_MHA'] = 1
configs['rnn_units']['rnn_type'] = 'GRU'
configs['rnn_units']['input_enc_rnn_depth'] = 5
configs['rnn_units']['input_enc_rnn_bi'] = 1
configs['model']['control_future_cells'] = 1
configs['model']['all_layers_neurons'] = 64

ident = 'Decoders-%s_MHAfirst-%s_RNN-%s_Depth-%s_bi-%s_NNs-%s_' % (
		configs['model']['control_future_cells'],	
		configs['model']['MHA_RNN'],
		configs['rnn_units']['rnn_type'],
		configs['rnn_units']['input_enc_rnn_depth'],
		configs['rnn_units']['input_enc_rnn_bi'],
		configs['model']['all_layers_neurons'])

ident = 'test_'
current_run_dt = ident + str(dt.datetime.now().strftime('%d.%m.%Y-%H.%M.%S.%f')[:-3])
print(current_run_dt)
print(runmain(current_run_dt, configs))