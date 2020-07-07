from fastai.tabular import *

class DashTabularModel:

	@staticmethod
	def create_tabular_model(databunch, response):

		if response['type'] == 'default':
			out_sz = response['default']['out_sz']
			layers = response['default']['layers']
			emb_drop = response['default']['emb_drop']
			ps = response['default']['ps']
			y_range = response['default']['y_range']
			use_bn = response['default']['use_bn']
			bn_final = response['default']['bn_final']

		model = TabularModel(
			emb_szs=databunch.get_emb_szs(),
			n_cont=len(databunch.cont_names),
			out_sz=out_sz,
			layers=layers,
			emb_drop=emb_drop,
			ps=ps,
			y_range=y_range,
			use_bn=use_bn,
			bn_final=bn_final
		)

		return model