from fastai.tabular import *

response = {
	'loss': 'pre-defined',

	'pre-defined': {
		func: CrossEntropyFlat
	}

	'custom': {
		fname: None,
		func: None
	}
}

def create_tabular_loss(response):

	if response['loss'] == 'pre-defined':
		loss_func = response['pre-defined']['func']

	else:
		import response['custom']['fname'] as fname
		func = fname.response['custom']['func']
		loss_func = FlattenedLoss(func)

	return loss_func