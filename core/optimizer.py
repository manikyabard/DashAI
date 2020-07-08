from fastai.tabular import *
import torch.optim

class DashOptimizer:
	
	@staticmethod
	def create_optimize(response):
		try:
			if response['chosen_opt'] == 'SGD':
				arg = response['arguments']['SGD']
				opt_func = partial(
					optim.SGD,
					lr=arg['lr'],
					momentum=arg['momentum'],
					dampening=arg['dampening'],
					nesterov=arg['nesterov'],
					weight_decay=arg['weight_decay']
				)


			if response['chosen_opt'] == 'RMSProp':
				arg = response['arguments']['RMSProp']
				opt_func = partial(
					optim.RMSprop,
					lr=arg['lr'],
					eps=arg['eps'],
					momentum=arg['momentum'],
					alpha=arg['alpha'],
					weight_decay=arg['weight_decay'],
					centered=arg['centered']
				)


			if response['chosen_opt'] == 'Adam':
				arg = response['arguments']['Adam']
				opt_func = partial(
					optim.Adam,
					lr=arg['lr'],
					betas=(arg['momentum'], arg['alpha']),
					eps=arg['eps'],
					weight_decay=arg['weight_decay'],
					amsgrad=arg['amsgrad']
				)


			if response['chosen_opt'] == 'AdamW':
				arg = response['arguments']['AdamW']
				opt_func = partial(
					optim.AdamW,
					lr=arg['lr'],
					betas=(arg['momentum'], arg['alpha']),
					eps=arg['eps'],
					amsgrad=arg['amsgrad'],
					weight_decay=arg['weight_decay']
				)


			if response['chosen_opt'] == 'Adadelta':
				arg = response['arguments']['Adadelta']
				opt_func = partial(
					optim.Adadelta,
					lr=arg['lr'],
					eps=arg['eps'],
					weight_decay=arg['weight_decay'],
					rho=arg['rho']
				)

			if response['chosen_opt'] == 'Adagrad':
				arg = response['arguments']['Adagrad']
				opt_func = partial(
					optim.Adagrad,
					lr=arg['lr'],
					lr_decay=arg['lr_decay'],
					eps=arg['eps'],
					weight_decay=arg['weight_decay']
				)

			if response['chosen_opt'] == 'SparseAdam':
				arg = response['arguments']['SparseAdam']
				opt_func = partial(
					optim.SparseAdam,
					lr=arg['lr'],
					betas=(arg['momentum'], arg['alpha']),
					eps=arg['eps']
				)

			if response['chosen_opt'] == 'Adamax':
				arg = response['arguments']['Adamax']
				opt_func = partial(
					optim.Adamax,
					lr=arg['lr'],
					betas=(arg['momentum'], arg['alpha']),
					eps=arg['eps'],
					weight_decay=arg['weight_decay']
				)

			if response['chosen_opt'] == 'ASGD':
				arg = response['arguments']['ASGD']
				opt_func = partial(optim.ASGD,
					lr=arg['lr'],
					momentum=arg['momentum'],
					alpha=arg['alpha'],
					lambd=arg['lambd'],
					t0=arg['t0'],
					weight_decay=arg['weight_decay']
				)

			return opt_func

		except Exception as e:
			print(e)