#!/usr/bin/env python3
import copy

from fastai.utils.mod_display import progress_disabled_ctx
from fastai.metrics import accuracy
from ax import optimize


class DashVerum:
	def __init__(self, response, learn):
		self.response = response
		self.learn = learn

	def init_parameters(self):
		parameters = []
		for k, v in self.response.items():
			if v['flag']:
				parameters.append(v['param'])
		return parameters

	def get_param_value(self, param):
		eval_reqd = ['learning rate']
		if self.response[param]['flag']:
			if param in eval_reqd:
				return eval(self.response[param]['param'])
			return self.response[param]['param']
		return self.response[param]['default']

	def evaluation_fn(self):
		lr = self.get_param_value('learning rate')
		num_epochs = self.get_param_value('num_epochs')
		moms = (self.get_param_value('momentum0'), self.get_param_value('momentum1'))
		ps = self.get_param_value('dropout_ps')
		wd = self.get_param_value('weight_decay')
		use_bn = self.get_param_value('use_bn')

		learn = copy.deepcopy(self.learn)
		learn.model.ps = ps
		learn.model.wd = wd
		learn.model.use_bn = use_bn

		validation_set = learn.data.valid_dl
		learn.data.valid_dl = None

		with progress_disabled_ctx(learn) as learn:
			learn.fit_one_cycle(num_epochs, max_lr=lr, moms=moms)

		learn.data.valid_dl = validation_set
		if self.response['metric']['name'] == 'error':
			metric = learn.validate()[0]
		else:
			metric = learn.validate(metrics=accuracy)[0]

		return metric

	def veritize(self):
		total_trials = self.response['metric']['num_trials'] if self.response['metric']['num_trials'] else 20
		best_parameters, best_values, experiment, model = optimize(
			parameters=self.init_parameters(),
			evaluation_function=self.evaluation_fn(),
			minimize=self.response['metric']['minimize'],
			total_trials=total_trials
		)
		# TODO finish.
