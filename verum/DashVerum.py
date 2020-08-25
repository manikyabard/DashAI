#!/usr/bin/env python3
import copy, json
from pathlib import Path
from fastai.utils.mod_display import progress_disabled_ctx
from fastai.metrics import accuracy
from ax import optimize
from fastai.basic_train import *

from text.learner import DashTextLearner
from tabular.learner import DashTabularLearner
from vision.learner import DashVisionLearner
from collabfilter.learner import DashCollabLearner
from core.train import DashTrain
from insights.DashInsights import DashInsights
from captum.insights.attr_vis import AttributionVisualizer

path = Path('./')
learner_class_map = {
	'collab': DashCollabLearner,
	'tabular': DashTabularLearner,
	'text': DashTextLearner,
	'vision': DashVisionLearner
}
class DashVerum:
	v_resp = {}
	learn = None
	def __init__(self, response, data, learn):
		DashVerum.v_resp = response
		DashVerum.learn = learn
		self.data = data

	def init_parameters(self):
		parameters = []
		for k, v in DashVerum.v_resp.items():
			if isinstance(v, dict) and 'flag' in v:
				if v['flag']:
					parameters.append(v['param'])
		return parameters

	def get_param_value(self, param):
		eval_reqd = ['learning rate']
		if DashVerum.v_resp[param]['flag']:
			if param in eval_reqd:
				return eval(DashVerum.v_resp[param]['param'])
			return DashVerum.v_resp[param]['param']
		return DashVerum.v_resp[param]['default']

	@staticmethod
	def evaluation_fn(parameters):
		# lr = self.get_param_value('learning_rate')
		# num_epochs = self.get_param_value('num_epochs')
		# moms = (self.get_param_value('momentum0'), self.get_param_value('momentum1'))
		# ps = self.get_param_value('dropout_ps')
		# wd = self.get_param_value('weight_decay')
		# use_bn = self.get_param_value('use_bn')

		lr = parameters['learning_rate']
		num_epochs = parameters['num_epochs']
		moms = (parameters['momentum0'], parameters['momentum1'])
		ps = parameters['dropout_ps']
		wd = parameters['weight_decay']
		use_bn = parameters['use_bn']


		# learn = load_learner('./','verum_test.pkl')
		# learn.data = self.data
		with open('./data/response.json') as f:
			response = json.load(f)
		application = response['task']
		save_dir = Path(response['save']['save_dir'])
		save_name = Path(response['save']['save_name'])
		learner_class = learner_class_map[application]
		learn = getattr(learner_class, f'create_{application}_learner')(response)

		learn.model.ps = ps
		learn.model.wd = wd
		learn.model.use_bn = use_bn

		validation_set = learn.data.valid_dl
		learn.data.valid_dl = None

		with progress_disabled_ctx(learn) as learn:
			learn.fit_one_cycle(num_epochs, max_lr=lr, moms=moms)

		learn.data.valid_dl = validation_set
		if DashVerum.v_resp['metric']['name'] == 'error':
			metric = learn.validate()[0]
		else:
			metric = learn.validate(metrics=eval(DashVerum.v_resp['metric']['name']))[0]

		return metric

	def veritize(self):
		total_trials = DashVerum.v_resp['metric']['num_trials'] if DashVerum.v_resp['metric']['num_trials'] else 20
		best_parameters, best_values, experiment, model = optimize(
			parameters=self.init_parameters(),
			evaluation_function=self.evaluation_fn,
			minimize=DashVerum.v_resp['metric']['minimize'],
			total_trials=total_trials
		)
		if DashVerum.v_resp['return']:
			DashVerum.learn.model.ps = best_parameters['dropout_ps']
			DashVerum.learn.model.wd = best_parameters['weight_decay']
			DashVerum.learn.model.use_bn = best_parameters['use_bn']

			return (
				DashVerum.learn,
				best_parameters['learning_rate'], best_parameters['num_epochs'],
				(best_parameters['momentum0'], best_parameters['momentum1'])
			)

		else:
			pass  # TODO show this on console.
