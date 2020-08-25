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

		lr = (
			eval(DashVerum.v_resp['learning_rate']['default']) if not DashVerum.v_resp['learning_rate']['flag']
			else parameters['learning_rate']
		)
		num_epochs = (
			DashVerum.v_resp['num_epochs']['default'] if not DashVerum.v_resp['num_epochs']['flag']
			else parameters['num_epochs']
		)
		moms = (
			(
				DashVerum.v_resp['momentum0']['default'] if not DashVerum.v_resp['momentum0']['flag']
				else parameters['momentum0']
			),
			(
				DashVerum.v_resp['momentum1']['default'] if not DashVerum.v_resp['learning_rate']['flag']
				else parameters['momentum1']
			)
		)
		ps = (
			DashVerum.v_resp['dropout_ps']['default'] if not DashVerum.v_resp['learning_rate']['flag']
			else parameters['dropout_ps']
		)
		# wd = (
		# 	DashVerum.v_resp['weight_decay']['default'] if not DashVerum.v_resp['weight_decay']['flag']
		# 	else parameters['weight_decay']
		# )
		use_bn = (
			DashVerum.v_resp['use_bn']['default'] if not DashVerum.v_resp['use_bn']['flag']
			else parameters['use_bn']
		)


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
		# learn.model.wd = wd
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
		total_trials = (
			DashVerum.v_resp['metric']['num_trials']
			if DashVerum.v_resp['metric']['num_trials'] is not None
			else 2
		)
		best_parameters, best_values, experiment, model = optimize(
			parameters=self.init_parameters(),
			evaluation_function=self.evaluation_fn,
			minimize=DashVerum.v_resp['metric']['minimize'],
			total_trials=total_trials
		)
		if DashVerum.v_resp['return']:
			try: DashVerum.learn.model.ps = best_parameters['dropout_ps']
			except: pass
			# try: DashVerum.learn.model.wd = best_parameters['weight_decay']
			# except: pass
			try: DashVerum.learn.model.use_bn = best_parameters['use_bn']
			except: pass

			return_list = [DashVerum.learn]
			try: return_list.append(best_parameters['learning_rate'])
			except: return_list.append(DashVerum.v_resp['learning_rate']['default'])
			try: return_list.append(best_parameters['num_epochs'])
			except: return_list.append(DashVerum.v_resp['num_epochs']['default'])
			momentum_list = []
			try: momentum_list.append(best_parameters['momentum0'])
			except: momentum_list.append(DashVerum.v_resp['momentum0']['default'])
			try: momentum_list.append(best_parameters['momentum1'])
			except: momentum_list.append(DashVerum.v_resp['momentum1']['default'])
			return_list.append(tuple(momentum_list))

			return tuple(return_list)

		else:
			print(json.dumps(best_parameters, indent=4))
