import fastai
import json
import torch.optim

from fastai.tabular import *

from .databunch import DashTabularDatabunch
from .metric import DashTabularMetric
from .model import DashTabularModel
from .loss import DashTabularLoss
from .optimizer import DashTabularOptimizer

import fastai

class DashTabularLearner:

	@staticmethod
	def create_tabular_learner():
		with open('./data/response.json') as f:
			response = json.load(f)
		
		path = Path('./')
		databunch = DashTabularDatabunch.create_tabular_databunch(response)
		metrics = DashTabularMetric.create_tabular_metric(response)
		# if response["model"]["type"] == "default":
		# 	model = DashTabularModel.create_tabular_model(databunch, response['model'])

		loss = DashTabularLoss.create_tabular_loss(response['loss'])
		opt = DashTabularOptimizer.create_tabular_optimizer(response['optimizer'])

		# Test it out
		if response["model"]["type"] == "custom":
			layers = []
			for l in response["model"]["custom"]["layers"]:
				layers.append(eval(l))
			model = DashTabularModel.create_custom_tabular_model(databunch, layers, **response["model"]["custom"]["extra_args"])
			learn = Learner(databunch, model, metrics=metrics, loss_func=loss, opt_func=opt)
			return learn


		learn = tabular_learner(
			databunch,
			layers=[200,100],
			metrics=metrics,
			opt_func=opt
#			,
#			loss_func = loss #Passing loss gives errors
		)

		return learn