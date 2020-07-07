import fastai
import json
import torch.optim

from fastai.tabular import *
from .databunch import DashTabularDatabunch
from .metric import DashTabularMetric
from .model import DashTabularModel
from .loss import DashTabularLoss
from .optimizer import DashTabularOptimizer

class DashTabularLearner:

	@staticmethod
	def create_tabular_learner():
		with open('./data/response.json') as f:
			response = json.load(f)
		
		databunch = DashTabularDatabunch.create_tabular_databunch(response)
		metrics = DashTabularMetric.create_tabular_metric(response)
		# model = DashTabularModel.create_tabular_model(databunch, response['model'])
		loss = DashTabularLoss.create_tabular_loss(response['loss'])
		opt = DashTabularOptimizer.create_optimize(response['optimizer'])

		learn = tabular_learner(
			databunch,
			layers=[200,100],
			metrics=metrics,
			opt_func=opt
		)

		return learn