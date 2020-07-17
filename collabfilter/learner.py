import fastai
import json
import torch.optim

from fastai.collab import *

from .databunch import DashCollabDatabunch
from .metric import DashCollabMetric
from .model import DashCollabModel
from .loss import DashCollabLoss
from .optimizer import DashCollabOptimizer

import fastai


class DashCollabLearner:
	
	@staticmethod
	def create_collab_learner(response):
		# with open('./data/response.json') as f:
		# 	response = json.load(f)
	
		path = Path('./')
		databunch = DashCollabDatabunch.create_collab_databunch(response)
		metrics = DashCollabMetric.create_collab_metric(response["core"])
		'''
		if response["collab"]["model"]["type"] == "default":
			model = DashCollabModel.create_collab_model(databunch, response["collab"]['model'])'''
		#Custom model not sure how to do
		loss = DashCollabLoss.create_collab_loss(response["core"]['loss'])
		opt = DashCollabOptimizer.create_collab_optimizer(response["core"]['optimizer'])
		learn=collab_learner(databunch,n_factors=50, y_range=(0.,5.),metrics=metrics,
			opt_func=opt,
			loss_func=loss)
		
		return learn
