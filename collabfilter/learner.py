from fastai.collab import *

from .databunch import DashCollabDatabunch
from .loss import DashCollabLoss
from .metric import DashCollabMetric
from .optimizer import DashCollabOptimizer


class DashCollabLearner:
	"""
	`Learner` suitable for collaborative filtering.
	"""

	@staticmethod
	def create_collab_learner(response):
		"""
		Create a Learner for collaborative filtering on `data`.
		"""

		path = Path('./')
		databunch = DashCollabDatabunch.create_collab_databunch(response)
		metrics = DashCollabMetric.create_collab_metric(response["core"])
		re_def = response["collab"]['model']

		if re_def['type'] == 'default':
			n_factor = re_def['default']['n_factor']
			use_nn = re_def['default']['use_nn']
			layers = re_def['default']['layers']
			emb_drop = re_def['default']['emb_drop']
			ps = re_def['default']['ps']
			y_range = tuple(re_def['default']['y_range'])
			use_bn = re_def['default']['use_bn']
			bn_final = re_def['default']['bn_final']
			emb_szs = databunch.get_emb_szs()
		# if response["collab"]["model"]["type"] == "default":
		# model = DashCollabModel.create_collab_model(databunch, response["collab"]['model'])
		# Custom model not sure how to do
		loss = DashCollabLoss.create_collab_loss(response["core"]['loss'])
		opt = DashCollabOptimizer.create_collab_optimizer(response["core"]['optimizer'])
		if response["collab"]["model"]["type"] == "default":
			learn = collab_learner(databunch, n_factors=n_factor, y_range=y_range, metrics=metrics,
								   opt_func=opt,
								   loss_func=loss)
			'''
			learn=collab_learner(databunch,n_factor=n_factor,metrics=metrics,
			opt_func=opt,
			loss_func=loss)'''
		# y_range=y_range,use_bn=use_bn,bn_final=bn_final,n_factor=n_factor)
		# layers=layers,ps=ps,emb_drop=emb_drop,y_range=y_range,use_bn=use_bn,bn_final=bn_final,n_factor=n_factor)

		return learn
