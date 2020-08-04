from fastai.collab import *
from core.loss import DashLoss

class DashCollabLoss:
	'''
	Loss function for collaborative filtering
	'''

	@staticmethod
	def create_collab_loss(response):
		# TODO: Need to verify that the metric actually works for tabular (Maybe this can also be done on the front end side by giving limited options)
		return DashLoss.create_loss(response)