from core.optimizer import DashOptimizer


class DashCollabOptimizer:
	"""
	Optimizer for collaborative filtering
	"""

	@staticmethod
	def create_collab_optimizer(response):
		# TODO: Need to verify that the metric actually works for tabular
		#  (Maybe this can also be done on the front end side by giving limited options)
		return DashOptimizer.create_optimize(response)
