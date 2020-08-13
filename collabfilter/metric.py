from core.metrics import DashMetrics


class DashCollabMetric:
	"""
	Metrics for Collaborative filtering
	"""

	@staticmethod
	def create_collab_metric(response):
		# TODO: Need to verify that the metric actually works for tabular
		#  (Maybe this can also be done on the front end side by giving limited options)
		return DashMetrics.create_metric(response)
