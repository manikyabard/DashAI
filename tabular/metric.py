from fastai.tabular import *
from core.metrics import DashMetrics

class DashTabularMetric:

	@staticmethod
	def create_tabular_metric(response):
		return DashMetrics.create_metric(response)
