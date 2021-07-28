from core.loss import DashLoss


class DashTabularLoss:

	@staticmethod
	def create_tabular_loss(response):
		# TODO: Need to verify that the metric actually works for tabular
		#  (Maybe this can also be done on the front end side by giving limited options)
		return DashLoss.create_loss(response)
