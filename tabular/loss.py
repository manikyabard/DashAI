from fastai.tabular import *
import fastai

class DashTabularLoss:

	@staticmethod
	def create_tabular_loss(response):
		if response['type'] == 'pre-defined':
			if response['pre-defined']['func'] == 'CrossEntropyFlat':
				loss_func = fastai.layers.CrossEntropyFlat

			if response['pre-defined']['func'] == 'MSELossFlat':
				loss_func = fastai.layers.MSELossFlat

			if response['pre-defined']['func'] == 'BCEFlat':
				loss_func = fastai.layers.BCEFlat

			if response['pre-defined']['func'] == 'BCEWithLogitsFlat':
				loss_func = fastai.layers.BCEWithLogitsFlat

		else:
			#import response['custom']['fname']
			func = fname.response['custom']['func']
			loss_func = FlattenedLoss(func)

		return loss_func