from fastai.layers import *
from pathlib import Path
import sys, os

class DashLoss:

	@staticmethod
	def create_loss(response):
		try:
			if response['type'] == 'pre-defined':
				if response['pre-defined']['func'] == 'CrossEntropyFlat':
					loss_func = CrossEntropyFlat

				if response['pre-defined']['func'] == 'MSELossFlat':

					loss_func = MSELossFlat

				if response['pre-defined']['func'] == 'BCEFlat':
					loss_func = BCEFlat

				if response['pre-defined']['func'] == 'BCEWithLogitsFlat':
					loss_func = BCEWithLogitsFlat
			else:
				#import response['custom']['fname']
				func = fname.response['custom']['func']
				loss_func = FlattenedLoss(func)

			return loss_func

		except Exception as e:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)