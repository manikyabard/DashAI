import os
import sys

from fastai import layers

class DashLoss:

	@staticmethod
	def create_loss(response):
		try:
			if response['type'] == 'pre-defined':
				if hasattr(layers, f"response['pre-defined']['func']"):
					loss_func = getattr(layers, f"response['pre-defined']['func']()")
			else:
				#import response['custom']['fname']
				func = fname.response['custom']['func']
				loss_func = FlattenedLoss(func)

			return loss_func

		except Exception as e:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)