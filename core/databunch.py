import fastai
from pathlib import Path
import sys, os
from fastai.vision import *
import importlib


#just for now
from data.func.img2bbox import func_img2bbox


class DashDatabunch:

	@staticmethod
	def split_databunch(response, src):
		try:
			path = Path('./')

			if hasattr(src, f"split_{response['validation']['method']}"):
				if response['validation']['method'] == 'none':
					args = {}

				if response['validation']['method'] == 'by_rand_pct':
					args = {
						'valid_pct': response['validation']['rand_pct']['valid_pct'],
						'seed': response['validation']['rand_pct']['seed']
					}

				if response['validation']['method'] == 'subsets':
					args = {
						'train_size': response['validation']['subsets']['train_size'],
						'valid_size': response['validation']['subsets']['valid_size'],
						'seed': response['validation']['subsets']['seed']
					}

				if response['validation']['method'] == 'by_files':       #TODO: test it out
					args = {'valid_name': response['validation']['files']['valid_names']}

				if response['validation']['method'] == 'by_fname_file':
					args = {
						'fname': response['validation']['fname_files']['fname'],
						'path': response['validation']['fname_files']['path']
					}

				if response['validation']['method'] == 'by_folder':
					args = {
						'train': response['validation']['folder']['train'],
						'valid': response['validation']['folder']['valid']
					}

				if response['validation']['method'] == 'by_idx':
					df = pd.open_csv(response['csv_name'])
					valid_idx = range(len(df)-response['validation']['idx']['valid_idx'], len(df))
					args = {'valid_idx': valid_idx}

				if response['validation']['method'] == 'by_idxs':
					args = {
						'train_idx': response['validation']['idxs']['train_idx'],
						'valid_idx': response['validation']['idxs']['valid_idx']
					}

				if response['validation']['method'] == 'by_list':
					args = {
						'train': response['validation']['list']['train'],
						'valid': response['validation']['list']['valid']
					}

				# Probably won't work right now. Need to use impportlib or something similar
				# if response['validation']['method'] == 'valid_func':
				# 	funcfile = response['validation']['valid_func']['fname']
				# 	import funcfile as fname
				# 	func = fname.response['validation']['valid_func']['func']
				# 	src = src.split_by_valid_func(func)

				if response['validation']['method'] == 'from_df':
					args = {'col': response['validation']['from_df']['col']}

				return getattr(src, f"split_{response['validation']['method']}")(**args)

		except Exception as e:
			print(e)

	@staticmethod
	def label_databunch(response, src):
		# try:
		if hasattr(src, f"label_{response['label']['method']}"):
			if response['label']['method'] == 'from_df':      #TODO test it out
				if response['label']['from_df']['default']:
					args = {'cols': response['dep_var']}
				else:
					args = response['label']['from_df']

			if response['label']['method'] == 'empty':
				args = {}

			if response['label']['method'] == 'const':
				args = {
					'const': response['label']['const']['const'],
					'label_cls': response['label']['const']['label_cls']
				}

			# TODO Find a better way to do this
			if response['label']['method'] == 'from_func':
				images, lbl_bbox = get_annotations('data/coco_tiny/train.json')
				img2bboxd = dict(zip(images, lbl_bbox))

				src = src.label_from_func(lambda o:img2bboxd[o.name])
				return src

			if response['label']['method'] == 're':
				args = {
					'pat': response['label']['re']['pat'],
					'full_path': response['label']['re']['full_path']
				}
			if response['label']['method'] == 'from_folder':
				args = {}

			return getattr(src, f"label_{response['label']['method']}")(**args)

		# except Exception as e:
		# 	exc_type, exc_obj, exc_tb = sys.exc_info()
		# 	fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		# 	print(exc_type, fname, exc_tb.tb_lineno)

	@staticmethod
	def create_databunch(response, src, **kwargs):
		path = Path('./')

		return src.databunch(
				path=path,
				bs=response['bs'],
				val_bs=response['val_bs'],
				num_workers=response['num_workers'],
				device=response['device'],
				no_check=response['no_check'],
				**kwargs
			)