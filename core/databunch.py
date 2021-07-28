from fastai.vision import *


class DashDatabunch:
	"""
	Provides helper functions for splitting, labelling, and creating databunch for your itemlists
	"""

	@staticmethod
	def split_databunch(response, src):
		"""
		Splits databunch according to the method specified in the response/DashUI.
		"""
		# try:
		path = Path('./')
		response_split = response["core"]["data"]
		if hasattr(src, f"split_{response_split['validation']['method']}"):
			if response_split['validation']['method'] == 'none':
				args = {}

			if response_split['validation']['method'] == 'by_rand_pct':
				args = {
					'valid_pct': response_split['validation']['by_rand_pct']['valid_pct'],
					'seed': response_split['validation']['by_rand_pct']['seed']

				}

			if response_split['validation']['method'] == 'subsets':
				args = {
					'train_size': response_split['validation']['by_subsets']['train_size'],
					'valid_size': response_split['validation']['by_subsets']['valid_size'],
					'seed': response_split['validation']['by_subsets']['seed']
				}

			if response_split['validation']['method'] == 'by_files':  # TODO: test it out
				args = {'valid_name': response_split['validation']['by_files']['valid_names']}

			if response_split['validation']['method'] == 'by_fname_file':
				args = {
					'fname': response_split['validation']['by_fname_files']['fname'],
					'path': response_split['validation']['by_fname_files']['path']
				}

			if response_split['validation']['method'] == 'by_folder':
				args = {
					'train': response_split['validation']['by_folder']['train'],
					'valid': response_split['validation']['by_folder']['valid']
				}
			# For tabular, same csv; for vision, csv with labels
			if response_split['validation']['method'] == 'by_idx':
				df = pd.open_csv(response_split['validation']['csv_name'])
				valid_idx = range(len(df) - response_split['validation']['by_idx']['valid_idx'], len(df))
				args = {'valid_idx': valid_idx}

			if response_split['validation']['method'] == 'by_idxs':
				args = {
					'train_idx': response_split['validation']['by_idxs']['train_idx'],
					'valid_idx': response_split['validation']['by_idxs']['valid_idx']
				}

			if response_split['validation']['method'] == 'by_list':
				args = {
					'train': response_split['validation']['by_list']['train'],
					'valid': response_split['validation']['by_list']['valid']
				}

			if response_split['validation']['method'] == 'by_valid_func':
				with open(response['validation']['by_valid_func']['location']) as f:
					valid_func_text = f.read()
				exec(valid_func_text)
				args = {'func': valid_func}

			if response_split['validation']['method'] == 'from_df':
				args = {'col': response_split['validation']['from_df']['col']}

			return getattr(src, f"split_{response_split['validation']['method']}")(**args)

	# except Exception as e:
	# 	print(e)

	@staticmethod
	def label_databunch(response, src):
		"""
		Labels itemlist according to the method specified in the response/DashUI.
		"""
		# try:
		response_lab = response["core"]["data"]
		if hasattr(src, f"label_{response_lab['label']['method']}"):
			# A bit specific to tabular, safe to remove the defaults
			if response_lab['label']['method'] == 'from_df':  # TODO test it out
				args = response_lab['label']['from_df']

			if response_lab['label']['method'] == 'empty':
				args = {}

			if response_lab['label']['method'] == 'const':
				args = {
					'const': response_lab['label']['const']['const'],
					'label_cls': response_lab['label']['const']['label_cls']
				}

			if response_lab['label']['method'] == 'from_func':
				with open(response_lab['label']['from_func']['location']) as f:
					label_func_text = f.read()
				exec(label_func_text)
				args = {'func': label_func}

			if response_lab['label']['method'] == 're':
				args = {
					'pat': response_lab['label']['re']['pat'],
					'full_path': response_lab['label']['re']['full_path']
				}

			if response_lab['label']['method'] == 'from_folder':
				args = {}

			if response_lab['label']['method'] == 'for_lm':
				args = {}

			return getattr(src, f"label_{response_lab['label']['method']}")(**args)

	# except Exception as e:
	# 	exc_type, exc_obj, exc_tb = sys.exc_info()
	# 	fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
	# 	print(exc_type, fname, exc_tb.tb_lineno)

	@staticmethod
	def create_databunch(response, src, **kwargs):
		"""
		Create a databunch using the itemlist and the parameters specified in response/DashUI
		"""
		path = Path('./')
		response_data = response["core"]["data"]
		return src.databunch(
			path=path,
			bs=response_data['bs'],
			val_bs=response_data['val_bs'],
			num_workers=response_data['num_workers'],
			device=response_data['device'],
			no_check=response_data['no_check'],
			**kwargs
		)
