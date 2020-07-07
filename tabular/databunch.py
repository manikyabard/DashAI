from fastai.tabular import *

class DashTabularDatabunch:

	@staticmethod
	def create_tabular_databunch(response):
		try:

			path = Path('./')
			#Step 1: Provide inputs
			df = pd.read_csv(path/f'{response["csv_name"]}')

			procs = list()
			if response['FillMissing']:

				if response['FillMissing']['fill_strategy'] == 'MEDIAN':
					fill_strategy = FillStrategy.MEDIAN
				if response['FillMissing']['fill_strategy'] == 'COMMON':
					fill_strategy = FillStrategy.COMMON
				if response['FillMissing']['fill_strategy'] == 'CONSTANT':
					fill_strategy = FillStrategy.CONSTANT

				procs.append(partial(FillMissing, fill_strategy=fill_strategy,
									fill_val=response['FillMissing']['fill_val'],
									add_col = response['FillMissing']['add_col']))

			if response['Categorify']:
				procs.append(Categorify)

			if response['Normalize']:
				procs.append(Normalize)

			procs = listify(procs)

			cat_names = response['cat_names']
			cont_names = response['cont_names']
			src = TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

			# Step 2: Split data into train and valid
			if response['validation']['method'] == 'none':
				src = src.split_none()

			if response['validation']['method'] == 'rand_pct':
				src = src.split_by_rand_pct(
					valid_pct=response['validation']['rand_pct']['valid_pct'],
					seed=response['validation']['rand_pct']['seed']
				)

			if response['validation']['method'] == 'subsets':
				src = src.split_subsets(
					train_size=response['validation']['subsets']['train_size'],
					valid_size=response['validation']['subsets']['valid_size'],
					seed=response['validation']['subsets']['seed']
				)

			if response['validation']['method'] == 'files':       #TODO: test it out
				src = src.split_by_files(valid_name=response['validation']['files']['valid_names'])

			if response['validation']['method'] == 'fname_file':
				src = src.split_by_fname_file(
					fname=response['validation']['fname_files']['fname'],
					path=response['validation']['fname_files']['path']
				)

			if response['validation']['method'] == 'folder':
				src = src.split_by_folder(
					train=response['validation']['folder']['train'],
					valid=response['validation']['folder']['valid']
				)

			if response['validation']['method'] == 'idx':
				valid_idx = range(len(df)-response['validation']['idx']['valid_idx'], len(df))
				src = src.split_by_idx(valid_idx)

			if response['validation']['method'] == 'idxs':
				src = src.split_by_idxs(
					train_idx = response['validation']['idxs']['train_idx'],
					valid_idx = response['validation']['idxs']['valid_idx']
				)

			if response['validation']['method'] == 'list':
				src = src.split_by_list(
					train = response['validation']['list']['train'],
					valid = response['validation']['list']['valid']
				)

			if response['validation']['method'] == 'valid_func':
				funcfile = response['validation']['valid_func']['fname']
				import funcfile as fname
				func = fname.response['validation']['valid_func']['func']
				src = src.split_by_valid_func(func)

			if response['validation']['method'] == 'from_df':
				src = src.split_from_df(col=response['validation']['from_df']['col'])

			# Step 3: Label the inputs
			if response['label']['method'] == 'from_df':      #TODO test it out
				if not response['label']['from_df']['classes']:
					src = src.label_from_df(cols=response['dep_var'])
				else:
					src.label_from_df(
						cols=response['label']['from_df']['cols'],
						label_cls=response['label']['from_df']['label_cls'],
						one_hot=response['label']['from_df']['one_hot'],
						classes=response['label']['from_df']['classes']
					)

			if response['label']['method'] == 'empty':
				src = src.label_empty()

			if response['label']['method'] == 'const':
				src = src.label_const(
					const=response['label']['const']['const'],
					label_cls=response['label']['const']['label_cls']
				)

			if response['label']['method'] == 'from_func':
				funcfile = response['label']['from_func']['fname']
				import funcfile as fname
				func = fname.response['label']['from_func']['func']
				src = src.label_from_func(func)

			if response['label']['method'] == 're':
				src = src.label_from_re(
					pat=response['label']['re']['pat'],
					full_path=response['label']['re']['full_path']
				)

			# Optional: add test
			if response['test_df']:
				src.add_test(TabularList.from_df(response['test_df'],
							 cat_names=cat_names,
							 cont_names=cont_names,
							 processor=src.train.x.processor))

			# Step 4: Convert to databunch
			return src.databunch(
				path=path,
				bs=response['bs'],
				val_bs=response['val_bs'],
				num_workers=response['num_workers'],
				device=response['device'],
				no_check=response['no_check']
			)

		except Exception as e:
			print('Exception:', e)