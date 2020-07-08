from fastai.tabular import *
from core.databunch import DashDatabunch

class DashTabularDatabunch:

	@staticmethod
	def create_tabular_databunch(response):
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

		src = DashDatabunch.split_databunch(response, src)
		src = DashDatabunch.label_databunch(response, src)

		# Optional: add test
		if response['test_df']:
			src.add_test(TabularList.from_df(response['test_df'],
						 cat_names=cat_names,
						 cont_names=cont_names,
						 processor=src.train.x.processor))

		# Step 4: Convert to databunch
		return DashDatabunch.create_databunch(response, src)