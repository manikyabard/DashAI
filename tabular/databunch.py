from fastai.tabular import *

from core.databunch import DashDatabunch


class DashTabularDatabunch:

	@staticmethod
	def create_tabular_databunch(response) -> 'DataBunch':
		"""
		Creates a tabular databunch using the values specified in response/DashUI.
		Uses the csv name name given in response>tabular>input to create the input dataframe.
		Applies the required transformations, and creates a tabular list using the specified cat and cont column names.
		The tabular list is splitted into training and validation sets using the specified method and then labelled.
		Finally it is converted to a databunch object and returned.
		"""
		path = Path('./')
		# Step 1: Provide inputs
		response_tab = response["tabular"]
		df = pd.read_csv(path / f'{response_tab["input"]["csv_name"]}')

		procs = list()
		# print(f"{response_tab['transform']['FillMissing']['fill_strategy']}")
		if response_tab["transform"]['FillMissing']:
			if hasattr(FillStrategy, f"{response_tab['transform']['FillMissing']['fill_strategy']}"):
				fill_strategy = getattr(FillStrategy, f"{response_tab['transform']['FillMissing']['fill_strategy']}")

			procs.append(partial(FillMissing, fill_strategy=fill_strategy,
								 fill_val=response_tab['transform']['FillMissing']['fill_val'],
								 add_col=response_tab['transform']['FillMissing']['add_col']))

		if response_tab['transform']['Categorify']:
			procs.append(Categorify)

		if response_tab['transform']['Normalize']:
			procs.append(Normalize)

		procs = listify(procs)

		cat_names = response_tab["input"]['cat_names']
		cont_names = response_tab["input"]['cont_names']
		src = TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

		src = DashDatabunch.split_databunch(response, src)
		src = DashDatabunch.label_databunch(response, src)

		# Optional: add test  #TODO Test it out
		if response["tabular"]["input"]['test_df']["has_test"]:
			test_df = pd.read_csv(path / f"{response['tabular']['input']['test_df']['csv_name']}")
			src.add_test(TabularList.from_df(test_df,
											 cat_names=cat_names,
											 cont_names=cont_names,
											 processor=src.train.x.processor))

		# Step 4: Convert to databunch
		return DashDatabunch.create_databunch(response, src)
