from fastai.collab import *
from core.databunch import DashDatabunch


class DashCollabDatabunch:
	"""
	Base DataBunch for collaborative filtering.
	"""

	@staticmethod
	def create_collab_databunch(response):
		"""
		Creates a databunch for collaborative filtering using the values specified in response/DashUI.
		Uses the csv name given in response>collab>input to create the input dataframe.
		Creates a cat list using the user name and item name provided in the data.
		Applies the required transformations and creates a collab list using the specified cat list.
		The collab list is split into training and validation sets using the specified method and then labelled.
		Finally it is converted to a databunch object and returned.
		"""
		path = Path('./')
		# Step 1: Provide inputs
		response_col = response['collab']
		df = pd.read_csv(path / f'{response_col["input"]["csv_name"]}')
		procs = list()
		# print(f"{response_col['transform']['FillMissing']['fill_strategy']}")
		if response_col["transform"]['FillMissing']:
			if hasattr(FillStrategy, f"{response_col['transform']['FillMissing']['fill_strategy']}"):
				fill_strategy = getattr(FillStrategy, f"{response_col['transform']['FillMissing']['fill_strategy']}")

			procs.append(partial(FillMissing, fill_strategy=fill_strategy,
								 fill_val=response_col['transform']['FillMissing']['fill_val'],
								 add_col=response_col['transform']['FillMissing']['add_col']))

		if response_col['transform']['Categorify']:
			procs.append(Categorify)

		if response_col['transform']['Normalize']:
			procs.append(Normalize)

		procs = listify(procs)

		user_name = response_col['input']['user_name']
		item_name = response_col['input']['item_name']
		rating = response_col['input']['rating']
		cat_names = [user_name, item_name]
		src = CollabList.from_df(df, cat_names=cat_names, procs=procs)
		src = DashDatabunch.split_databunch(response, src)
		# src = DashDatabunch.label_databunch(response, src)
		# src=src.split_by_rand_pct(valid_pct=0.2,seed=None)
		src = src.label_from_df(cols=rating)
		# if test is not None: src.add_test(CollabList.from_df(test, cat_names=cat_names))

		if response["collab"]["input"]['test_df']["has_test"]:
			test_df = pd.read_csv(path / f"{response['collab']['input']['test_df']['csv_name']}")
			src.add_test(CollabList.from_df(test_df, cat_names=cat_names))

		return DashDatabunch.create_databunch(response, src)
