from fastai.text import *
from core.databunch import DashDatabunch
from fastai.data_block import *


# Options for tokenizer and numericalizer not given

class DashTextDatabunch:
	
	@staticmethod
	def create_text_databunch(response):
		if response['text']['model']['type'] == 'classifier':
			response_lm = response
			response_lm['core']['data']['label']['method'] = 'for_lm'
			
			# The state needs to be the same during creation of the
			# two `DataBunch` objects for a classification task to
			# prevent errors.
			state = np.random.get_state()
			np.random.set_state(state)
			src_lm = DashTextDatabunch.get_itemlist(response_lm["text"]["input"])
			src_lm = DashDatabunch.split_databunch(response_lm, src_lm)
			src_lm = DashDatabunch.label_databunch(response_lm, src_lm)
			src = DashTextDatabunch.get_itemlist(response["text"]["input"])
			src = DashDatabunch.split_databunch(response, src)
			src = DashDatabunch.label_databunch(response, src)
			np.random.seed(seed=None)
			
			return (DashDatabunch.create_databunch(response, src),
					DashDatabunch.create_databunch(response_lm, src_lm))
		
		src = DashTextDatabunch.get_itemlist(response["text"]["input"])
		src = DashDatabunch.split_databunch(response, src)
		src = DashDatabunch.label_databunch(response, src)
		
		# Add transforms and test
		
		return DashDatabunch.create_databunch(response, src)
	
	@staticmethod
	def get_itemlist(response):
		if response["method"] == "from_csv":
			return TextList.from_csv(**response["from_csv"])
		if response["method"] == "from_folder":
			return TextList.from_folder(**response["from_folder"])
