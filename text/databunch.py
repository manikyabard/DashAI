from fastai.text import *
from core.databunch import DashDatabunch
from fastai.data_block import *


# Options for tokenizer and numericalizer not given

class DashTextDatabunch:

	@staticmethod
	def create_text_databunch(response):
		src = DashTextDatabunch.get_itemlist(response["text"]["input"])
		src = DashDatabunch.split_databunch(response, src)
		src = DashDatabunch.label_databunch(response, src)

		#Add transforms and test

		return DashDatabunch.create_databunch(response, src)

	@staticmethod
	def get_itemlist(response):
		if response["method"] == "from_csv":
			return TextList.from_csv(**response["from_csv"])
		if response["method"] == "from_folder":
			return TextList.from_folder(**response["from_folder"])
