#!/usr/bin/env python3
import fastai

from insights.DashInsights import DashInsights
from fastai.text import *
from fastai.vision import *
from fastai.tabular import *
from captum.insights.attr_vis import AttributionVisualizer


def main():
	# path = untar_data(URLs.IMDB_SAMPLE)
	# bs = 16
	# data_lm = (
	# 	TextList.from_csv(
	# 		path, 'texts.csv', cols='text',
	# 		processor=DashInsights.get_processors_for_lm()
	# 	)
	# 		.split_by_rand_pct()
	# 		.label_for_lm()
	# 		.databunch(bs=bs)
	# )
	# data_clas = (
	# 	TextList.from_csv(
	# 		path, 'texts.csv', cols='text', vocab=data_lm.vocab,
	# 		processor=DashInsights.get_processors_for_clas(data_lm.vocab)
	# 	)
	# 		.split_by_rand_pct()
	# 		.label_from_df(cols='label')
	# 		.transform()
	# 		.databunch(
	# 			bs=bs,
	# 			collate_fn=partial(
	# 				pad_collate, pad_idx=1, pad_first=False, backwards=False
	# 			)
	# 		)
	# )
	# lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3, pretrained=False)
	# lm.save_encoder('lm_encoder')
	# awd = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, pretrained=False)
	# awd.load_encoder('lm_encoder')
	# fastai.torch_core.defaults.device = 'cpu'
	
	path = untar_data(URLs.MNIST_SAMPLE)
	bs = 2
	data = ImageDataBunch.from_folder(path, valid_pct=0.2, size=28)
	learn = cnn_learner(data, models.resnet18, metrics=error_rate)

	# path = untar_data(URLs.ADULT_SAMPLE)
	# bs = 16
	# df = pd.read_csv(path / 'adult.csv')
	# procs = [FillMissing, Categorify, Normalize]
	# valid_idx = range(len(df) - 2000, len(df))
	# dep_var = 'salary'
	# cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
	# 			 'native-country']
	# data = TabularDataBunch.from_df(path, df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)
	# learn = tabular_learner(data, layers=[200, 100], emb_szs={'native-country': 10}, metrics=accuracy)

	insights_obj = DashInsights(path, bs, learn, 'vision')
	visualizer = AttributionVisualizer(
		models=[insights_obj.model],
		score_func=insights_obj.score_func,
		classes=insights_obj.data.classes,
		features=insights_obj.features,
		dataset=insights_obj.formatted_data_iter(),
		application= insights_obj.application
	)

	visualizer.serve(debug=True)


if __name__ == "__main__":
	main()
