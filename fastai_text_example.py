#!/usr/bin/env python3
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from captum.insights import AttributionVisualizer, Batch
from captum.insights.features import ImageFeature, TextFeature

from fastai.text import *
from fastai.text.transform import Tokenizer, Vocab
from fastai.text.data import TokenizeProcessor, NumericalizeProcessor
import fastai.train

path = untar_data(URLs.IMDB_SAMPLE)
bs = 16


def get_processors_for_lm():
	tokenizer = Tokenizer(post_rules=[replace_all_caps, deal_caps, limit_tokens])
	vocab = None
	return [
		TokenizeProcessor(tokenizer),
		NumericalizeProcessor(vocab)
	]


def get_processors_for_clas(vocab):
	tokenizer = Tokenizer(post_rules=[replace_all_caps, deal_caps, limit_tokens])
	return [
		TokenizeProcessor(tokenizer=tokenizer),
		NumericalizeProcessor(vocab=vocab)
	]


def limit_tokens(x: list) -> list:
	limit = 70
	if len(x) > limit:
		x = x[:limit]
	return x


data_lm = (TextList.from_csv(path, 'texts.csv', cols='text', processor=get_processors_for_lm())
		   .split_by_rand_pct()
		   .label_for_lm()
		   .databunch(bs=bs)
		   )
data_clas = (TextList.from_csv(path, 'texts.csv', cols='text', vocab=data_lm.vocab, processor=get_processors_for_clas(data_lm.vocab))
			 .split_by_rand_pct()
			 .label_from_df(cols='label')
			 .transform()
			 .databunch(bs=bs)
			 )

lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3, pretrained=False)
# lm.unfreeze()
# lm.fit_one_cycle(1, 1e-3, moms=(0.8, 0.7))
lm.save_encoder('lm_encoder')

awd = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, pretrained=False)
awd.load_encoder('lm_encoder')
# awd.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
# awd.freeze_to(-2)
# awd.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
# awd.freeze_to(-3)
# awd.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
# awd.unfreeze()
# awd.fit_one_cycle(1, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))

# awd.export('/content/awd.pth')
fastai.torch_core.defaults.device = 'cpu'
# awd = fastai.train.load_learner('.','/content/awd.pth')
awd.data = data_clas
awd.model.eval()

vocab = awd.data.x.vocab


def get_classes():
	classes = [
		"Negative",
		"Positive"
	]
	return classes


def get_pretrained_model():
	path = untar_data(URLs.IMDB_SAMPLE)
	bs = 16
	data_lm = (TextList.from_csv(path, 'texts.csv', cols='text', processor=get_processors_for_lm())
			   .split_by_rand_pct()
			   .label_for_lm()
			   .databunch(bs=bs)
			   )
	data_clas = (TextList.from_csv(path, 'texts.csv', cols='text', vocab=data_lm.vocab,
								   processor=get_processors_for_clas(data_lm.vocab))
				 .split_by_rand_pct()
				 .label_from_df(cols='label')
				 .transform()
				 .databunch(bs=bs)
				 )
	
	lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3, pretrained=False)
	# lm.unfreeze()
	# lm.fit_one_cycle(1, 1e-3, moms=(0.8, 0.7))
	lm.save_encoder('lm_encoder')
	
	awd = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, pretrained=False)
	awd.load_encoder('lm_encoder')
	# awd.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
	# awd.freeze_to(-2)
	# awd.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
	# awd.freeze_to(-3)
	# awd.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
	# awd.unfreeze()
	# awd.fit_one_cycle(1, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
	
	# awd.export('/content/awd.pth')
	fastai.torch_core.defaults.device = 'cpu'
	# awd = fastai.train.load_learner('.','/content/awd.pth')
	awd.data = data_clas
	awd.model.eval()
	
	return awd.model


def stoi(token):
	return vocab.stoi[token]


def awd_dataset(texts, targets, awd):
	for text, target in zip(texts, targets):
		t, t_len = encode_text(text)
		t, t_len = t.unsqueeze(0), t_len.unsqueeze(0)
		target_idx = awd.data.x.vocab.stoi[target]

		yield Batch(
			inputs=(t,), labels=(target_idx,), additional_args=t_len
		)


def encode_text(text):
	text_arr = text.lower().split()
	vec = torch.zeros(len(text_arr), device=torch.device('cpu')).long()
	for i, token in enumerate(text_arr):
		index = stoi(token)
		vec[i] = index
	return vec, torch.tensor(len(text_arr), device=torch.device('cpu'))


def baseline_text(t):
	baseline = torch.ones_like(t) * vocab.stoi['.']
	baseline[0, 0] = vocab.stoi['xxbos']
	return baseline


def score_func(o):
	if isinstance(o, tuple):
		o = o[0]
	return F.softmax(o, dim=1)


def itos(input):
	return [vocab.itos[int(i)] for i in input.squeeze(0)]


# def input_text_transform(x):
# 	return interp_emb.indices_to_embeddings(x)


# def baseline_func(sentence):
# 	sentence_tokens = awd.data.one_item(sentence)
# 	reversed_tokens = [vocab.itos[w] for w in sentence_tokens[0].tolist()[0]]
# 	baseline = torch.ones_like(sentence_tokens[0]) * vocab.stoi['.']
# 	baseline[0, 0] = vocab.stoi['xxbos']  # beginning of sentence is always #1
# 	return baseline


def baseline_func_str(sentence):
	baseline = torch.ones_like(sentence) * vocab.stoi['.']
	baseline[0] = vocab.stoi['xxbos']
	return baseline


# def formatted_data_iter():
# 	dataloader = awd.data
# 	while True:
# 		sentences, labels = dataloader.one_batch()
# 		yield Batch(inputs=sentences, labels=labels)
# 		# for sentence, label in zip(sentences, labels):
# 		# 	yield Batch(inputs=sentence, labels=label)


def formatted_data_iter():
	dataloader = awd.data
	dataloader.batch_size = 64
	while True:
		sentences, labels = dataloader.one_batch()
		for sentence, label in zip(sentences, labels):
			yield Batch(inputs=sentence.unsqueeze(0), labels=label.unsqueeze(0))


def main():
	model = get_pretrained_model()
	visualizer = AttributionVisualizer(
		models=[model],
		score_func=score_func,
		classes=get_classes(),
		features=[
			TextFeature(
				"Sentence",
				baseline_transforms=[baseline_func_str],
				visualization_transform=itos,
				input_transforms=[]
			)
		],
		dataset=formatted_data_iter(),
	)
	
	visualizer.serve(debug=True)


if __name__ == "__main__":
	main()
