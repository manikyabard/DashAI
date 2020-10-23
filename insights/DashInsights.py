#!/usr/bin/env python3
from fastai.text import *
from fastai.text.data import TokenizeProcessor, NumericalizeProcessor
from fastai.text.transform import Tokenizer

from captum.attr import configure_interpretable_embedding_layer
from captum.insights import Batch
from captum.insights.attr_vis.features import TextFeature, ImageFeature, TabularFeature


class DashInsights:
	limit = None
	def __init__(
			self, path, bs, learn, application,
			baseline_func_default=False, baseline_token=None, baseline_fn=None
	):
		self.path = path if path else None
		self.bs = bs if bs else 8
		self.data = learn.data
		self.model = learn.model.eval()
		self.application = application
		self.baseline_func_default = baseline_func_default
		if self.application == 'text':
			self.vocab = learn.data.x.vocab
		self.baseline_token = (
			baseline_token if baseline_token
			else '.' if self.application == 'text'
			else 0
		)
		self.baseline_fn = baseline_fn
		self.features = (
			[
				TextFeature(
					"Sentence",
					baseline_transforms=(
						[self.baseline_fn] if self.baseline_fn
						else None if self.baseline_func_default
						else [self.baseline_func]
					),
					visualization_transform=self.itos,
					input_transforms=[]
				)
			] if self.application == 'text'
			else [
				ImageFeature(
					'Image',
					baseline_transforms=(
						[self.baseline_fn] if self.baseline_fn
						else None if self.baseline_func_default
						else [self.baseline_func]
					),
					input_transforms=[]
				)
			] if self.application == 'vision'
			else [
				TabularFeature(
					'Table',
					baseline_transforms=(
						[self.baseline_fn] if self.baseline_fn
						else None if self.baseline_func_default
						else [self.baseline_func]
					)
				)
			]
		)

	# @staticmethod
	# def get_processors_for_lm():
	# 	tokenizer = Tokenizer(post_rules=[replace_all_caps, deal_caps, DashInsights.limit_tokens])
	# 	vocab = None
	# 	return [
	# 		TokenizeProcessor(tokenizer),
	# 		NumericalizeProcessor(vocab)
	# 	]

	# @staticmethod
	# def get_processors_for_clas(vocab):
	# 	tokenizer = Tokenizer(post_rules=[replace_all_caps, deal_caps, DashInsights.limit_tokens])
	# 	return [
	# 		TokenizeProcessor(tokenizer=tokenizer),
	# 		NumericalizeProcessor(vocab=vocab)
	# 	]

	@staticmethod
	def get_processors_for_lm():
		tokenizer = Tokenizer(post_rules=[replace_all_caps, deal_caps, DashInsights.limit_tokens])
		vocab = None
		return [
			TokenizeProcessor(tokenizer),
			NumericalizeProcessor(vocab)
		]

	@staticmethod
	def get_processors_for_clas(vocab):
		tokenizer = Tokenizer(post_rules=[replace_all_caps, deal_caps, DashInsights.limit_tokens])
		return [
			TokenizeProcessor(tokenizer=tokenizer),
			NumericalizeProcessor(vocab=vocab)
		]

	@staticmethod
	def limit_tokens(x: list) -> list:
		# Under the assumption that the first input is the longest
		if not DashInsights.limit:
			DashInsights.limit = len(x)
		limit = DashInsights.limit
		if len(x) > limit:
			x = x[:limit]
		return x

	@staticmethod
	def score_func(o):
		if isinstance(o, tuple):
			o = o[0]
		return F.softmax(o, dim=1)

	def stoi(self, token):
		return self.vocab.stoi[token]

	def itos(self, input):
		return [self.vocab.itos[int(i)] for i in input.squeeze(0)]

	def dataset(self, texts, targets):
		for text, target in zip(texts, targets):
			t, t_len = self.encode_text(text)
			t, t_len = t.unsqueeze(0), t_len.unsqueeze(0)
			target_idx = self.vocab.stoi[target]

			yield Batch(
				inputs=(t,), labels=(target_idx,), additional_args=t_len
			)

	def encode_text(self, text):
		text_arr = text.lower().split()
		vec = torch.zeros(len(text_arr), device=torch.device('cpu')).long()
		for i, token in enumerate(text_arr):
			index = self.stoi(token)
			vec[i] = index
		return vec, torch.tensor(len(text_arr), device=torch.device('cpu'))

	def baseline_func(self, input):
		if self.application == 'text':
			baseline = torch.ones_like(input) * self.vocab.stoi[self.baseline_token]
			baseline[0] = self.vocab.stoi['xxbos']
		elif self.application == 'vision':
			baseline = input * self.baseline_token
		elif self.application == 'tabular':
			# baseline = (torch.zeros(len(self.data.x.cat_names)), torch.zeros(len(self.data.x.cont_names)))
			baseline = torch.zeros_like(input)
		return baseline

	def formatted_data_iter(self):
		dataloader = self.data
		if self.application == 'tabular':
			dataloader.batch_size = 1
			while True:
				inputs, labels = dataloader.one_batch()
				yield Batch(inputs=tuple(inputs), labels=labels)
				# if DashInsights.limit:
				# 	self.model[0].bptt = DashInsights.limit
		else:
			dataloader.batch_size = self.bs
			while True:
				inputs, labels = dataloader.one_batch()
				for input, label in zip(inputs, labels):
					yield Batch(inputs=input.unsqueeze(0), labels=label.unsqueeze(0))

	def configure_embedding_layer(self):
		interp_embs = []
		for layer in self.model:
			if isinstance(layer, nn.Embedding):
				interp_embs.append(configure_interpretable_embedding_layer(self.model, layer))
				break
