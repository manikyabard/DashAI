from tabular.learner import DashTabularLearner
from core.train import DashTrain
import json

with open('data/response.json') as f:
	response = json.load(f)
learn = DashTabularLearner.create_tabular_learner(response)

with open('./data/train.json') as f:
	train_response = json.load(f)
DashTrain.fit_one_cycle(train_response, learn)
