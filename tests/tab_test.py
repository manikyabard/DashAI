from tabular.learner import DashTabularLearner
from core.train import DashTrain
import json

with open('data/response.json') as f:
	response = json.load(f)
learn = DashTabularLearner.create_tabular_learner(response)

# learn.fit_one_cycle(2)
with open('./data/train.json') as f:
	train_response = json.load(f)
DashTrain.fit_one_cycle(train_response, learn)
