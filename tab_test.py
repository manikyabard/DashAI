from tabular.learner import DashTabularLearner
from core.train import DashTrain
import json

with open('./data/response_new.json') as f:
	response = json.load(f)
learn = DashTabularLearner.create_tabular_learner(response)

# print(learn.loss_func)
# print(learn)
# learn.fit_one_cycle(2)
with open('./data/train.json') as f:
	train_response = json.load(f)
DashTrain.fit_one_cycle(train_response, learn)
