from tabular.learner import DashTabularLearner
import json

with open('./data/response_new.json') as f:
	response = json.load(f)
learn = DashTabularLearner.create_tabular_learner(response)

print(learn.loss_func)
learn.fit_one_cycle(2)
