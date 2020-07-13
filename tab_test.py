from tabular.learner import DashTabularLearner
import json

with open('./data/response_new.json') as f:
	response = json.load(f)
print('So far, so good...')
learn = DashTabularLearner.create_tabular_learner(response)
print('Yay!')

print(learn.loss_func)
learn.fit_one_cycle(2)
