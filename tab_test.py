from tabular.learner import DashTabularLearner
import json

with open('./data/response-tab-reg-mult.json') as f:
	response = json.load(f)
print('So far, so good...')
learn = DashTabularLearner.create_tabular_learner(response)
print('Yay!')

learn.fit_one_cycle(2)
print(learn)