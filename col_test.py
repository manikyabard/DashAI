from collabfilter.learner import DashCollabLearner
import json

with open('./data/response_joe.json') as f:
	response = json.load(f)
print('So far, so good...')
learn = DashCollabLearner.create_collab_learner(response)
print(learn)
learn.fit_one_cycle(1)
print('Yay!')
