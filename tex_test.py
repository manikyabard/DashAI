from text.learner import DashTextLearner
import json
import path

path = path.Path('./')
with open('data/response.json') as f:
	response = json.load(f)
learn = DashTextLearner.create_text_learner(response)
print('created learner')

learn.fit_one_cycle(1)
print('done')
