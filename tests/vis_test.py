from vision.learner import DashVisionLearner
import json
import path

path = path.Path('./')
with open('data/response.json') as f:
	response = json.load(f)
learn = DashVisionLearner.create_vision_learner(response)

print('Created learner!')
print('Now training...')
learn.fit_one_cycle(1)
print('Done!')