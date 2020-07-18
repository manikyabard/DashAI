from vision.databunch import DashVisionDatabunch
from vision.learner import DashVisionLearner
import json
import torch.optim
import path

path = path.Path('./')
with open('./data/response_new.json') as f:
	response = json.load(f)
learn = DashVisionLearner.create_vision_learner(response)
print('created learner')
print(learn)
# databunch.show_batch(rows=2, figsize=(6,6))
print('done')