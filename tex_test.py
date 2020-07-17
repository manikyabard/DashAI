from text.learner import DashTextLearner
import json
import torch.optim
import path

path = path.Path('./')
with open('./data/response_new.json') as f:
			response = json.load(f)
learn = DashTextLearner.create_text_learner(response)
print('created learner')

learn.fit_one_cycle(1)
# databunch.show_batch(rows=2, figsize=(6,6))
print('done')