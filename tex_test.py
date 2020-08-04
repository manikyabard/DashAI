from text.learner import DashTextLearner
import json
import path
from core.train import DashTrain

path = path.Path('./')
with open('./data/response_new.json') as f:
	response = json.load(f)
learn = DashTextLearner.create_text_learner(response)
print('created learner')


# print(learn.loss_func)
# learn.fit_one_cycle(1)

with open('./data/train.json') as f:
	train_response = json.load(f)
# DashTrain.fit_one_cycle(train_response, learn)
# DashTrain.lr_find(train_response, learn)


