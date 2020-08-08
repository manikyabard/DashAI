#!/usr/bin/env python3
import json
import path

from text.learner import DashTextLearner
from tabular.learner import DashTabularLearner
from vision.learner import DashVisionLearner
from collabfilter.learner import DashCollabLearner

path = path.Path('./')
with open('data/response.json') as f:
	response = json.load(f)
learn = DashTextLearner.create_text_learner(response)
print('created learner')

print('done')
