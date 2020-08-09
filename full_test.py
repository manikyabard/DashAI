#!/usr/bin/env python3
import json
from pathlib import Path

from text.learner import DashTextLearner
from tabular.learner import DashTabularLearner
from vision.learner import DashVisionLearner
from collabfilter.learner import DashCollabLearner
from core.train import DashTrain
from insights.DashInsights import DashInsights
from captum.insights.attr_vis import AttributionVisualizer

path = Path('./')
learner_class_map = {
	'collab': DashCollabLearner,
	'tabular': DashTabularLearner,
	'text': DashTextLearner,
	'vision': DashVisionLearner
}


def main():
	print('STEP 1: Creating the learner.')
	with open('./data/response.json') as f:
		response = json.load(f)
	application = response['task']
	learner_class = learner_class_map[application]
	learn = getattr(learner_class, f'create_{application}_learner')(response)
	print('Created learner; completed step 1.')

	# print('STEP 2 (optional): Optimizing the hyper-parameters.')
	# step_2 = False
	# try:
	# 	import ax
	# 	from verum.DashVerum import DashVerum
	# 	step_2 = True
	# 	with open('./data/verum.json') as f:
	# 		response = json.load(f)
	# 	verum = DashVerum(response, learn)
	# 	learn, metric, lr, num_epochs, moms = verum.veritize()
	# 	print('Hyper-parameters optimized; completed step 2.')
	# except ImportError:
	# 	print('Skipping step 2 as module `ax` is not installed.')
	#
	# print('STEP 3: Training the model.')
	# with open('./data/train.json') as f:
	# 	response = json.load(f)
	# if step_2:
	# 	response['fit']['epochs'] = num_epochs
	# 	response['fit']['lr'] = lr
	# 	response['fit_one_cycle']['max_lr'] = lr
	# 	response['fit_one_cycle']['moms'] = str(moms)
	#
	# getattr(DashTrain, response['type'])(response, learn)
	# print('Trained model; completed step 3.')

	print('STEP 4: Visualizing the attributions.')
	insights = DashInsights(path, learn.data.batch_size, learn, application)
	visualizer = AttributionVisualizer(
		models=[insights.model],
		score_func=insights.score_func,
		classes=insights.data.classes,
		features=insights.features,
		dataset=insights.formatted_data_iter(),
		application=insights.application
	)

	visualizer.serve()
	print('Completed visualization; completed step 4. Congratulations!')
	print('Now we need to add model-saving and production-serving.')


if __name__ == '__main__':
	main()
